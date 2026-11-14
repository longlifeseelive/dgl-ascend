"""CSRToCOO correctness and msprof performance tests on Ascend NPU.

Correctness:
    python tests/ascend/test_csr_to_coo_npu.py

The unittest class can also be collected by pytest when pytest is installed.

Profile every performance case in a separate msprof collection:
    python tests/ascend/test_csr_to_coo_npu.py --run-msprof \
        --msprof-output ./msprof_csr_to_coo

Profile one case directly:
    msprof --output=./msprof_csr_to_coo/typical_false \
        --ascendcl=on --runtime-api=on --task-time=l2 --ai-core=on \
        --aic-metrics=PipeUtilization \
        python tests/ascend/test_csr_to_coo_npu.py \
        --performance --perf-case typical_false

The public DGL calls intentionally exercise both C++ branches:

* graph.edges(form="all", order=None) on a CSR graph calls
  CSRToCOO(csr, false) and returns COO row/col/data.
* graph.adj_tensors("coo") on a fresh CSR graph calls
  CSRToCOO(csr, true) and returns COO row/col. COO data is empty because the
  input CSR data is consumed as the output order.

A fresh graph is used for every data_as_order=True conversion so that DGL's
cached COO representation cannot bypass CSRToCOO.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import unittest
from unittest import mock
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import dgl
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


CANONICAL_ETYPE = ("_U", "_E", "_V")
ID_DTYPES = (torch.int32, torch.int64)


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


NPU_AVAILABLE = _npu_is_available()


@dataclass(frozen=True)
class CSRCase:
    """A deterministic CSR test specification stored directly in this file."""

    name: str
    num_rows: int
    num_cols: int
    nnz: int
    data_as_orders: Tuple[bool, ...]
    with_data: bool = True
    row_pattern: str = "balanced"
    col_pattern: str = "mixed"
    data_pattern: str = "identity"
    seed: int = 20261114


# Functional matrix requested for CSRToCOO. Shapes are explicit and no external
# graph files or case configurations are used.
FUNCTIONAL_CASES = (
    CSRCase(
        "long_tail_row",
        1024,
        1024,
        8192,
        (False, True),
        row_pattern="long_tail",
    ),
    CSRCase("typical", 10000, 10000, 500000, (False, True)),
    CSRCase("stress", 100000, 100000, 2000000, (False, True)),
    CSRCase(
        "single_sparse_row",
        1,
        4096,
        1024,
        (False, True),
        col_pattern="single_row_sparse",
    ),
    CSRCase(
        "repeated_single_column",
        4096,
        1,
        8192,
        (False, True),
        col_pattern="single_column",
    ),
    CSRCase(
        "unsorted_within_row",
        4096,
        4096,
        65536,
        (False, True),
        col_pattern="descending",
    ),
    CSRCase(
        "reverse_edge_ids",
        4096,
        4096,
        65536,
        (False, True),
        data_pattern="reverse",
    ),
    CSRCase(
        "sampled_subgraph",
        12000,
        2000,
        60000,
        (False, True),
        row_pattern="sampled",
    ),
    CSRCase(
        "without_data",
        4096,
        4096,
        65536,
        (False,),
        with_data=False,
    ),
    CSRCase(
        "valid_reverse_permutation",
        4096,
        4096,
        65536,
        (True,),
        data_pattern="reverse",
    ),
    CSRCase(
        "valid_cyclic_permutation",
        4096,
        4096,
        65536,
        (True,),
        data_pattern="cyclic",
    ),
    CSRCase(
        "self_loops",
        4096,
        4096,
        4096,
        (False, True),
        col_pattern="self_loop",
    ),
)


# Performance matrix requested for msprof. All performance cases use int32 IDs.
PERFORMANCE_CASES = {
    "valid_reverse_true": ("valid_reverse_permutation", True),
    "typical_false": ("typical", False),
    "typical_true": ("typical", True),
    "sampled_subgraph_false": ("sampled_subgraph", False),
    "sampled_subgraph_true": ("sampled_subgraph", True),
}
CASES_BY_NAME = {case.name: case for case in FUNCTIONAL_CASES}

# Additional, independent workloads for input-distribution and scaling audits.
# These do not change the original functional/performance matrices.
GENERALIZATION_CASES = (
    CSRCase("gen_empty_graph", 0, 17, 0, (False, True), with_data=False),
    CSRCase("gen_empty_rows", 8193, 257, 0, (False, True), with_data=False),
    CSRCase("gen_single_edge", 31, 47, 1, (False, True), row_pattern="needle_last"),
    CSRCase("gen_tiny_odd", 37, 91, 113, (False, True)),
    CSRCase("gen_small_odd", 257, 509, 4099, (False, True)),
    CSRCase("gen_single_dense_row", 1, 32749, 131073, (False, True)),
    CSRCase("gen_few_dense_rows", 7, 32749, 131071, (False, True)),
    CSRCase("gen_degree_128", 1009, 4093, 129152, (False, True)),
    CSRCase("gen_degree_128_plus", 1009, 4093, 129153, (False, True)),
    CSRCase("gen_uniform_medium", 16381, 12289, 262147, (False, True)),
    CSRCase("gen_uniform_large", 250003, 5003, 3000017, (False, True)),
    CSRCase("gen_sparse_million", 1000003, 509, 65539, (False, True),
            row_pattern="sparse_scattered"),
    CSRCase("gen_hub_first", 16381, 12289, 524309, (False, True),
            row_pattern="hub_first"),
    CSRCase("gen_hub_middle", 16381, 12289, 524309, (False, True),
            row_pattern="hub_middle"),
    CSRCase("gen_hub_last", 16381, 12289, 524309, (False, True),
            row_pattern="hub_last"),
    CSRCase("gen_power_law", 32771, 32749, 500009, (False, True),
            row_pattern="power_law"),
    CSRCase("gen_cluster_first", 65539, 4093, 262147, (False, True),
            row_pattern="cluster_first"),
    CSRCase("gen_cluster_last", 65539, 4093, 262147, (False, True),
            row_pattern="cluster_last"),
    CSRCase("gen_without_data", 16381, 12289, 262147, (False, True),
            with_data=False),
    CSRCase("gen_perm_identity", 8191, 4093, 131101, (False, True)),
    CSRCase("gen_perm_reverse", 8191, 4093, 131101, (False, True),
            data_pattern="reverse"),
    CSRCase("gen_perm_cyclic", 8191, 4093, 131101, (False, True),
            data_pattern="cyclic"),
    CSRCase("gen_perm_blocks", 8191, 4093, 131101, (False, True),
            data_pattern="block_shuffle"),
    CSRCase("gen_perm_pairs", 8191, 4093, 131101, (False, True),
            data_pattern="pair_swap"),
    CSRCase("gen_perm_stride", 8191, 4093, 131101, (False, True),
            data_pattern="stride"),
    CSRCase("gen_perm_random", 8191, 4093, 131101, (False, True),
            data_pattern="random"),
    CSRCase("gen_random_large", 32771, 32749, 1000003, (False, True),
            data_pattern="random"),
    CSRCase("gen_random_small", 4099, 1009, 65539, (False, True),
            data_pattern="random"),
    CSRCase("gen_needle_sparse", 262147, 257, 1, (False, True),
            row_pattern="needle_last"),
    CSRCase("gen_single_column", 131071, 1, 524309, (False, True),
            col_pattern="single_column"),
    CSRCase("gen_many_columns", 31, 1000003, 524309, (False, True)),
    CSRCase("gen_dense_medium", 512, 4093, 1000003, (False, True)),
)


def _balanced_counts(num_rows: int, nnz: int) -> torch.Tensor:
    if num_rows == 0:
        assert nnz == 0
        return torch.zeros(0, dtype=torch.int64)
    quotient, remainder = divmod(nnz, num_rows)
    counts = torch.full((num_rows,), quotient, dtype=torch.int64)
    counts[:remainder] += 1
    return counts


def _row_counts(case: CSRCase) -> torch.Tensor:
    if case.row_pattern == "balanced":
        return _balanced_counts(case.num_rows, case.nnz)

    if case.row_pattern == "long_tail":
        # Every row is non-empty and one hot row owns 75% of the remaining
        # entries, producing a deliberately skewed degree distribution.
        assert case.nnz >= case.num_rows
        counts = torch.ones(case.num_rows, dtype=torch.int64)
        remaining = case.nnz - case.num_rows
        hot = remaining * 3 // 4
        counts[case.num_rows // 2] += hot
        tail = remaining - hot
        quotient, remainder = divmod(tail, case.num_rows)
        counts += quotient
        counts[:remainder] += 1
        return counts

    if case.row_pattern == "sampled":
        # 80% of the edges are concentrated in the first 10% of source rows.
        hot_rows = max(1, case.num_rows // 10)
        hot_nnz = case.nnz * 4 // 5
        counts = torch.zeros(case.num_rows, dtype=torch.int64)
        counts[:hot_rows] = _balanced_counts(hot_rows, hot_nnz)
        cold_rows = case.num_rows - hot_rows
        if cold_rows:
            counts[hot_rows:] = _balanced_counts(
                cold_rows, case.nnz - hot_nnz
            )
        return counts

    generator = torch.Generator().manual_seed(case.seed)
    if case.row_pattern == "needle_last":
        counts = torch.zeros(case.num_rows, dtype=torch.int64)
        counts[-1] = case.nnz
        return counts
    if case.row_pattern in ("hub_first", "hub_middle", "hub_last"):
        hot = case.nnz * 9 // 10
        counts = _balanced_counts(case.num_rows, case.nnz - hot)
        position = {"hub_first": 0, "hub_middle": case.num_rows // 2,
                    "hub_last": case.num_rows - 1}[case.row_pattern]
        counts[position] += hot
        return counts
    if case.row_pattern in ("sparse_scattered", "cluster_first", "cluster_last"):
        divisor = 1000 if case.row_pattern == "sparse_scattered" else 100
        active = max(1, case.num_rows // divisor)
        counts = torch.zeros(case.num_rows, dtype=torch.int64)
        if case.row_pattern == "sparse_scattered":
            positions = torch.randperm(case.num_rows, generator=generator)[:active]
        elif case.row_pattern == "cluster_first":
            positions = torch.arange(active)
        else:
            positions = torch.arange(case.num_rows - active, case.num_rows)
        counts[positions] = _balanced_counts(active, case.nnz)
        return counts
    if case.row_pattern == "power_law":
        weights = torch.arange(1, case.num_rows + 1, dtype=torch.float64).pow(-1.3)
        samples = torch.multinomial(weights, case.nnz, replacement=True,
                                    generator=generator)
        counts = torch.bincount(samples, minlength=case.num_rows)
        return counts[torch.randperm(case.num_rows, generator=generator)]

    raise ValueError(f"Unknown row pattern: {case.row_pattern}")


def _column_indices(case: CSRCase) -> torch.Tensor:
    positions = torch.arange(case.nnz, dtype=torch.int64)
    if case.col_pattern == "single_column":
        return torch.zeros(case.nnz, dtype=torch.int64)
    if case.col_pattern == "descending":
        return case.num_cols - 1 - positions.remainder(case.num_cols)
    if case.col_pattern == "single_row_sparse":
        return (positions * 3 + 1).remainder(case.num_cols)
    if case.col_pattern == "self_loop":
        assert case.num_rows == case.num_cols == case.nnz
        return positions
    if case.col_pattern == "mixed":
        # Deterministic non-monotonic columns without RNG state.
        return (positions * 104729 + 8191).remainder(case.num_cols)
    raise ValueError(f"Unknown column pattern: {case.col_pattern}")


def _edge_ids(case: CSRCase) -> Optional[torch.Tensor]:
    if not case.with_data:
        return None
    edge_ids = torch.arange(case.nnz, dtype=torch.int64)
    if case.data_pattern == "identity":
        return edge_ids
    if case.data_pattern == "reverse":
        return edge_ids.flip(0)
    if case.data_pattern == "cyclic":
        return edge_ids.roll(case.nnz // 3)
    generator = torch.Generator().manual_seed(case.seed + 17)
    if case.data_pattern == "random":
        return torch.randperm(case.nnz, generator=generator)
    if case.data_pattern == "pair_swap":
        paired = case.nnz // 2 * 2
        edge_ids[:paired] = edge_ids[:paired].reshape(-1, 2).flip(1).reshape(-1)
        return edge_ids
    if case.data_pattern == "stride":
        stride = 17
        while math.gcd(stride, case.nnz) != 1:
            stride += 2
        return (edge_ids * stride).remainder(case.nnz)
    if case.data_pattern == "block_shuffle":
        blocks = (case.nnz + 255) // 256
        order = torch.randperm(blocks, generator=generator)
        return torch.cat([edge_ids[int(b) * 256:min((int(b) + 1) * 256, case.nnz)]
                          for b in order])
    raise ValueError(f"Unknown data pattern: {case.data_pattern}")


def _make_csr(
    case: CSRCase, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Materialize one deterministic, valid CSR input on CPU."""
    counts = _row_counts(case)
    assert counts.numel() == case.num_rows
    assert int(counts.sum()) == case.nnz

    indptr = torch.empty(case.num_rows + 1, dtype=torch.int64)
    indptr[0] = 0
    indptr[1:] = counts.cumsum(0)
    indices = _column_indices(case)
    data = _edge_ids(case)

    assert int(indptr[-1]) == case.nnz
    assert indices.shape == (case.nnz,)
    assert not indices.numel() or (
        int(indices.min()) >= 0 and int(indices.max()) < case.num_cols
    )

    return (
        indptr.to(dtype),
        indices.to(dtype),
        None if data is None else data.to(dtype),
    )


def _copy_csr_to(
    csr: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr, indices, data = csr
    empty_or_data = (
        torch.empty(0, dtype=dtype, device=device)
        if data is None
        else data.to(device)
    )
    return indptr.to(device), indices.to(device), empty_or_data


def _make_dgl_csr_graph(
    case: CSRCase,
    csr: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
) -> dgl.DGLGraph:
    indptr, indices, data = _copy_csr_to(csr, dtype, device)
    return dgl.heterograph(
        {CANONICAL_ETYPE: ("csr", (indptr, indices, data))},
        num_nodes_dict={"_U": case.num_rows, "_V": case.num_cols},
        idtype=dtype,
        device=device,
    )


def _dgl_csr_to_coo(
    graph: dgl.DGLGraph, data_as_order: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Invoke CSRToCOO through DGL's native graph API."""
    if data_as_order:
        row, col = graph.adj_tensors("coo", etype=CANONICAL_ETYPE)
        return row, col, None

    row, col, data = graph.edges(
        form="all", order=None, etype=CANONICAL_ETYPE
    )
    # DGL represents a missing data array as a zero-length ID array.
    return row, col, None if data.numel() == 0 else data


def _native_csr_to_coo(case, device_csr, data_as_order):
    """Optional CSR-specific host interface; never replaces the graph path."""
    row, col, data = torch.ops.dgl_ascend.csr_to_coo(
        *device_csr, case.num_cols, data_as_order)
    return row, col, None if data.numel() == 0 else data


def _assert_same_tensor(
    label: str, npu_value: torch.Tensor, cpu_value: torch.Tensor
) -> None:
    actual = npu_value.cpu()
    assert actual.dtype == cpu_value.dtype, (
        f"{label}: dtype mismatch: NPU={actual.dtype}, CPU={cpu_value.dtype}"
    )
    assert actual.shape == cpu_value.shape, (
        f"{label}: shape mismatch: NPU={tuple(actual.shape)}, "
        f"CPU={tuple(cpu_value.shape)}"
    )
    if not torch.equal(actual, cpu_value):
        mismatch = torch.nonzero(actual != cpu_value, as_tuple=False)
        first = int(mismatch[0]) if mismatch.numel() else -1
        raise AssertionError(
            f"{label}: value mismatch at index {first}: "
            f"NPU={actual[first].item()}, CPU={cpu_value[first].item()}"
        )


def _assert_result(
    case: CSRCase,
    dtype: torch.dtype,
    data_as_order: bool,
    npu_result: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    cpu_result: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
) -> None:
    dtype_name = str(dtype).split(".")[-1]
    prefix = f"{case.name}/{dtype_name}/order={data_as_order}"
    npu_row, npu_col, npu_data = npu_result
    cpu_row, cpu_col, cpu_data = cpu_result

    assert npu_row.shape == (case.nnz,)
    assert npu_col.shape == (case.nnz,)
    assert npu_row.dtype == dtype
    assert npu_col.dtype == dtype
    _assert_same_tensor(f"{prefix}/row", npu_row, cpu_row)
    _assert_same_tensor(f"{prefix}/col", npu_col, cpu_col)

    expected_data = case.with_data and not data_as_order
    assert (npu_data is not None) == expected_data, (
        f"{prefix}/data: expected "
        f"{'an nnz-sized tensor' if expected_data else 'None'}"
    )
    assert (cpu_data is not None) == expected_data
    if expected_data:
        assert npu_data is not None and cpu_data is not None
        assert npu_data.shape == (case.nnz,)
        assert npu_data.dtype == dtype
        _assert_same_tensor(f"{prefix}/data", npu_data, cpu_data)


class TestCSRToCOONPU(unittest.TestCase):
    @unittest.skipUnless(
        NPU_AVAILABLE, "Ascend NPU is required for CSRToCOO tests"
    )
    def test_functional_matrix(self) -> None:
        """Compare native CPU and NPU CSRToCOO outputs exactly."""
        npu_device = torch.device("npu:0")
        torch.npu.set_device(npu_device)

        for case in FUNCTIONAL_CASES:
            for data_as_order in case.data_as_orders:
                for dtype in ID_DTYPES:
                    with self.subTest(
                        case=case.name,
                        data_as_order=data_as_order,
                        dtype=str(dtype).split(".")[-1],
                    ):
                        csr = _make_csr(case, dtype)
                        # Separate fresh graphs prevent a cached COO from
                        # bypassing conversion.
                        cpu_graph = _make_dgl_csr_graph(
                            case, csr, dtype, torch.device("cpu")
                        )
                        npu_graph = _make_dgl_csr_graph(
                            case, csr, dtype, npu_device
                        )
                        cpu_result = _dgl_csr_to_coo(
                            cpu_graph, data_as_order
                        )
                        npu_result = _dgl_csr_to_coo(
                            npu_graph, data_as_order
                        )
                        torch.npu.synchronize(npu_device)
                        _assert_result(
                            case,
                            dtype,
                            data_as_order,
                            npu_result=npu_result,
                            cpu_result=cpu_result,
                        )

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_native_interface(self) -> None:
        """Check the new interface on the original and distribution matrices."""
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        for case in FUNCTIONAL_CASES + GENERALIZATION_CASES:
            for dtype in ID_DTYPES:
                csr = _make_csr(case, dtype)
                device_csr = _copy_csr_to(csr, dtype, device)
                for ordered in case.data_as_orders:
                    with self.subTest(case=case.name, dtype=dtype, order=ordered):
                        graph = _make_dgl_csr_graph(
                            case, csr, dtype, torch.device("cpu"))
                        expected = _dgl_csr_to_coo(graph, ordered)
                        actual = _native_csr_to_coo(case, device_csr, ordered)
                        _assert_result(case, dtype, ordered, actual, expected)

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_native_stream_and_lifetime(self) -> None:
        """Wait for non-current producers and retain outputs after inputs die."""
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        producer = torch.npu.Stream(device=device)
        delay = torch.ones(1024 * 1024, device=device)
        history = []
        for dtype in ID_DTYPES:
            for ordered in (False, True):
                for iteration in range(4):
                    case = CSRCase(
                        "native_stream_lifetime", 257, 509, 8193, (ordered,),
                        row_pattern="hub_middle" if iteration % 2 else "balanced",
                        data_pattern="random", seed=20261114 + iteration)
                    source = _make_csr(case, dtype)
                    target = _copy_csr_to(source, dtype, device)
                    inputs = tuple(torch.zeros_like(t) for t in target)
                    torch.npu.synchronize(device)
                    with torch.npu.stream(producer):
                        for _ in range(8):
                            delay.add_(1)
                        for destination, value in zip(inputs, target):
                            destination.copy_(value)
                        ready = torch.npu.Event()
                        ready.record(producer)
                    actual = _native_csr_to_coo(case, inputs, ordered)
                    self.assertTrue(ready.query())
                    cpu_graph = _make_dgl_csr_graph(
                        case, source, dtype, torch.device("cpu"))
                    expected = _dgl_csr_to_coo(cpu_graph, ordered)
                    _assert_result(case, dtype, ordered, actual, expected)
                    history.append((case, dtype, ordered, actual, expected))
                    del inputs, target, cpu_graph
        gc.collect()
        torch.npu.empty_cache()
        for case, dtype, ordered, actual, expected in history:
            _assert_result(case, dtype, ordered, actual, expected)

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_graph_format_and_fallback_contract(self) -> None:
        """Compare graph API values and cache transitions with the old route."""
        from dgl import heterograph_index as hgi

        case = CSRCase("graph_contract", 17, 13, 113, (False, True),
                       row_pattern="hub_middle", data_pattern="random")
        for dtype in ID_DTYPES:
            source = _make_csr(case, dtype)
            for device in (torch.device("cpu"), torch.device("npu:0")):
                for storage in ("csr", "coo", "csc"):
                    with self.subTest(dtype=dtype, device=device, storage=storage):
                        def make_graph():
                            if storage == "csc":
                                arrays = _copy_csr_to(source, dtype, device)
                                return dgl.heterograph(
                                    {CANONICAL_ETYPE: ("csc", arrays)},
                                    num_nodes_dict={"_U": case.num_cols,
                                                    "_V": case.num_rows},
                                    idtype=dtype, device=device)
                            graph = _make_dgl_csr_graph(case, source, dtype, device)
                            return graph.formats("coo") if storage == "coo" else graph

                        actual_graph = make_graph()
                        with mock.patch.object(hgi, "_csr_to_coo_npu_graph",
                                               return_value=None):
                            reference_graph = make_graph()
                        self.assertEqual(actual_graph.formats(),
                                         reference_graph.formats())
                        for operation in ("edges", "coo", "edges", "eid", "coo"):
                            def invoke(graph):
                                if operation == "coo":
                                    return graph.adj_tensors(
                                        "coo", etype=CANONICAL_ETYPE)
                                return graph.edges(
                                    form="all", etype=CANONICAL_ETYPE,
                                    order="eid" if operation == "eid" else None)
                            actual = invoke(actual_graph)
                            with mock.patch.object(hgi, "_csr_to_coo_npu_graph",
                                                   return_value=None):
                                expected = invoke(reference_graph)
                            for i, (value, reference) in enumerate(zip(actual, expected)):
                                _assert_same_tensor(f"{storage}/{operation}/{i}",
                                                    value, reference.cpu())
                            self.assertEqual(actual_graph.formats(),
                                             reference_graph.formats())
                        actual = actual_graph._graph.adjacency_matrix_tensors(
                            0, True, "coo")
                        with mock.patch.object(hgi, "_csr_to_coo_npu_graph",
                                               return_value=None):
                            expected = reference_graph._graph.adjacency_matrix_tensors(
                                0, True, "coo")
                        self.assertEqual(actual[:2], expected[:2])
                        for value, reference in zip(actual[2:], expected[2:]):
                            _assert_same_tensor("transpose", value, reference.cpu())
                        if storage == "csr":
                            restricted = make_graph().formats("csr")
                            before = restricted.formats()
                            converted = hgi._csr_to_coo_npu_graph(
                                restricted._graph, 0, True)
                            self.assertIsNone(converted)
                            self.assertEqual(restricted.formats(), before)

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_graph_heterogeneous_relations(self) -> None:
        """The requested edge type, rather than relation zero, owns the CSR."""
        keys = (("_U", "first", "_V"), ("_U", "second", "_V"))
        for dtype in ID_DTYPES:
            cases = [CSRCase("relation", 17, 13, 113, (False, True),
                             data_pattern="random", seed=20261114 + i,
                             row_pattern="hub_middle" if i else "balanced")
                     for i in range(2)]
            sources = [_make_csr(c, dtype) for c in cases]
            cpu = dgl.heterograph(
                {key: ("csr", _copy_csr_to(csr, dtype, torch.device("cpu")))
                 for key, csr in zip(keys, sources)},
                num_nodes_dict={"_U": 17, "_V": 13}, idtype=dtype)
            npu = dgl.heterograph(
                {key: ("csr", _copy_csr_to(csr, dtype, torch.device("npu:0")))
                 for key, csr in zip(keys, sources)},
                num_nodes_dict={"_U": 17, "_V": 13}, idtype=dtype,
                device=torch.device("npu:0"))
            for key in reversed(keys):
                for ordered in (False, True):
                    actual = (npu.adj_tensors("coo", etype=key) if ordered else
                              npu.edges(form="all", order=None, etype=key))
                    expected = (cpu.adj_tensors("coo", etype=key) if ordered else
                                cpu.edges(form="all", order=None, etype=key))
                    for value, reference in zip(actual, expected):
                        _assert_same_tensor(f"relation/{key}/{ordered}", value,
                                            reference)

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_graph_stream_and_lifetime(self) -> None:
        """Exercise the public graph route without an intervening FFI barrier."""
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        producer = torch.npu.Stream(device=device)
        consumer = torch.npu.Stream(device=device)
        delay = torch.ones(1024 * 1024, device=device)
        history, consumed = [], []
        for dtype in ID_DTYPES:
            for ordered in (False, True):
                for iteration in range(4):
                    case = CSRCase("graph_stream", 257, 509, 8193, (ordered,),
                                   row_pattern="hub_middle", data_pattern="random",
                                   seed=20261114 + iteration)
                    source = _make_csr(case, dtype)
                    target = _copy_csr_to(source, dtype, device)
                    initial = CSRCase("initial", 257, 509, 8193, (ordered,))
                    inputs = _copy_csr_to(_make_csr(initial, dtype), dtype, device)
                    graph = _make_dgl_csr_graph(case, inputs, dtype, device)
                    cpu = _make_dgl_csr_graph(case, source, dtype, torch.device("cpu"))
                    expected = _dgl_csr_to_coo(cpu, ordered)
                    torch.npu.synchronize(device)
                    with torch.npu.stream(producer):
                        for _ in range(8):
                            delay.add_(1)
                        for destination, value in zip(inputs, target):
                            destination.copy_(value)
                        ready = torch.npu.Event()
                        ready.record(producer)
                    actual = _dgl_csr_to_coo(graph, ordered)
                    self.assertTrue(ready.query())
                    _assert_result(case, dtype, ordered, actual, expected)
                    with torch.npu.stream(consumer):
                        actual[0].record_stream(consumer)
                        actual[1].record_stream(consumer)
                        combined = actual[0] + actual[1]
                    consumed.append((combined, expected[0] + expected[1]))
                    history.append((case, dtype, ordered, actual, expected))
                    del graph, inputs, target, cpu
        gc.collect()
        torch.npu.empty_cache()
        for case, dtype, ordered, actual, expected in history:
            _assert_result(case, dtype, ordered, actual, expected)
        for actual, expected in consumed:
            _assert_same_tensor("consumer_stream", actual, expected)


def _run_native_performance_case(perf_name, warmup, repeat):
    """Time the optional tensor interface separately from graph API results."""
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    case_name, ordered = PERFORMANCE_CASES[perf_name]
    case = CASES_BY_NAME[case_name]
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    csr = _make_csr(case, torch.int32)
    device_csr = _copy_csr_to(csr, torch.int32, device)
    print(f"NATIVE_CONFIG name={perf_name} shape=({case.num_rows},"
          f"{case.num_cols},{case.nnz}) warmup={warmup} repeat={repeat}",
          flush=True)
    for _ in range(warmup):
        _native_csr_to_coo(case, device_csr, ordered)
    torch.npu.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat):
        _native_csr_to_coo(case, device_csr, ordered)
    torch.npu.synchronize(device)
    average_ms = (time.perf_counter() - start) * 1000 / repeat
    print(f"NATIVE_END {perf_name} average_ms={average_ms:.6f}", flush=True)
    return average_ms


def _run_performance_case(
    perf_name: str, warmup: int, repeat: int
) -> float:
    """Run one isolated int32 workload for collection by msprof."""
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    if perf_name not in PERFORMANCE_CASES:
        raise ValueError(f"Unknown performance case: {perf_name}")

    case_name, data_as_order = PERFORMANCE_CASES[perf_name]
    case = CASES_BY_NAME[case_name]
    dtype = torch.int32
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    csr = _make_csr(case, dtype)

    # data_as_order=True caches COO, so independent graphs are required for
    # every profiled call. False can repeatedly use the same CSR graph.
    graph_count = warmup + repeat if data_as_order else 1
    graphs = [
        _make_dgl_csr_graph(case, csr, dtype, device)
        for _ in range(graph_count)
    ]

    print(
        f"PERF_CONFIG name={perf_name} shape=({case.num_rows},"
        f"{case.num_cols},{case.nnz}) dtype=int32 "
        f"data_as_order={data_as_order} warmup={warmup} repeat={repeat}",
        flush=True,
    )
    for iteration in range(warmup):
        graph = graphs[iteration] if data_as_order else graphs[0]
        _dgl_csr_to_coo(graph, data_as_order)
    torch.npu.synchronize(device)

    print(f"PERF_START {perf_name}", flush=True)
    start = time.perf_counter()
    for iteration in range(repeat):
        graph_index = warmup + iteration if data_as_order else 0
        _dgl_csr_to_coo(graphs[graph_index], data_as_order)
    torch.npu.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    average_ms = elapsed_ms / repeat
    print(
        f"PERF_END {perf_name} total_ms={elapsed_ms:.6f} "
        f"average_ms={average_ms:.6f}",
        flush=True,
    )
    return average_ms


def _run_generalization(args: argparse.Namespace) -> None:
    """Audit new inputs without changing the original five-case benchmark."""
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.set_num_threads(8)
    cases = [c for c in GENERALIZATION_CASES
             if args.general_case == "all" or c.name == args.general_case]
    dtypes = ID_DTYPES if args.general_dtype == "all" else (
        getattr(torch, args.general_dtype),)
    orders = (False, True) if args.general_order == "all" else (
        args.general_order == "true",)
    libraries = sorted({line.split()[-1] for line in
                        Path("/proc/self/maps").read_text().splitlines()
                        if "libdgl.so" in line})
    report = {
        "variant": args.general_variant,
        "dgl_python": str(Path(dgl.__file__).resolve()),
        "libraries": libraries,
        "warmup": args.warmup, "repeat": args.repeat,
        "measurements": args.measurements, "results": [],
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in libraries},
        "python_files_sha256": {
            name: hashlib.sha256((Path(dgl.__file__).parent / name).read_bytes()).hexdigest()
            for name in ("_ffi/base.py", "_ffi/_ctypes/function.py",
                         "heterograph_index.py", "backend/pytorch/tensor.py")
        },
    }
    args.general_output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.general_output.write_text(json.dumps(report, indent=2) + "\n")

    print("GENERAL_ENV", json.dumps({k: v for k, v in report.items()
                                     if k != "results"}), flush=True)
    save()
    for case in cases:
        for dtype in dtypes:
            csr = _make_csr(case, dtype)
            digest = hashlib.sha256()
            for tensor in csr:
                digest.update(b"None" if tensor is None
                              else tensor.contiguous().numpy().tobytes())
            counts = (csr[0][1:] - csr[0][:-1]).to(torch.int64)
            adjacent = (float((csr[2][1:] - csr[2][:-1]).abs().eq(1).float().mean())
                        if csr[2] is not None and case.nnz > 1 else None)
            device_csr = _copy_csr_to(csr, dtype, device)
            for ordered in orders:
                result = {
                    "case": case.name, "dtype": str(dtype).split(".")[-1],
                    "data_as_order": ordered, "M": case.num_rows,
                    "N": case.num_cols, "E": case.nnz,
                    "row_pattern": case.row_pattern,
                    "data_pattern": case.data_pattern,
                    "with_data": case.with_data, "seed": case.seed,
                    "input_sha256": digest.hexdigest(),
                    "nonempty_rows": int(counts.ne(0).sum()),
                    "max_degree": int(counts.max()) if counts.numel() else 0,
                    "adjacent_id_fraction": adjacent,
                }
                try:
                    cpu_graph = _make_dgl_csr_graph(
                        case, csr, dtype, torch.device("cpu"))
                    check_graph = _make_dgl_csr_graph(case, device_csr, dtype, device)
                    expected = _dgl_csr_to_coo(cpu_graph, ordered)
                    actual = _dgl_csr_to_coo(check_graph, ordered)
                    torch.npu.synchronize(device)
                    _assert_result(case, dtype, ordered, actual, expected)
                    del cpu_graph, check_graph, actual, expected
                    gc.collect()
                    times = []
                    for _ in range(args.measurements):
                        graph_count = args.warmup + args.repeat if ordered else 1
                        graphs = [_make_dgl_csr_graph(case, device_csr, dtype, device)
                                  for _ in range(graph_count)]
                        for iteration in range(args.warmup):
                            _dgl_csr_to_coo(graphs[iteration if ordered else 0], ordered)
                        torch.npu.synchronize(device)
                        start = time.perf_counter()
                        for iteration in range(args.repeat):
                            index = args.warmup + iteration if ordered else 0
                            _dgl_csr_to_coo(graphs[index], ordered)
                        torch.npu.synchronize(device)
                        times.append((time.perf_counter() - start) * 1000 / args.repeat)
                        del graphs
                        gc.collect()
                    result.update(correct=True, round_ms=times,
                                  mean_ms=sum(times) / len(times))
                except Exception as error:
                    result.update(correct=False, error=repr(error))
                    report["results"].append(result)
                    save()
                    raise
                report["results"].append(result)
                print("GENERAL_RESULT", json.dumps(result), flush=True)
                save()
            del device_csr
            gc.collect()


def _run_msprof(
    output_dir: Path, perf_cases: Sequence[str], warmup: int, repeat: int,
    native: bool = False,
) -> None:
    msprof = shutil.which("msprof")
    if msprof is None:
        raise RuntimeError("msprof was not found in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for perf_name in perf_cases:
        case_output = output_dir / perf_name
        command = [
            msprof,
            f"--output={case_output}",
            "--ascendcl=on",
            "--runtime-api=on",
            "--task-time=l2",
            "--ai-core=on",
            "--aic-metrics=PipeUtilization",
            sys.executable,
            str(script),
            "--native-performance" if native else "--performance",
            "--perf-case",
            perf_name,
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--performance",
        action="store_true",
        help="Run NPU workloads, normally under msprof.",
    )
    mode.add_argument(
        "--run-msprof",
        action="store_true",
        help="Launch one msprof process per requested performance case.",
    )
    mode.add_argument("--generalization", action="store_true",
                      help="Audit additional shapes and distributions using existing helpers.")
    mode.add_argument("--native-performance", action="store_true",
                      help="Time the optional native CSR interface, separately from --performance.")
    parser.add_argument(
        "--perf-case",
        choices=("all", *PERFORMANCE_CASES.keys()),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument("--general-case", default="all",
                        choices=("all", *(c.name for c in GENERALIZATION_CASES)))
    parser.add_argument("--general-dtype", default="all",
                        choices=("all", "int32", "int64"))
    parser.add_argument("--general-order", default="all",
                        choices=("all", "false", "true"))
    parser.add_argument("--general-variant", default="optimized")
    parser.add_argument("--general-output", type=Path)
    parser.add_argument(
        "--msprof-output", type=Path, default=Path("./msprof_csr_to_coo")
    )
    parser.add_argument("--msprof-interface", choices=("graph", "native"),
                        default="graph")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.measurements <= 0:
        parser.error("warmup must be >= 0; repeat and measurements must be > 0")
    if args.generalization and args.general_output is None:
        parser.error("--generalization requires --general-output")
    return args


if __name__ == "__main__":
    cli_args = _parse_args()
    if cli_args.generalization:
        _run_generalization(cli_args)
        raise SystemExit(0)
    selected_cases = (
        tuple(PERFORMANCE_CASES)
        if cli_args.perf_case == "all"
        else (cli_args.perf_case,)
    )
    if cli_args.native_performance:
        for selected_case in selected_cases:
            _run_native_performance_case(
                selected_case, cli_args.warmup, cli_args.repeat)
        raise SystemExit(0)
    if not cli_args.performance and not cli_args.run_msprof:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            TestCSRToCOONPU
        )
        test_result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if test_result.wasSuccessful() else 1)
    if cli_args.run_msprof:
        _run_msprof(
            cli_args.msprof_output,
            selected_cases,
            warmup=cli_args.warmup,
            repeat=cli_args.repeat,
            native=cli_args.msprof_interface == "native",
        )
    else:
        for selected_case in selected_cases:
            _run_performance_case(
                selected_case,
                warmup=cli_args.warmup,
                repeat=cli_args.repeat,
            )
