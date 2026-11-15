"""CSRTranspose correctness and full graph-API performance on Ascend NPU.

    python tests/ascend/test_csr_transpose_npu.py
    python tests/ascend/test_csr_transpose_npu.py --performance --repeat 100 --warmup 20
    python tests/ascend/test_csr_transpose_npu.py --run-msprof --repeat 100 --warmup 20

Fresh CSR graphs prevent the cached CSC format from bypassing CSRTranspose.
Generation, upload and graph construction are outside the timed region;
g.adj_tensors("csc"), all output conversion, and final synchronization are timed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

import dgl
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

ETYPE = ("_U", "_E", "_V")
DTYPES = (torch.int32, torch.int64)
NPU_AVAILABLE = torch_npu is not None and torch.npu.is_available()


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    cols: int
    nnz: int
    with_data: bool
    rows_kind: str = "balanced"
    cols_kind: str = "mixed"
    a100_ms: float | None = None


PERFORMANCE_CASES = (
    Case("long_tail_no_data", 10000, 10000, 500000, False,
         "long_tail", a100_ms=2.656580),
    Case("long_tail_with_data", 10000, 10000, 500000, True,
         "long_tail", a100_ms=2.871920),
    Case("square_no_data", 4096, 4096, 65536, False, a100_ms=2.122950),
    Case("square_with_data", 4096, 4096, 65536, True, a100_ms=1.905390),
    Case("sampled_no_data", 12000, 2000, 60000, False,
         "sampled", a100_ms=1.601890),
    Case("sampled_with_data", 12000, 2000, 60000, True,
         "sampled", a100_ms=2.286970),
)
EXTRA_CASES = (
    Case("empty", 0, 0, 0, False),
    Case("zero_edges", 17, 23, 0, True),
    Case("zero_source_rows", 0, 17, 0, False),
    Case("zero_columns", 17, 0, 0, False),
    Case("single_last_edge", 262147, 257, 1, True, "needle"),
    Case("odd_tail", 37, 91, 113, True),
    Case("one_dense_row", 1, 509, 8193, False),
    Case("one_column", 4099, 1, 65539, True),
    Case("duplicate_columns", 257, 509, 8193, True, cols_kind="duplicates"),
    Case("unsorted_columns", 257, 509, 8193, False, cols_kind="descending"),
    Case("band_boundary", 257, 16385, 8193, True),
    Case("empty_bands", 257, 40001, 8193, False, cols_kind="last"),
    Case("first_column", 257, 20003, 8193, True, cols_kind="first"),
)
CASES = {case.name: case for case in PERFORMANCE_CASES + EXTRA_CASES}


def balanced(rows, nnz):
    if rows == 0:
        assert nnz == 0
        return torch.empty(0, dtype=torch.int64)
    q, r = divmod(nnz, rows)
    counts = torch.full((rows,), q, dtype=torch.int64)
    counts[:r] += 1
    return counts


def make_csr(case, dtype):
    if case.rows_kind == "long_tail":
        counts = torch.ones(case.rows, dtype=torch.int64)
        remaining = case.nnz - case.rows
        hot = remaining * 3 // 4
        counts += balanced(case.rows, remaining - hot)
        counts[case.rows // 2] += hot
    elif case.rows_kind == "sampled":
        hot_rows = max(1, case.rows // 10)
        hot_edges = case.nnz * 4 // 5
        counts = torch.cat((balanced(hot_rows, hot_edges),
                            balanced(case.rows - hot_rows, case.nnz - hot_edges)))
    elif case.rows_kind == "needle":
        counts = torch.zeros(case.rows, dtype=torch.int64)
        counts[-1] = case.nnz
    else:
        counts = balanced(case.rows, case.nnz)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    positions = torch.arange(case.nnz, dtype=torch.int64)
    if case.nnz == 0:
        indices = positions
    elif case.cols_kind == "first":
        indices = torch.zeros_like(positions)
    elif case.cols_kind == "last":
        indices = torch.full_like(positions, case.cols - 1)
    elif case.cols_kind == "duplicates":
        indices = (positions // 7).remainder(min(5, case.cols))
    elif case.cols_kind == "descending":
        indices = case.cols - 1 - positions.remainder(case.cols)
    else:
        indices = (positions * 104729 + 8191).remainder(case.cols)
    data = positions.flip(0) if case.with_data else torch.empty(0, dtype=torch.int64)
    assert int(indptr[-1]) == case.nnz
    return tuple(x.to(dtype) for x in (indptr, indices, data))


def make_graph(case, csr, dtype, device):
    return dgl.heterograph(
        {ETYPE: ("csr", tuple(x.to(device) for x in csr))},
        num_nodes_dict={"_U": case.rows, "_V": case.cols},
        idtype=dtype, device=device)


def transpose(graph):
    return graph.adj_tensors("csc", etype=ETYPE)


def assert_result(case, dtype, actual, expected):
    sizes = (case.cols + 1, case.nnz, case.nnz)
    for label, size, value, reference in zip(
            ("indptr", "indices", "data"), sizes, actual, expected):
        assert value.device.type == "npu", (case.name, label, value.device)
        assert value.dtype == dtype and value.shape == (size,)
        cpu = value.cpu()
        assert torch.equal(cpu, reference), (case.name, dtype, label,
                                            "NPU result differs from stable CPU transpose")
    assert int(actual[0][-1].cpu()) == case.nnz


class TestCSRTransposeNPU(unittest.TestCase):
    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_functional_matrix(self):
        torch.npu.set_device(0)
        for case in PERFORMANCE_CASES + EXTRA_CASES:
            for dtype in DTYPES:
                with self.subTest(case=case.name, dtype=dtype):
                    csr = make_csr(case, dtype)
                    expected = transpose(make_graph(case, csr, dtype, torch.device("cpu")))
                    graph = make_graph(case, csr, dtype, torch.device("npu:0"))
                    actual = transpose(graph)
                    torch.npu.synchronize()
                    assert_result(case, dtype, actual, expected)

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_randomized(self):
        # Independent random degrees/columns also exercise duplicate edges and
        # non-divisible partitions, beyond the six measured input shapes.
        shapes = ((3, 7, 19), (127, 31, 2053), (1027, 511, 4099),
                  (17, 16384, 2049), (41, 32769, 4103), (110003, 70001, 2051))
        for seed, (m, n, e) in enumerate(shapes):
            rng = torch.Generator().manual_seed(20261115 + seed)
            source_rows = torch.randint(m, (e,), generator=rng)
            counts = torch.bincount(source_rows, minlength=m)
            p = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
            j = torch.randint(n, (e,), generator=rng)
            for with_data in (False, True):
                case = Case(f"random_{seed}_{with_data}", m, n, e, with_data)
                a = torch.randperm(e, generator=rng) if with_data else torch.empty(0, dtype=torch.int64)
                for dtype in DTYPES:
                    with self.subTest(case=case.name, dtype=dtype):
                        csr = tuple(x.to(dtype) for x in (p, j, a))
                        expected = transpose(make_graph(case, csr, dtype, torch.device("cpu")))
                        actual = transpose(make_graph(case, csr, dtype, torch.device("npu:0")))
                        assert_result(case, dtype, actual, expected)

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_mutation_and_lifetime(self):
        case = Case("mutated", 257, 509, 8193, True, cols_kind="duplicates")
        history = []
        for dtype in DTYPES:
            source = make_csr(case, dtype)
            target = tuple(x.to("npu:0") for x in source)
            for shift in (1, 7, 31):
                degrees = source[0].diff().roll(shift)
                changed_ptr = torch.cat((source[0][:1], degrees.cumsum(0).to(dtype)))
                cpu = (changed_ptr, (source[1] + shift).remainder(case.cols),
                       source[2].roll(shift))
                graph = make_graph(case, target, dtype, torch.device("npu:0"))
                expected = transpose(make_graph(case, cpu, dtype, torch.device("cpu")))
                target[0].copy_(cpu[0].to("npu:0"))
                target[1].copy_(cpu[1].to("npu:0"))
                target[2].copy_(cpu[2].to("npu:0"))
                actual = transpose(graph)
                assert_result(case, dtype, actual, expected)
                history.append((dtype, actual, expected))
                del graph
        gc.collect()
        torch.npu.empty_cache()
        for dtype, actual, expected in history:
            assert_result(case, dtype, actual, expected)


def performance(name, warmup, repeat):
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    case = CASES[name]
    dtype, device = torch.int32, torch.device("npu:0")
    torch.npu.set_device(device)
    csr = make_csr(case, dtype)
    digest = hashlib.sha256(b"".join(x.numpy().tobytes() for x in csr)).hexdigest()
    graphs = [make_graph(case, csr, dtype, device) for _ in range(warmup + repeat)]
    print(f"PERF_CONFIG name={name} shape=({case.rows},{case.cols},{case.nnz}) "
          f"dtype=int32 with_data={case.with_data} warmup={warmup} repeat={repeat} "
          f"input_sha256={digest}", flush=True)
    for graph in graphs[:warmup]:
        transpose(graph)
    torch.npu.synchronize(device)
    start = time.perf_counter()
    for i in range(repeat):
        transpose(graphs[warmup + i])
    torch.npu.synchronize(device)
    average_ms = (time.perf_counter() - start) * 1000 / repeat
    print(f"PERF_END {name} average_ms={average_ms:.6f}", flush=True)
    return dict(case=name, M=case.rows, N=case.cols, E=case.nnz,
                with_data=case.with_data, input_sha256=digest,
                npu_ms=average_ms, a100_ms=case.a100_ms,
                a100_over_npu=case.a100_ms / average_ms if case.a100_ms else None,
                limit_ms=case.a100_ms / .3 if case.a100_ms else None,
                passes_03=average_ms <= case.a100_ms / .3 if case.a100_ms else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--performance", action="store_true")
    mode.add_argument("--run-msprof", action="store_true")
    parser.add_argument("--perf-case", default="all",
                        choices=("all", *CASES))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--msprof-output", type=Path, default=Path("/tmp/csr_transpose_msprof"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be >= 0 and repeat must be > 0")
    cases = ([c.name for c in PERFORMANCE_CASES] if args.perf_case == "all"
             else [args.perf_case])
    if args.run_msprof:
        executable = shutil.which("msprof")
        if not executable:
            raise RuntimeError("msprof is not in PATH")
        for name in cases:
            subprocess.run([executable, f"--output={args.msprof_output / name}",
                            "--ascendcl=on", "--runtime-api=on", "--task-time=l2",
                            "--ai-core=on", "--aic-metrics=PipeUtilization",
                            sys.executable, str(Path(__file__).resolve()),
                            "--performance", "--perf-case", name,
                            "--warmup", str(args.warmup), "--repeat", str(args.repeat)],
                           check=True)
    elif args.performance:
        report = dict(warmup=args.warmup, repeat=args.repeat,
                      test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      dgl_python=str(Path(dgl.__file__).resolve()), results=[])
        for name in cases:
            report["results"].append(performance(name, args.warmup, args.repeat))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCSRTransposeNPU)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
