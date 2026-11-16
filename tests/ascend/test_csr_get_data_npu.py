"""CSRGetData correctness and msprof performance tests on Ascend NPU.

    python tests/ascend/test_csr_get_data_npu.py
    python tests/ascend/test_csr_get_data_npu.py --performance --perf-case typical_eids
    python tests/ascend/test_csr_get_data_npu.py --run-msprof --msprof-output ./msprof_csr_get_data
    python tests/ascend/test_csr_get_data_npu.py --generalization --general-output /tmp/csr_get_data_general.json

EID calls graph._graph.edge_ids_one (supports missing edges); weights call
_sparse_ops._csrmask, whose CSRMask implementation dispatches CSRGetData.
Input construction/upload is excluded. Input/output FFI conversions, kernel,
allocation and final synchronization are included. No lookup result is cached.
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
from typing import Sequence

import dgl
from dgl import _sparse_ops
from dgl import heterograph_index
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

ETYPE = ("_U", "_E", "_V")
INTERFACE = "graph"


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


NPU_AVAILABLE = _npu_is_available()
ID_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    cols: int
    nnz: int
    queries: int
    weighted: bool
    a100_ms: float | None = None
    row_pattern: str = "balanced"


PERF_CASES = (
    Case("typical_eids", 10000, 10000, 500000, 65536, False, .260647),
    Case("typical_f32", 10000, 10000, 500000, 65536, True, .297075),
    Case("sampled_f32", 12000, 2000, 60000, 60000, True, .246708, "sampled"),
    Case("square_f32", 4096, 4096, 65536, 32768, True, .266096),
)
EXTRA_CASES = (
    Case("empty_queries", 17, 31, 43, 0, True),
    Case("empty_edges", 17, 31, 0, 39, True),
    Case("tiny_odd", 7, 17, 23, 37, True),
    Case("single_query", 7, 17, 23, 1, False),
    Case("long_row", 37, 65537, 32771, 4099, True, row_pattern="last"),
    Case("tall_sparse", 200003, 257, 2049, 4099, True),
    Case("large_batch", 1009, 1031, 65539, 131073, True),
)
CASES = {c.name:c for c in PERF_CASES + EXTRA_CASES}
GENERALIZATION_CASES = PERF_CASES + EXTRA_CASES
PERFORMANCE_CASES = {case.name: case for case in PERF_CASES}


def balanced(m, e):
    result = torch.full((m,), e // m, dtype=torch.int64)
    result[:e % m] += 1
    return result


def make_inputs(case, dtype=torch.int32, data_kind="identity", weight_dtype=torch.float32,
                seed=20261115, sorted_cols=False):
    rng = torch.Generator().manual_seed(seed)
    if case.row_pattern == "sampled":
        hot, edges = max(1, case.rows // 10), case.nnz * 4 // 5
        degree = torch.cat((balanced(hot, edges), balanced(case.rows-hot, case.nnz-edges)))
    elif case.row_pattern == "last":
        degree = torch.zeros(case.rows, dtype=torch.int64)
        degree[-1] = case.nnz
    else:
        degree = balanced(case.rows, case.nnz)
    p = torch.cat((torch.zeros(1, dtype=torch.int64), degree.cumsum(0)))
    positions = torch.arange(case.nnz, dtype=torch.int64)
    source = torch.repeat_interleave(torch.arange(case.rows), degree)
    j = (positions * 104729 + source * 31 + 8191).remainder(case.cols - 1)
    if data_kind == "none":
        a = torch.empty(0, dtype=torch.int64)
    elif data_kind == "identity":
        a = positions
    elif data_kind == "reverse":
        a = positions.flip(0)
    else:
        a = torch.randperm(case.nnz, generator=rng)
    if sorted_cols:
        # Stable per-row sorting, retaining the original edge-ID association.
        order = torch.argsort(source * case.cols + j, stable=True)
        j = j[order]
        a = (a if a.numel() else positions)[order]
    if case.nnz:
        picked = torch.randint(case.nnz, (case.queries,), generator=rng)
        r, c = source[picked], j[picked].clone()
    else:
        r = torch.randint(case.rows, (case.queries,), generator=rng)
        c = torch.zeros(case.queries, dtype=torch.int64)
    # Reserved last column produces guaranteed misses at exactly odd queries.
    c[1::2] = case.cols - 1
    weights = ((positions.to(torch.float64) + .125) * .37).to(weight_dtype)
    return tuple(t.to(dtype) for t in (p,j,a,r,c)) + (weights,)


def make_runner(case, values, device, interface=None):
    p,j,a,r,c,w = (t.to(device) for t in values)
    graph = dgl.heterograph({ETYPE:("csr",(p,j,a))},
                            num_nodes_dict={"_U":case.rows,"_V":case.cols},
                            idtype=p.dtype, device=device)
    if graph.formats()["created"] != ["csr"]:
        raise AssertionError("Lookup must use the original CSR")
    query_graph = None
    if case.weighted:
        # COO is already materialized before timing; CSRMask performs no format conversion.
        rr, cc = torch.broadcast_tensors(r,c)
        query_graph = dgl.heterograph({ETYPE:(rr,cc)},
                    num_nodes_dict={"_U":case.rows,"_V":case.cols},
                    idtype=p.dtype, device=device)
        if query_graph.formats()["created"] != ["coo"]:
            raise AssertionError("Queries must already have COO format")
        def run():
            return _sparse_ops._csrmask(graph._graph,w,query_graph._graph)
    else:
        def run():
            return graph._graph.edge_ids_one(0,r,c)
    if (interface or INTERFACE) == "native" and torch.device(device).type == "npu":
        def run():
            if case.weighted:
                result=torch.ops.dgl_ascend._try_csr_get_data_mask(
                    int(graph._graph.handle.value),w,int(query_graph._graph.handle.value))
                if result is None:raise RuntimeError("Native CSRMask format guard rejected input")
                return result
            return torch.ops.dgl_ascend._csr_get_data_graph(
                int(graph._graph.handle.value),0,r,c,w,not case.weighted,0.)
    if (interface or INTERFACE) == "ffi" and torch.device(device).type == "npu":
        def run():
            if case.weighted:
                answer=_sparse_ops._CAPI_DGLCSRMask(
                    graph._graph,dgl.backend.to_dgl_nd(w),query_graph._graph)
            else:
                answer=heterograph_index._CAPI_DGLHeteroEdgeIdsOne(
                    graph._graph,0,dgl.backend.to_dgl_nd(r),dgl.backend.to_dgl_nd(c))
            return dgl.backend.from_dgl_nd(answer)
    return run, (graph,query_graph,p,j,a,r,c,w)


def exact_check(actual, expected):
    if actual.device.type != "npu" or actual.dtype != expected.dtype:
        raise AssertionError(f"Wrong output device/dtype: {actual.device}/{actual.dtype}")
    if not torch.equal(actual.cpu(),expected):
        mismatch = int((actual.cpu()!=expected).sum()) if actual.shape==expected.shape else -1
        raise AssertionError(f"Output differs from CPU reference ({mismatch} entries)")


@unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
class TestCSRGetDataNPU(unittest.TestCase):
    def test_invalid_indices_and_broadcast_errors(self):
        packed=dgl._ffi.function.get_global_func("ascend._CAPI_DGLCSRGetData")
        for dtype in (torch.int32,torch.int64):
            p=torch.tensor([0,2,3,4],dtype=dtype)
            data=torch.tensor([3,1,0,2],dtype=dtype)
            invalid=[(-1,1),(3,1),(0,-1),(0,4),(-1,-1)]
            if dtype==torch.int64:invalid.extend(((1<<40,1),(0,1<<40)))
            for sorted_cols in (False,True):
                j=torch.tensor([1,2,0,3] if sorted_cols else [2,1,0,3],dtype=dtype)
                for weighted,wtype in ((False,torch.float32),(True,torch.float32),(True,torch.float64)):
                    for row,col in invalid:
                        expected="Invalid row index" if row<0 or row>=3 else "Invalid col index"
                        for device in ("cpu","npu:0"):
                            with self.subTest(dtype=dtype,sorted=sorted_cols,weighted=weighted,
                                              weight_dtype=wtype,row=row,col=col,device=device):
                                matrix=dgl.ndarray.SparseMatrix.__new__(dgl.ndarray.SparseMatrix)
                                matrix.__setstate__((2,3,4,tuple(x.to(device) for x in (p,j,data)),[sorted_cols]))
                                r=torch.tensor([row],dtype=dtype,device=device)
                                c=torch.tensor([col],dtype=dtype,device=device)
                                w=torch.arange(4,dtype=wtype,device=device)
                                with self.assertRaisesRegex(dgl.DGLError,expected):
                                    packed(matrix,dgl.backend.to_dgl_nd(r),dgl.backend.to_dgl_nd(c),
                                           not weighted,dgl.backend.to_dgl_nd(w),2.5)
            cpu=dgl.heterograph({ETYPE:("csr",(p,j,data))},
                                num_nodes_dict={"_U":3,"_V":4},idtype=dtype)
            npu=cpu.to("npu:0")
            for g,device in ((cpu,"cpu"),(npu,"npu:0")):
                r=torch.tensor([0,1],dtype=dtype,device=device)
                c=torch.tensor([1,0,3],dtype=dtype,device=device)
                for call in (lambda:g._graph.edge_ids_one(0,r,c),lambda:g.edge_ids(r,c)):
                    with self.assertRaisesRegex(dgl.DGLError,"Invalid row and col id array"):
                        call()
                for row,col in invalid:
                    r=torch.tensor([row],dtype=dtype,device=device)
                    c=torch.tensor([col],dtype=dtype,device=device)
                    expected="Invalid row index" if row<0 or row>=3 else "Invalid col index"
                    with self.assertRaisesRegex(dgl.DGLError,expected):
                        g._graph.edge_ids_one(0,r,c)

    def test_error_status_reset_and_query_tiles(self):
        for dtype in (torch.int32,torch.int64):
            case=Case("errors",7,17,23,2561,False)
            values=make_inputs(case,dtype,"random")
            expected,_=make_runner(case,values,"cpu")
            run,owners=make_runner(case,values,"npu:0",interface="graph")
            r,c=owners[-3:-1]
            for index in (0,63,64,511,512,2560):
                previous=int(r[index].cpu())
                r[index]=-1
                with self.assertRaisesRegex(dgl.DGLError,"Invalid row index: -1"):
                    run()
                r[index]=previous
                exact_check(run(),expected())
            # Broadcast errors use the original source position when reporting.
            graph=owners[0]
            with self.assertRaisesRegex(dgl.DGLError,"Invalid col index: -1"):
                graph._graph.edge_ids_one(0,r,torch.tensor([-1],dtype=dtype,device="npu:0"))
            exact_check(run(),expected())

    def test_graph_format_fallbacks(self):
        case=Case("formats",7,17,23,37,False)
        for dtype in (torch.int32,torch.int64):
            values=make_inputs(case,dtype,"random")
            for fmt in ("csr","coo","csc"):
                with self.subTest(dtype=dtype,format=fmt):
                    _,owners=make_runner(case,values,"cpu")
                    cpu=owners[0].formats(fmt)
                    npu=cpu.to("npu:0")
                    r,c=values[3][::2].contiguous(),values[4][::2].contiguous()
                    expected=cpu._graph.edge_ids_one(0,r,c)
                    actual=npu._graph.edge_ids_one(0,r.to("npu:0"),c.to("npu:0"))
                    exact_check(actual,expected)
                    self.assertEqual(npu.formats(),cpu.formats())

    def test_native_graph_contract(self):
        for dtype in (torch.int32,torch.int64):
            for weighted in (False,True):
                case=Case("native_contract",7,17,23,37,weighted)
                values=make_inputs(case,dtype,"random",torch.float64)
                cpu,_=make_runner(case,values,"cpu")
                native,owners=make_runner(case,values,"npu:0",interface="native")
                graph_api,_=make_runner(case,values,"npu:0",interface="graph")
                expected=cpu()
                exact_check(native(),expected)
                exact_check(graph_api(),expected)
                if weighted:
                    g,_,_,_,_,r,c,w=owners
                    with self.assertRaisesRegex(RuntimeError,"autograd"):
                        torch.ops.dgl_ascend._csr_get_data_graph(
                            int(g._graph.handle.value),0,r,c,w.detach().requires_grad_(),False,0.)

    def test_native_heterogeneous_relation(self):
        dtype=torch.int64
        specs={('u','a','v'):('csr',(torch.tensor([0,1,1],dtype=dtype),
               torch.tensor([1],dtype=dtype),torch.tensor([0],dtype=dtype))),
               ('u','z','v'):('csr',(torch.tensor([0,1,2],dtype=dtype),
               torch.tensor([5,3],dtype=dtype),torch.tensor([1,0],dtype=dtype)))}
        cpu=dgl.heterograph(specs,num_nodes_dict={'u':2,'v':7},idtype=dtype)
        npu=cpu.to('npu:0')
        r=torch.tensor([0,1,0],dtype=dtype);c=torch.tensor([5,3,2],dtype=dtype)
        etype=cpu.get_etype_id(('u','z','v'))
        expected=cpu._graph.edge_ids_one(etype,r,c)
        actual=torch.ops.dgl_ascend._csr_get_data_graph(
            int(npu._graph.handle.value),etype,r.to('npu:0'),c.to('npu:0'),
            torch.empty(0,device='npu:0'),True,0.)
        exact_check(actual,expected)

    def test_functional_matrix(self):
        for case in PERF_CASES + EXTRA_CASES:
            for dtype in (torch.int32,torch.int64):
                for data in ("none","identity","reverse","random"):
                    for weighted in (False,True):
                        custom = Case(case.name,case.rows,case.cols,case.nnz,case.queries,
                                      weighted,row_pattern=case.row_pattern)
                        for wtype in ((torch.float32,torch.float64) if weighted else (torch.float32,)):
                            with self.subTest(case=case.name,dtype=dtype,data=data,weighted=weighted,wtype=wtype):
                                values = make_inputs(custom,dtype,data,wtype)
                                cpu,_ = make_runner(custom,values,"cpu")
                                npu,_ = make_runner(custom,values,"npu:0")
                                exact_check(npu(),cpu())

    def test_broadcast_duplicates_and_mutation(self):
        for dtype in (torch.int32,torch.int64):
            for weighted in (False,True):
                case = Case("duplicates",3,17,7,7,weighted)
                p=torch.tensor([0,4,4,7],dtype=dtype)
                j=torch.tensor([5,2,5,7,3,3,1],dtype=dtype)
                a=torch.tensor([6,5,4,3,2,1,0],dtype=dtype)
                w=torch.arange(7,dtype=torch.float64)+.123456789
                for r,c in (([0],[5,2,7,16]),([0,1,2,0],[3]),
                            ([0,0,2,2],[5,5,3,1])):
                    values=(p,j,a,torch.tensor(r,dtype=dtype),torch.tensor(c,dtype=dtype),w)
                    cpu,_=make_runner(case,values,"cpu")
                    npu,owners=make_runner(case,values,"npu:0")
                    exact_check(npu(),cpu())
                    # Reuse the same graph and input addresses with changed data/weights.
                    owners[4].copy_(a.flip(0).to("npu:0"))
                    owners[-1].add_(2.)
                    changed=(p,j,a.flip(0),values[3],values[4],w+2.)
                    cpu,_=make_runner(case,changed,"cpu")
                    result=npu()
                    exact_check(result,cpu())
                    del npu,owners
                    gc.collect()
                    torch.npu.empty_cache()
                    exact_check(result,cpu())

    def test_noncurrent_producer(self):
        case=Case("stream",127,257,2049,1027,True)
        values=make_inputs(case,torch.int64,"random",torch.float64)
        cpu,_=make_runner(case,values,"cpu")
        npu,owners=make_runner(case,values,"npu:0")
        stream=torch.npu.Stream()
        with torch.npu.stream(stream):
            owners[-1].add_(1.25)
        modified=values[:-1]+(values[-1]+1.25,)
        cpu,_=make_runner(case,modified,"cpu")
        exact_check(npu(),cpu())

    def test_sparse_matrix_sorted_and_fillers(self):
        packed=dgl._ffi.function.get_global_func("ascend._CAPI_DGLCSRGetData")
        for dtype in (torch.int32,torch.int64):
            for sorted_cols in (False,True):
                for wtype in (torch.float32,torch.float64):
                    case=Case("sorted_native",7,65537,1027,137,True,row_pattern="last")
                    values=make_inputs(case,dtype,"random",wtype,sorted_cols=sorted_cols)
                    for broadcast in (False,True):
                        p,j,a,r,c,w=values
                        if broadcast:r=r[:1]
                        def invoke(device):
                            tensors=tuple(t.to(device) for t in (p,j,a,r,c,w))
                            pp,jj,aa,rr,cc,ww=tensors
                            matrix=dgl.ndarray.SparseMatrix.__new__(dgl.ndarray.SparseMatrix)
                            matrix.__setstate__((2,case.rows,case.cols,(pp,jj,aa),[sorted_cols]))
                            answer=packed(matrix,dgl.backend.to_dgl_nd(rr),
                                dgl.backend.to_dgl_nd(cc),False,dgl.backend.to_dgl_nd(ww),-7.123456789)
                            return dgl.backend.from_dgl_nd(answer)
                        exact_check(invoke("npu:0"),invoke("cpu"))

    def test_int64_high_columns_and_special_weights(self):
        packed=dgl._ffi.function.get_global_func("ascend._CAPI_DGLCSRGetData")
        p=torch.tensor([0,5,5],dtype=torch.int64)
        j=torch.tensor([5,2**32+5,2**40+5,7,2**32+5],dtype=torch.int64)
        a=torch.tensor([4,3,2,1,0],dtype=torch.int64)
        r=torch.zeros(5,dtype=torch.int64)
        c=torch.tensor([5,2**32+5,2**40+5,2**39+5,7],dtype=torch.int64)
        for wtype in (None,torch.float32,torch.float64):
            w=torch.tensor([float("inf"),float("-inf"),float("nan"),1.25,-9.5],
                           dtype=wtype or torch.float32)
            def invoke(device):
                pp,jj,aa,rr,cc,ww=(t.to(device) for t in (p,j,a,r,c,w))
                matrix=dgl.ndarray.SparseMatrix.__new__(dgl.ndarray.SparseMatrix)
                matrix.__setstate__((2,2,2**41,(pp,jj,aa),[False]))
                result=packed(matrix,dgl.backend.to_dgl_nd(rr),dgl.backend.to_dgl_nd(cc),
                              wtype is None,dgl.backend.to_dgl_nd(ww),-123.5)
                return dgl.backend.from_dgl_nd(result)
            actual,expected=invoke("npu:0"),invoke("cpu")
            if wtype:
                self.assertEqual(actual.dtype,wtype)
                bits=torch.int32 if wtype==torch.float32 else torch.int64
                actual,expected=actual.view(bits),expected.view(bits)
            exact_check(actual,expected)

    def test_csrmask_autograd(self):
        case=Case("autograd",3,7,3,3,True)
        for dtype in (torch.int32,torch.int64):
            values=tuple(torch.tensor(x,dtype=dtype) for x in
                         ([0,2,2,3],[1,5,2],[2,0,1],[0,2,1],[5,2,3]))
            values+=(torch.tensor([.25,1.5,-2.]),)
            outputs=[]
            for device in ("cpu","npu:0"):
                _,owners=make_runner(case,values,device)
                g,b,*_=owners
                w=owners[-1].detach().requires_grad_()
                out=dgl.backend.csrmask(g._graph,w,b._graph)
                out.backward(torch.ones_like(out))
                outputs.append((out.detach(),w.grad))
            exact_check(outputs[1][0],outputs[0][0])
            exact_check(outputs[1][1],outputs[0][1])

    def test_noncompact_eids(self):
        packed=dgl._ffi.function.get_global_func("ascend._CAPI_DGLCSRGetData")
        for dtype in (torch.int32,torch.int64):
            for wtype in (torch.float32,torch.float64):
                ids=tuple(torch.tensor(x,dtype=dtype) for x in
                          ([0,3],[2,3,4],[11,7,19],[0,0,0,0],[2,3,4,7]))
                weights=torch.arange(32,dtype=wtype)+.125
                def invoke(device):
                    p,j,a,r,c=(t.to(device) for t in ids)
                    w=weights.to(device)
                    matrix=dgl.ndarray.SparseMatrix.__new__(dgl.ndarray.SparseMatrix)
                    matrix.__setstate__((2,1,8,(p,j,a),[False]))
                    result=packed(matrix,dgl.backend.to_dgl_nd(r),dgl.backend.to_dgl_nd(c),
                                  False,dgl.backend.to_dgl_nd(w),-7.5)
                    return dgl.backend.from_dgl_nd(result)
                exact_check(invoke("npu:0"),invoke("cpu"))


def _run_performance_case(perf_name, warmup, repeat, *, native=False):
    """Run one isolated int32 workload, using CSRToCOO's timing boundaries."""
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    case = CASES[perf_name]
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    values = make_inputs(case, torch.int32)
    run, owners = make_runner(case, values, device,
                              interface="native" if native else "graph")
    prefix = "NATIVE" if native else "PERF"
    print(f"{prefix}_CONFIG name={perf_name} shape=({case.rows},"
          f"{case.cols},{case.nnz}) dtype=int32 Q={case.queries} "
          f"weighted={case.weighted} warmup={warmup} repeat={repeat}", flush=True)
    for _ in range(warmup):
        run()
    torch.npu.synchronize(device)
    if not native:
        print(f"PERF_START {perf_name}", flush=True)
    start = time.perf_counter()
    for _ in range(repeat):
        run()
    torch.npu.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    average_ms = elapsed_ms / repeat
    if native:
        print(f"NATIVE_END {perf_name} average_ms={average_ms:.6f}", flush=True)
    else:
        print(f"PERF_END {perf_name} total_ms={elapsed_ms:.6f} "
              f"average_ms={average_ms:.6f}", flush=True)
    return average_ms


def _run_native_performance_case(perf_name, warmup, repeat):
    return _run_performance_case(perf_name, warmup, repeat, native=True)


def _general_configs(case):
    return [(data, weighted, weight_dtype)
            for data in ("none", "random")
            for weighted, weight_dtype in ((False, torch.float32),
                                           (True, torch.float32),
                                           (True, torch.float64))]


def _general_runner(case, dtype, device, config):
    data, weighted, weight_dtype = config
    custom = Case(case.name, case.rows, case.cols, case.nnz, case.queries,
                  weighted, row_pattern=case.row_pattern)
    values = make_inputs(custom, dtype, data, weight_dtype)
    cpu, cpu_owners = make_runner(custom, values, "cpu", interface="graph")
    run, owners = make_runner(custom, values, device, interface="graph")
    exact_check(run(), cpu())
    p, _, a = values[:3]
    degree = p[1:].to(torch.int64) - p[:-1].to(torch.int64)
    metadata = {"Q": case.queries, "weighted": weighted,
                "weight_dtype": str(weight_dtype).split(".")[-1] if weighted else None,
                "row_pattern": case.row_pattern, "data_pattern": data,
                "with_data": bool(a.numel()), "seed": 20261115,
                "input_sha256": hashlib.sha256(b"".join(t.numpy().tobytes() for t in values)).hexdigest(),
                "nonempty_rows": int(degree.ne(0).sum()),
                "max_degree": int(degree.max()) if degree.numel() else 0,
                "adjacent_id_fraction": float((a[1:] - a[:-1]).abs().eq(1).float().mean())
                if a.numel() > 1 else None}
    return run, owners, metadata


def _general_report(args):
    # Keep the GENERAL_ENV/GENERAL_RESULT schema used by CSRToCOO.
    libraries = sorted({line.split()[-1] for line in
                        Path("/proc/self/maps").read_text().splitlines()
                        if "libdgl.so" in line})
    return {
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


def _run_generalization(args):
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.set_num_threads(8)
    cases = [c for c in GENERALIZATION_CASES
             if args.general_case == "all" or c.name == args.general_case]
    dtypes = ID_DTYPES if args.general_dtype == "all" else (
        getattr(torch, args.general_dtype),)
    report = _general_report(args)
    args.general_output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.general_output.write_text(json.dumps(report, indent=2) + "\n")

    print("GENERAL_ENV", json.dumps({k: v for k, v in report.items()
                                     if k != "results"}), flush=True)
    save()
    for case in cases:
        for dtype in dtypes:
            for config in _general_configs(case):
                result = {"case": case.name, "dtype": str(dtype).split(".")[-1],
                          "M": case.rows, "N": case.cols, "E": case.nnz}
                try:
                    run, owners, metadata = _general_runner(case, dtype, device, config)
                    result.update(metadata)
                    times = []
                    for _ in range(args.measurements):
                        for _ in range(args.warmup):
                            run()
                        torch.npu.synchronize(device)
                        start = time.perf_counter()
                        for _ in range(args.repeat):
                            run()
                        torch.npu.synchronize(device)
                        times.append((time.perf_counter() - start) * 1000 / args.repeat)
                    result.update(correct=True, round_ms=times,
                                  mean_ms=sum(times) / len(times))
                    del run, owners
                    gc.collect()
                except Exception as error:
                    result.update(correct=False, error=repr(error))
                    report["results"].append(result)
                    save()
                    raise
                report["results"].append(result)
                print("GENERAL_RESULT", json.dumps(result), flush=True)
                save()


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


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--performance", action="store_true",
                      help="Run NPU workloads, normally under msprof.")
    mode.add_argument("--run-msprof", action="store_true",
                      help="Launch one msprof process per requested performance case.")
    mode.add_argument("--generalization", action="store_true",
                      help="Audit additional shapes and distributions using existing helpers.")
    mode.add_argument("--native-performance", action="store_true",
                      help="Time the optional native CSR interface, separately from --performance.")
    parser.add_argument("--perf-case", choices=("all", *PERFORMANCE_CASES), default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument("--general-case", default="all",
                        choices=("all", *(c.name for c in GENERALIZATION_CASES)))
    parser.add_argument("--general-dtype", default="all",
                        choices=("all", "int32", "int64"))
    parser.add_argument("--general-variant", default="optimized")
    parser.add_argument("--general-output", type=Path)
    parser.add_argument("--msprof-output", type=Path, default=Path("./msprof_csr_get_data"))
    parser.add_argument("--msprof-interface", choices=("graph", "native"), default="graph")

    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.measurements <= 0:
        parser.error("warmup must be >= 0; repeat and measurements must be > 0")
    if args.generalization and args.general_output is None:
        parser.error("--generalization requires --general-output")
    return args


def main():
    args = _parse_args()

    if args.generalization:
        _run_generalization(args)
        return
    selected_cases = (tuple(PERFORMANCE_CASES) if args.perf_case == "all"
                      else (args.perf_case,))
    if args.native_performance:
        for name in selected_cases:
            _run_native_performance_case(name, args.warmup, args.repeat)
        return
    if not args.performance and not args.run_msprof:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCSRGetDataNPU)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.run_msprof:
        _run_msprof(args.msprof_output, selected_cases, args.warmup, args.repeat,
                    native=args.msprof_interface == "native")
    else:
        for name in selected_cases:
            _run_performance_case(name, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
