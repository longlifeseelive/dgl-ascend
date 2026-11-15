"""CSRGetData exact correctness and full DGL graph-API NPU performance.

    python tests/ascend/test_csr_get_data_npu.py
    python tests/ascend/test_csr_get_data_npu.py --performance --output results.json
    python tests/ascend/test_csr_get_data_npu.py --run-msprof --msprof-output /tmp/csr_get_data_prof

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

import dgl
from dgl import _sparse_ops
import torch
import torch_npu  # noqa: F401

ETYPE = ("_U", "_E", "_V")


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


def make_runner(case, values, device):
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
    return run, (graph,query_graph,p,j,a,r,c,w)


def exact_check(actual, expected):
    if actual.device.type != "npu" or actual.dtype != expected.dtype:
        raise AssertionError(f"Wrong output device/dtype: {actual.device}/{actual.dtype}")
    if not torch.equal(actual.cpu(),expected):
        mismatch = int((actual.cpu()!=expected).sum()) if actual.shape==expected.shape else -1
        raise AssertionError(f"Output differs from CPU reference ({mismatch} entries)")


class TestCSRGetDataNPU(unittest.TestCase):
    def test_functional_matrix(self):
        count = 0
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
                                count += 1
        print(f"FUNCTIONAL_CHECKS {count}",flush=True)

    def test_broadcast_duplicates_and_mutation(self):
        count = 0
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
                    exact_check(npu(),cpu());count+=1
                    # Reuse the same graph and input addresses with changed data/weights.
                    owners[4].copy_(a.flip(0).to("npu:0"))
                    owners[-1].add_(2.)
                    changed=(p,j,a.flip(0),values[3],values[4],w+2.)
                    cpu,_=make_runner(case,changed,"cpu")
                    result=npu()
                    exact_check(result,cpu());count+=1
                    del npu,owners
                    gc.collect()
                    torch.npu.empty_cache()
                    exact_check(result,cpu())
        print(f"BROADCAST_MUTATION_CHECKS {count}",flush=True)

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
        checks=0
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
                        exact_check(invoke("npu:0"),invoke("cpu"));checks+=1
        print(f"SORTED_FILLER_CHECKS {checks}",flush=True)

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


def performance(case, warmup, repeat, rounds, dtype=torch.int32,
                data_kind="identity", weight_dtype=torch.float32):
    values=make_inputs(case,dtype,data_kind,weight_dtype)
    cpu,_=make_runner(case,values,"cpu")
    expected=cpu()
    run,owners=make_runner(case,values,"npu:0")
    digest=hashlib.sha256()
    for t in values:
        digest.update(t.numpy().tobytes())
    correct=True; error=None
    try:
        exact_check(run(),expected)
    except AssertionError as exc:
        correct=False;error=str(exc)
    times=[]
    for _ in range(rounds):
        for _ in range(warmup):run()
        torch.npu.synchronize()
        start=time.perf_counter()
        for _ in range(repeat):result=run()
        torch.npu.synchronize()
        times.append((time.perf_counter()-start)*1000/repeat)
        del result
    mean=sum(times)/len(times)
    print(f"PERF {case.name} {dtype} {data_kind} weighted={case.weighted}/{weight_dtype} "
          f"correct={correct} mean_ms={mean:.6f} rounds={times}",flush=True)
    return dict(case=case.name,rows=case.rows,cols=case.cols,nnz=case.nnz,
                queries=case.queries,weighted=case.weighted,dtype=str(dtype).split('.')[-1],
                weight_dtype=str(weight_dtype).split('.')[-1] if case.weighted else None,
                data=data_kind,hit_fraction=((case.queries+1)//2/case.queries
                    if case.nnz and case.queries else 0.),correct=correct,error=error,
                input_sha256=digest.hexdigest(),round_ms=times,npu_ms=mean,
                a100_ms=case.a100_ms,ratio=case.a100_ms/mean if case.a100_ms else None,
                limit_ms=case.a100_ms/.3 if case.a100_ms else None,
                passes_03=(correct and mean<=case.a100_ms/.3) if case.a100_ms else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--performance",action="store_true")
    modes.add_argument("--run-msprof",action="store_true")
    modes.add_argument("--extended",action="store_true")
    parser.add_argument("--perf-case",default="all",choices=("all",*CASES))
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--repeat",type=int,default=100)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--msprof-output",type=Path,default=Path("/tmp/csr_get_data_prof"))
    args=parser.parse_args()
    if args.warmup<0 or args.repeat<1 or args.rounds<1:parser.error("Invalid repetition counts")
    torch.npu.set_device(0)
    cases=(list(PERF_CASES + EXTRA_CASES) if args.extended and args.perf_case=="all"
           else list(PERF_CASES) if args.perf_case=="all" else [CASES[args.perf_case]])
    if args.run_msprof:
        for c in cases:
            subprocess.run([shutil.which("msprof") or "msprof",f"--output={args.msprof_output/c.name}",
                            "--ascendcl=on","--runtime-api=on","--task-time=l2","--ai-core=on",
                            "--aic-metrics=PipeUtilization",sys.executable,str(Path(__file__).resolve()),
                            "--performance","--perf-case",c.name,"--warmup",str(args.warmup),
                            "--repeat",str(args.repeat),"--rounds",str(args.rounds)],check=True)
    elif args.performance or args.extended:
        library=Path(dgl._ffi.base._LIB._name).resolve()
        report=dict(warmup=args.warmup,repeat=args.repeat,rounds=args.rounds,
                    test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    library=str(library),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),results=[])
        for c in cases:
            configs=([(torch.int32,"identity",c.weighted,torch.float32)] if not args.extended
                     else [(idtype,data,weighted,wtype)
                           for idtype in (torch.int32,torch.int64)
                           for data in ("none","random")
                           for weighted,wtype in ((False,torch.float32),
                                                 (True,torch.float32),(True,torch.float64))])
            for idtype,data,weighted,wtype in configs:
                custom=(Case(c.name,c.rows,c.cols,c.nnz,c.queries,weighted,
                             row_pattern=c.row_pattern) if args.extended else c)
                report['results'].append(performance(custom,args.warmup,args.repeat,
                                                     args.rounds,idtype,data,wtype))
                if args.output:
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
    else:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestCSRGetDataNPU)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
