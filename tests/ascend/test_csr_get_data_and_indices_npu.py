"""CSRGetDataAndIndices correctness and msprof performance tests on Ascend NPU.

python tests/ascend/test_csr_get_data_and_indices_npu.py
python tests/ascend/test_csr_get_data_and_indices_npu.py --performance --perf-case typical
python tests/ascend/test_csr_get_data_and_indices_npu.py --run-msprof --msprof-output ./msprof_csr_get_data_and_indices
python tests/ascend/test_csr_get_data_and_indices_npu.py --generalization --general-output /tmp/csr_get_data_and_indices_general.json

The default timed entry is g.edge_ids(..., return_uv=True), including DGL's
public validation and generic FFI. Queries are unique, as required by csr.h.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Sequence
import sys
import time
import unittest
from dataclasses import dataclass
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None
import dgl

ETYPE=('s','r','t')
INTERFACE='public'
LARGE_INT64=False


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


NPU_AVAILABLE = _npu_is_available()
ID_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class Case:
    name:str
    rows:int
    cols:int
    nnz:int
    queries:int
    pattern:str='balanced'
    a100_ms:float|None=None
PERF_CASES=(Case('typical',10000,10000,500000,65536,a100_ms=1.239220),
            Case('unsorted',4096,4096,65536,32768,a100_ms=.873748),
            Case('sampled',12000,2000,60000,60000,'sampled',1.069330))
EXTRA_CASES=(Case('empty_queries',17,31,43,0),Case('empty_edges',17,31,0,39),
             Case('empty_graph',0,0,0,0),Case('odd_tail',7,17,23,37),
             Case('long_row',37,65537,32771,4099,'hub'),
             Case('tall_sparse',200003,257,2049,4099),
             Case('large_batch',1009,1031,65539,131073),
             Case('parallel_edges',257,509,8193,257,'duplicates'),
             Case('one_column',37,1,8193,37))
CASES={c.name:c for c in PERF_CASES+EXTRA_CASES}
GENERALIZATION_CASES = PERF_CASES + EXTRA_CASES
PERFORMANCE_CASES = {case.name: case for case in PERF_CASES}

def balanced(n,e):
    if not n:return torch.empty(0,dtype=torch.int64)
    x=torch.full((n,),e//n,dtype=torch.int64);x[:e%n]+=1
    return x

def make_inputs(case,dtype=torch.int32,data_kind='identity',seed=20261115):
    rng=torch.Generator().manual_seed(seed)
    if case.pattern=='hub':
        deg=torch.zeros(case.rows,dtype=torch.int64);deg[-1]=case.nnz
    elif case.pattern=='sampled':
        hot=max(1,case.rows//10);edges=case.nnz*4//5
        deg=torch.cat((balanced(hot,edges),balanced(case.rows-hot,case.nnz-edges)))
    else:deg=balanced(case.rows,case.nnz)
    p=torch.cat((torch.zeros(1,dtype=torch.int64),deg.cumsum(0)))
    pos=torch.arange(case.nnz,dtype=torch.int64)
    source=torch.repeat_interleave(torch.arange(case.rows),deg)
    reserve=min(max(0,case.cols-1),max(1,(case.queries+max(1,case.rows)-1)//max(1,case.rows)))
    active=max(1,case.cols-reserve)
    multiplier=104729
    while math.gcd(multiplier,active)!=1:multiplier+=1
    within=pos-p[source] if case.nnz else pos
    j=(within*multiplier+source*31+8191).remainder(active)
    if case.pattern=='duplicates':j=j.remainder(min(5,active))
    if data_kind=='none':a=torch.empty(0,dtype=torch.int64)
    elif data_kind=='identity':a=pos
    elif data_kind=='reverse':a=pos.flip(0)
    else:a=torch.randperm(case.nnz,generator=rng)
    if not case.queries:r=c=torch.empty(0,dtype=torch.int64)
    elif case.cols==1:
        assert case.queries<=case.rows
        r=torch.randperm(case.rows,generator=rng)[:case.queries];c=torch.zeros_like(r)
    else:
        pairs=torch.unique(source*case.cols+j)
        h=min((case.queries+1)//2,pairs.numel())
        hits=pairs[torch.randperm(pairs.numel(),generator=rng)[:h]]
        missing=case.queries-h
        assert missing<=case.rows*reserve
        seq=torch.arange(missing,dtype=torch.int64)
        misses=(seq//reserve)*case.cols+active+seq.remainder(reserve)
        flat=torch.cat((hits,misses))
        flat=flat[torch.randperm(flat.numel(),generator=rng)]
        assert torch.unique(flat).numel()==case.queries
        r,c=flat//case.cols,flat.remainder(case.cols)
    return tuple(t.to(dtype) for t in (p,j,a,r,c))

def packed(p,j,a,r,c,m,n,sorted=False):
    matrix=dgl.ndarray.SparseMatrix.__new__(dgl.ndarray.SparseMatrix)
    matrix.__setstate__((2,m,n,(p,j,a),[sorted]))
    fn=dgl._ffi.function.get_global_func('ascend._CAPI_DGLCSRGetDataAndIndices')
    result=fn(matrix,dgl.backend.to_dgl_nd(r),dgl.backend.to_dgl_nd(c))
    return tuple(dgl.backend.from_dgl_nd(result(i)) for i in range(3))

def runner(case,values,device,interface=None):
    p,j,a,r,c=(t.to(device) for t in values)
    g=dgl.heterograph({ETYPE:('csr',(p,j,a))},
        num_nodes_dict={'s':case.rows,'t':case.cols},idtype=p.dtype,device=device)
    route=interface or INTERFACE
    if route=='public':
        def run():return g.edge_ids(r,c,return_uv=True,etype=ETYPE)
    elif route=='graph':
        def run():return g._graph.edge_ids_all(0,r,c)
    else:
        def run():return packed(p,j,a,r,c,case.rows,case.cols)
    return run,(g,p,j,a,r,c)

def exact(actual,expected):
    assert len(actual)==len(expected)==3
    for x,y in zip(actual,expected):
        assert x.device.type=='npu',(x.device,x.shape)
        assert x.dtype==y.dtype and x.shape==y.shape,(x.dtype,y.dtype,x.shape,y.shape)
        assert torch.equal(x.cpu(),y), 'Ordered output differs from DGL CPU'

def oracle(p,j,a,r,c):
    if not r.numel() or not c.numel():return tuple(torch.empty(0,dtype=p.dtype) for _ in range(3))
    rr,cc=torch.broadcast_tensors(r,c);out=[[],[],[]]
    for row,col in zip(rr.tolist(),cc.tolist()):
        for edge in range(int(p[row]),int(p[row+1])):
            if int(j[edge])==col:
                out[0].append(row);out[1].append(col);out[2].append(int(a[edge]) if a.numel() else edge)
    return tuple(torch.tensor(x,dtype=p.dtype) for x in out)

@unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
class TestCSRGetDataAndIndicesNPU(unittest.TestCase):
    def test_functional_matrix(self):
        for case in PERF_CASES+EXTRA_CASES:
            for dtype in (torch.int32,torch.int64):
                for data in ('none','identity','random'):
                    with self.subTest(case=case.name,dtype=dtype,data=data):
                        values=make_inputs(case,dtype,data)
                        cpu,_=runner(case,values,'cpu')
                        npu,_=runner(case,values,'npu:0')
                        exact(npu(),cpu())

    def test_sorted_broadcast_and_all_parallel_edges(self):
        for dtype in (torch.int32,torch.int64):
            p=torch.tensor([0,5,7,7],dtype=dtype)
            for sorted_cols in (False,True):
                j=torch.tensor([1,1,2,3,3,0,2] if sorted_cols else [3,1,3,2,1,2,0],dtype=dtype)
                for data in (torch.empty(0,dtype=dtype),torch.tensor([6,4,2,0,5,1,3],dtype=dtype)):
                    for rows,cols in (([0],[1,2,3,0]),([0,1,2],[2]),([0,1,2],[3,0,1]),([],[]),([],[0]),([0],[])):
                        r=torch.tensor(rows,dtype=dtype);c=torch.tensor(cols,dtype=dtype)
                        expected=oracle(p,j,data,r,c)
                        actual=packed(*(t.to('npu:0') for t in (p,j,data,r,c)),3,4,sorted_cols)
                        exact(actual,expected)
            # One query must return thousands of parallel edges, not just one.
            p=torch.tensor([0,4099],dtype=dtype);j=torch.full((4099,),3,dtype=dtype)
            a=torch.arange(4099,dtype=dtype).flip(0);r=torch.tensor([0,0],dtype=dtype);c=torch.tensor([3,4],dtype=dtype)
            for flag in (False,True):
                exact(packed(*(t.to('npu:0') for t in (p,j,a,r,c)),1,7,flag),oracle(p,j,a,r,c))

    def test_invalid_queries(self):
        for dtype in (torch.int32,torch.int64):
            case=Case('errors',3,7,11,3);values=make_inputs(case,dtype)
            _,owners=runner(case,values,'npu:0');g=owners[0]
            for row,col in ((-1,0),(3,0),(0,-1),(0,7)):
                r=torch.tensor([row],dtype=dtype,device='npu:0');c=torch.tensor([col],dtype=dtype,device='npu:0')
                with self.assertRaisesRegex(dgl.DGLError,'Invalid (row|col) index'):
                    g._graph.edge_ids_all(0,r,c)
            r=torch.tensor([0,1],dtype=dtype,device='npu:0');c=torch.tensor([1,2,3],dtype=dtype,device='npu:0')
            with self.assertRaisesRegex(dgl.DGLError,'Invalid row and col id array'):
                g._graph.edge_ids_all(0,r,c)
            npu,_=runner(case,values,'npu:0');cpu,_=runner(case,values,'cpu');exact(npu(),cpu())

    def test_public_input_forms_and_errors(self):
        for dtype in (torch.int32,torch.int64):
            case=Case('public',3,7,11,3)
            values=make_inputs(case,dtype)
            cpu=runner(case,values,'cpu')[1][0]
            npu=runner(case,values,'npu:0')[1][0]
            for rows,cols in ((0,1),([0,1],[1,0]),([],[]),([],[0]),([0],[])):
                exact(npu.edge_ids(rows,cols,return_uv=True),cpu.edge_ids(rows,cols,return_uv=True))
            for rows,cols in (([-1],[0]),([3],[0]),([0],[-1]),([0],[7]),([],[7]),([3],[]),([0,1],[1,2,3])):
                for g in (cpu,npu):
                    with self.assertRaises(dgl.DGLError):g.edge_ids(rows,cols,return_uv=True)
        # The public NPU path must retain exact high int64 column bounds too.
        dtype=torch.int64
        p=torch.tensor([0,2],dtype=dtype)
        j=torch.tensor([2**40+3,2**40+4],dtype=dtype)
        a=torch.tensor([1,0],dtype=dtype)
        cpu=dgl.heterograph({ETYPE:('csr',(p,j,a))},num_nodes_dict={'s':1,'t':2**40+5},idtype=dtype)
        npu=cpu.to('npu:0')
        rows=[0,0];cols=[2**40+3,2**40+4]
        exact(npu.edge_ids(rows,cols,return_uv=True),cpu.edge_ids(rows,cols,return_uv=True))

    def test_int64_high_columns_and_data(self):
        p=torch.tensor([0,4,5],dtype=torch.int64)
        j=torch.tensor([2**40+3,7,2**40+3,2**32+7,2**40+3],dtype=torch.int64)
        a=torch.tensor([2**48+5,2**48+7,2**48+11,2**48+13,2**48+17],dtype=torch.int64)
        r=torch.tensor([0,0,1,0],dtype=torch.int64);c=torch.tensor([2**40+3,2**32+7,2**40+3,8],dtype=torch.int64)
        exact(packed(*(t.to('npu:0') for t in (p,j,a,r,c)),2,2**41),oracle(p,j,a,r,c))

    def test_real_int64_row_above_uint32(self):
        if not LARGE_INT64:self.skipTest('enable --large-int64 (uses 32 GiB NPU memory)')
        torch.npu.empty_cache()
        rows=2**32+1
        nbytes=(rows+1)*8
        free,_=torch.npu.mem_get_info()
        if free<nbytes+2**30:self.skipTest('32 GiB plus 1 GiB free NPU memory required')
        p=torch.empty(rows+1,dtype=torch.int64,device='npu:0')
        torch.npu.synchronize()
        acl=ctypes.CDLL('libascendcl.so')
        acl.aclrtMemset.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_size_t]
        self.assertEqual(acl.aclrtMemset(p.data_ptr(),nbytes,0,nbytes),0)
        p[-1:]=1
        j=torch.tensor([2],dtype=torch.int64,device='npu:0')
        a=torch.empty(0,dtype=torch.int64,device='npu:0')
        g=dgl.heterograph({ETYPE:('csr',(p,j,a))},num_nodes_dict={'s':rows,'t':7},
                          idtype=torch.int64,device='npu:0')
        result=g.edge_ids([2**32],[2],return_uv=True)
        expected=tuple(torch.tensor(x,dtype=torch.int64) for x in ([2**32],[2],[0]))
        exact(result,expected)
        del result,g,p,j,a
        gc.collect();torch.npu.empty_cache()

    def test_stream_mutation_and_lifetime(self):
        for dtype in (torch.int32,torch.int64):
            case=Case('mutation',17,257,2053,129)
            values=make_inputs(case,dtype,'random')
            npu,owners=runner(case,values,'npu:0');cpu,_=runner(case,values,'cpu')
            exact(npu(),cpu())
            stream=torch.npu.Stream()
            degrees=values[0].diff().roll(3)
            ptr=torch.cat((values[0][:1],degrees.cumsum(0).to(dtype)))
            changed=(ptr,(values[1]+1).remainder(case.cols),values[2].flip(0))+values[3:]
            with torch.npu.stream(stream):
                for target,value in zip(owners[1:4],changed[:3]):target.copy_(value.to('npu:0'))
            cpu,_=runner(case,changed,'cpu');result=npu();expected=cpu();exact(result,expected)
            del npu,owners
            gc.collect();torch.npu.empty_cache();exact(result,expected)

    def test_heterogeneous_formats_and_noncontiguous_queries(self):
        for dtype in (torch.int32,torch.int64):
            case=Case('formats',7,17,23,17)
            values=make_inputs(case,dtype,'random');g=runner(case,values,'cpu')[1][0]
            for fmt in ('csr','csc','coo'):
                cpu=g.formats(fmt);npu=cpu.to('npu:0');r,c=values[-2:]
                rn=r.repeat_interleave(2).to('npu:0')[::2];cn=c.repeat_interleave(2).to('npu:0')[::2]
                expected=cpu.edge_ids(r,c,return_uv=True)
                actual=npu.edge_ids(rn,cn,return_uv=True);exact(actual,expected)
                self.assertEqual(cpu.formats(),npu.formats())
            p,j,a=values[:3]
            specs={('s','a','t'):('csr',(p,j,a)),('s','z','t'):('csr',(p,j,a.flip(0)))}
            cpu=dgl.heterograph(specs,num_nodes_dict={'s':7,'t':17},idtype=dtype);npu=cpu.to('npu:0')
            r,c=values[-2:]
            for etype in (('s','z','t'),('s','a','t')):
                exact(npu.edge_ids(r.to('npu:0'),c.to('npu:0'),return_uv=True,etype=etype),
                      cpu.edge_ids(r,c,return_uv=True,etype=etype))


def _run_performance_case(perf_name, warmup, repeat):
    """Time the public all-edge query, using CSRToCOO's output protocol."""
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    case = CASES[perf_name]
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    values = make_inputs(case, torch.int32)
    run, owners = runner(case, values, device, interface="public")
    print(f"PERF_CONFIG name={perf_name} shape=({case.rows},"
          f"{case.cols},{case.nnz}) dtype=int32 Q={case.queries} "
          f"warmup={warmup} repeat={repeat}", flush=True)
    for _ in range(warmup):
        run()
    torch.npu.synchronize(device)
    print(f"PERF_START {perf_name}", flush=True)
    start = time.perf_counter()
    for _ in range(repeat):
        run()
    torch.npu.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    average_ms = elapsed_ms / repeat
    print(f"PERF_END {perf_name} total_ms={elapsed_ms:.6f} "
          f"average_ms={average_ms:.6f}", flush=True)
    return average_ms


def _general_configs(case):
    return ("none", "random")


def _general_runner(case, dtype, device, config):
    values = make_inputs(case, dtype, config)
    cpu, cpu_owners = runner(case, values, "cpu", interface="public")
    run, owners = runner(case, values, device, interface="public")
    expected = cpu()
    exact(run(), expected)
    p, _, a = values[:3]
    degree = p[1:].to(torch.int64) - p[:-1].to(torch.int64)
    metadata = {"Q": case.queries, "output_matches": expected[0].numel(),
                "row_pattern": case.pattern, "data_pattern": config,
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
    parser.add_argument("--msprof-output", type=Path, default=Path("./msprof_csr_get_data_and_indices"))

    parser.add_argument("--large-int64", action="store_true",
                        help="Run real >uint32 dimensions (uses 32 GiB NPU memory)")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.measurements <= 0:
        parser.error("warmup must be >= 0; repeat and measurements must be > 0")
    if args.generalization and args.general_output is None:
        parser.error("--generalization requires --general-output")
    return args


def main():
    args = _parse_args()
    global LARGE_INT64
    LARGE_INT64 = args.large_int64
    if args.generalization:
        _run_generalization(args)
        return
    selected_cases = (tuple(PERFORMANCE_CASES) if args.perf_case == "all"
                      else (args.perf_case,))

    if not args.performance and not args.run_msprof:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCSRGetDataAndIndicesNPU)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.run_msprof:
        _run_msprof(args.msprof_output, selected_cases, args.warmup, args.repeat)
    else:
        for name in selected_cases:
            _run_performance_case(name, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
