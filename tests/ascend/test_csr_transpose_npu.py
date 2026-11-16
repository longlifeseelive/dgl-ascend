"""CSRTranspose correctness and msprof performance tests on Ascend NPU.

    python tests/ascend/test_csr_transpose_npu.py
    python tests/ascend/test_csr_transpose_npu.py --performance --perf-case long_tail_no_data
    python tests/ascend/test_csr_transpose_npu.py --run-msprof --msprof-output ./msprof_csr_transpose
    python tests/ascend/test_csr_transpose_npu.py --generalization --general-output /tmp/csr_transpose_general.json

Fresh CSR graphs prevent the cached CSC format from bypassing CSRTranspose.
Generation, upload and graph construction are outside the timed region;
g.adj_tensors("csc"), all output conversion, and final synchronization are timed.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import dgl
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

ETYPE = ("_U", "_E", "_V")
DTYPES = (torch.int32, torch.int64)
LARGE_INT64 = False
_STAGE_RUNTIME = None


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


NPU_AVAILABLE = _npu_is_available()


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


PERF_CASES = (
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
CASES = {case.name: case for case in PERF_CASES + EXTRA_CASES}
GENERALIZATION_CASES = PERF_CASES + EXTRA_CASES
ID_DTYPES = DTYPES
PERFORMANCE_CASES = {case.name: case for case in PERF_CASES}


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


class _TransposeStages:
    """Link the actual stage archive into a temporary test bridge, no new DGL API."""
    def __enter__(self):
        global _STAGE_RUNTIME
        if _STAGE_RUNTIME is None:
            root=Path(__file__).resolve().parents[2]
            archive=root/"build/lib/libdgl_ascendc_csr_transpose_kernel.a"
            if not archive.exists():
                raise unittest.SkipTest("CSRTranspose stage archive is required")
            toolkit=Path(os.environ.get("ASCEND_HOME_PATH","/usr/local/Ascend/ascend-toolkit/latest"))
            libdir=toolkit/"lib64"
            build=root/"build/csr_transpose_stage_test"
            build.mkdir(parents=True,exist_ok=True)
            source=build/"bridge.cpp";library=build/"bridge.so"
            source.write_text(r'''#include <cstdint>
extern "C" uint32_t aclrtlaunch_csr_transpose_count_int64_wide(uint32_t,void*,void*,void*,uint64_t,uint64_t,uint32_t);
extern "C" uint32_t aclrtlaunch_csr_transpose_prefix_int64_wide(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint64_t,uint32_t);
extern "C" uint32_t aclrtlaunch_csr_transpose_scatter_int64_wide(uint32_t,void*,void*,void*,void*,void*,void*,void*,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_csr_transpose_zero_int64(uint32_t,void*,void*,uint64_t);
#define EXPORTED extern "C" __attribute__((visibility("default"))) uint32_t
EXPORTED run_count(uint32_t b,void*s,void*i,void*c,uint64_t n,uint64_t f,uint32_t k){return aclrtlaunch_csr_transpose_count_int64_wide(b,s,i,c,n,f,k);}
EXPORTED run_prefix(uint32_t b,void*s,void*c,void*o,void*p,void*r,uint32_t n,uint64_t f,uint32_t k){return aclrtlaunch_csr_transpose_prefix_int64_wide(b,s,c,o,p,r,n,f,k);}
EXPORTED run_scatter(uint32_t b,void*s,void*p,void*i,void*d,void*o,void*x,void*y,uint64_t m,uint64_t n,uint64_t f,uint32_t k,uint32_t h){return aclrtlaunch_csr_transpose_scatter_int64_wide(b,s,p,i,d,o,x,y,m,n,f,k,h);}
EXPORTED run_zero(uint32_t b,void*s,void*p,uint64_t n){return aclrtlaunch_csr_transpose_zero_int64(b,s,p,n);}
''')
            digest=hashlib.sha256(source.read_bytes()+archive.read_bytes()).hexdigest()
            stamp=build/"sha256.txt"
            if not library.exists() or not stamp.exists() or stamp.read_text()!=digest:
                subprocess.run([os.environ.get("CXX","c++"),"-std=c++17","-shared","-fPIC",
                    "-fvisibility=hidden",str(source),str(archive),"-Wl,--exclude-libs,ALL",
                    "-L"+str(libdir),"-Wl,-rpath,"+str(libdir),"-lascendcl","-lruntime",
                    "-lc_sec","-lmmpa","-lunified_dlog","-ldl","-pthread","-o",str(library)],check=True)
                stamp.write_text(digest)
            bridge=ctypes.CDLL(str(library))
            acl=ctypes.CDLL(str(libdir/"libascendcl.so"))
            ptr,u32,u64=ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint64
            signatures={"count":[u32,ptr,ptr,ptr,u64,u64,u32],
                        "prefix":[u32,ptr,ptr,ptr,ptr,ptr,u32,u64,u32],
                        "scatter":[u32,ptr]+[ptr]*6+[u64]*3+[u32]*2,
                        "zero":[u32,ptr,ptr,u64]}
            for name,types in signatures.items():
                fn=getattr(bridge,"run_"+name);fn.argtypes=types;fn.restype=u32
            acl.aclrtCreateStream.argtypes=[ctypes.POINTER(ptr)]
            acl.aclrtSynchronizeStream.argtypes=[ptr]
            acl.aclrtDestroyStream.argtypes=[ptr]
            _STAGE_RUNTIME=bridge,acl
        self.bridge,self.acl=_STAGE_RUNTIME
        torch.npu.synchronize()
        self.stream=ctypes.c_void_p()
        assert self.acl.aclrtCreateStream(ctypes.byref(self.stream))==0
        return self

    def call(self,name,blocks,*args):
        torch.npu.synchronize()
        assert getattr(self.bridge,"run_"+name)(blocks,self.stream,*args)==0
        assert self.acl.aclrtSynchronizeStream(self.stream)==0

    def __exit__(self,*args):
        assert self.acl.aclrtDestroyStream(self.stream)==0


class TestCSRTransposeNPU(unittest.TestCase):
    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_int64_wide_prefix_boundaries(self):
        # Actual NPU prefix stages with counts/carries beyond both 32-bit ranges.
        counts_cpu=torch.tensor([[2**32+3,2**31+7,11,0,0,0,0,0],
                                 [19,2**33+5,23,0,0,0,0,0]],dtype=torch.int64)
        counts=counts_cpu.to("npu:0")
        total=int(counts_cpu.sum())
        with _TransposeStages() as stage:
            for initial in (0,2**40,2**63-total-32):
                first=0 if initial==0 else 1
                offsets=torch.empty_like(counts)
                ptr=torch.zeros(first+4,dtype=torch.int64,device="npu:0")
                carry=torch.zeros(8,dtype=torch.int64)
                carry[0]=initial;carry=carry.to("npu:0")
                stage.call("prefix",1,counts.data_ptr(),offsets.data_ptr(),ptr.data_ptr(),
                           carry.data_ptr(),2,first,3)
                expected=[initial]
                for col in range(3):expected.append(expected[-1]+int(counts_cpu[:,col].sum()))
                self.assertEqual(ptr[first:].cpu().tolist(),expected)
                for block in range(2):
                    self.assertEqual(offsets[block,:3].cpu().tolist(),
                        [expected[col]+sum(int(counts_cpu[b,col]) for b in range(block)) for col in range(3)])
                self.assertEqual(int(carry[0].cpu()),expected[-1])

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_int64_wide_count_and_scatter(self):
        # A high column band and high data payload exercise the real wide stages.
        first=2**40+5;nnz=37;blocks=3
        p=torch.tensor([0,17,21,37],dtype=torch.int64)
        j=first+torch.arange(nnz,dtype=torch.int64).remainder(3)
        a=2**48+torch.arange(nnz,dtype=torch.int64)
        p,j,a=(t.to("npu:0") for t in (p,j,a))
        counts=torch.empty((blocks,8),dtype=torch.int64,device="npu:0")
        offsets=torch.empty_like(counts)
        ptr=torch.zeros(4,dtype=torch.int64,device="npu:0")
        carry=torch.zeros(8,dtype=torch.int64,device="npu:0")
        idx=torch.empty(nnz,dtype=torch.int64,device="npu:0");data=torch.empty_like(idx)
        with _TransposeStages() as stage:
            stage.call("count",blocks,j.data_ptr(),counts.data_ptr(),nnz,first,3)
            stage.call("prefix",1,counts.data_ptr(),offsets.data_ptr(),ptr.data_ptr(),carry.data_ptr(),blocks,0,3)
            stage.call("scatter",blocks,p.data_ptr(),j.data_ptr(),a.data_ptr(),offsets.data_ptr(),
                       idx.data_ptr(),data.data_ptr(),3,nnz,first,3,1)
        order=[edge for col in range(3) for edge in range(nnz) if edge%3==col]
        self.assertEqual(ptr.cpu().tolist(),[0,13,25,37])
        self.assertEqual(idx.cpu().tolist(),[0 if e<17 else 1 if e<21 else 2 for e in order])
        self.assertEqual(data.cpu().tolist(),[2**48+e for e in order])

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_real_dimensions_above_uint32(self):
        if not LARGE_INT64:self.skipTest("enable --large-int64 for a 32 GiB boundary test")
        torch.npu.empty_cache()
        rows=2**32+1
        required=(rows+1)*8
        free,_=torch.npu.mem_get_info()
        if free<required+2**30:self.skipTest("32 GiB plus 1 GiB free NPU memory is required")
        # Full public Host path, not synthetic shape metadata or a cached result.
        with _TransposeStages() as stage:
            p=torch.empty(rows+1,dtype=torch.int64,device="npu:0")
            stage.call("zero",40,p.data_ptr(),p.numel())
            p[-1:]=1
            j=torch.tensor([2],dtype=torch.int64,device="npu:0")
            a=torch.empty(0,dtype=torch.int64,device="npu:0")
            g=dgl.heterograph({ETYPE:("csr",(p,j,a))},
                num_nodes_dict={"_U":rows,"_V":7},idtype=torch.int64,device="npu:0")
            result=g.adj_tensors("csc",etype=ETYPE)
            self.assertEqual(result[0].cpu().tolist(),[0,0,0,1,1,1,1,1])
            self.assertEqual(result[1].cpu().tolist(),[2**32])
            self.assertEqual(result[2].cpu().tolist(),[0])
            del result,g,p,j,a
            gc.collect();torch.npu.empty_cache()
        # Empty-edge transpose independently exercises a >uint32 output length.
        empty=torch.empty(0,dtype=torch.int64,device="npu:0")
        p=torch.zeros(1,dtype=torch.int64,device="npu:0")
        g=dgl.heterograph({ETYPE:("csr",(p,empty,empty))},
            num_nodes_dict={"_U":0,"_V":rows},idtype=torch.int64,device="npu:0")
        result=g.adj_tensors("csc",etype=ETYPE)
        self.assertEqual(result[0].numel(),rows+1)
        self.assertEqual(result[0][:2].cpu().tolist(),[0,0])
        self.assertEqual(result[0][-2:].cpu().tolist(),[0,0])
        self.assertEqual(result[1].numel(),0)
        del result,g
        gc.collect();torch.npu.empty_cache()

    @unittest.skipUnless(NPU_AVAILABLE, "Ascend NPU is required")
    def test_functional_matrix(self):
        torch.npu.set_device(0)
        for case in PERF_CASES + EXTRA_CASES:
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


def _run_performance_case(perf_name, warmup, repeat):
    """Fresh graphs ensure every measured call executes CSRTranspose."""
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    case = CASES[perf_name]
    dtype, device = torch.int32, torch.device("npu:0")
    torch.npu.set_device(device)
    csr = make_csr(case, dtype)
    graphs = [make_graph(case, csr, dtype, device) for _ in range(warmup + repeat)]
    print(f"PERF_CONFIG name={perf_name} shape=({case.rows},"
          f"{case.cols},{case.nnz}) dtype=int32 with_data={case.with_data} "
          f"warmup={warmup} repeat={repeat}", flush=True)
    for i in range(warmup):
        transpose(graphs[i])
    torch.npu.synchronize(device)
    print(f"PERF_START {perf_name}", flush=True)
    start = time.perf_counter()
    for i in range(repeat):
        transpose(graphs[warmup + i])
    torch.npu.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    average_ms = elapsed_ms / repeat
    print(f"PERF_END {perf_name} total_ms={elapsed_ms:.6f} "
          f"average_ms={average_ms:.6f}", flush=True)
    return average_ms


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
            csr = make_csr(case, dtype)
            p, _, a = csr
            degree = p[1:].to(torch.int64) - p[:-1].to(torch.int64)
            result = {"case": case.name, "dtype": str(dtype).split(".")[-1],
                      "M": case.rows, "N": case.cols, "E": case.nnz,
                      "row_pattern": case.rows_kind,
                      "data_pattern": "reverse" if case.with_data else "none",
                      "with_data": case.with_data, "seed": None,
                      "input_sha256": hashlib.sha256(b"".join(t.numpy().tobytes() for t in csr)).hexdigest(),
                      "nonempty_rows": int(degree.ne(0).sum()),
                      "max_degree": int(degree.max()) if degree.numel() else 0,
                      "adjacent_id_fraction": float((a[1:] - a[:-1]).abs().eq(1).float().mean())
                      if a.numel() > 1 else None}
            try:
                cpu = make_graph(case, csr, dtype, torch.device("cpu"))
                check = make_graph(case, csr, dtype, device)
                assert_result(case, dtype, transpose(check), transpose(cpu))
                del cpu, check
                times = []
                for _ in range(args.measurements):
                    graphs = [make_graph(case, csr, dtype, device)
                              for _ in range(args.warmup + args.repeat)]
                    for i in range(args.warmup):
                        transpose(graphs[i])
                    torch.npu.synchronize(device)
                    start = time.perf_counter()
                    for i in range(args.repeat):
                        transpose(graphs[args.warmup + i])
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
    parser.add_argument("--msprof-output", type=Path, default=Path("./msprof_csr_transpose"))

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
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCSRTransposeNPU)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.run_msprof:
        _run_msprof(args.msprof_output, selected_cases, args.warmup, args.repeat)
    else:
        for name in selected_cases:
            _run_performance_case(name, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
