"""Distribution/scale coverage and before/after CSRTranspose benchmarks.

Uses fresh CSR graphs, exact CPU references and uncached graph.adj_tensors('csc').
Only this operator's old host object is substituted when building a baseline;
working sources and the active libdgl.so are never overwritten.

Examples (after sourcing the existing Ascend/DGL environment):
  python tests/ascend/test_csr_transpose_generalization_npu.py --list
  python tests/ascend/test_csr_transpose_generalization_npu.py --build-baseline
  python tests/ascend/test_csr_transpose_generalization_npu.py \
      --library-dir build/csr_transpose_generalization_baseline --label baseline \
      --output-dir /workspace/outputs/csr_transpose/generalization/baseline
  python tests/ascend/test_csr_transpose_generalization_npu.py \
      --library-dir build --label optimized \
      --output-dir /workspace/outputs/csr_transpose/generalization/optimized \
      --compare /workspace/outputs/csr_transpose/generalization/baseline/results.json

New scenarios have no A100 measurements: the comparison is against the original
CSRToCOO/COOToCSR implementation on this NPU, including its CPU prefix/fallback.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    rows: int
    cols: int
    nnz: int
    row_pattern: str = "balanced"
    col_pattern: str = "random"


def scenarios():
    cases = [Scenario(f"edges_{e}", "edge_scale", 3079, 4093, e)
             for e in (0, 1, 31, 1023, 2049, 16381, 131071, 1048573, 4000003)]
    cases += [Scenario(f"rows_{pattern}", "row_distribution", 65537, 8191,
                       262139, pattern)
              for pattern in ("balanced", "random", "powerlaw", "first", "last",
                              "gaps", "sampled", "alternating")]
    cases += [Scenario(f"cols_{pattern}", "column_distribution", 8191, 12289,
                       262139, "random", pattern)
              for pattern in ("random", "ascending", "descending", "hotspot",
                              "single", "diagonal", "blocked")]
    cases += [Scenario(f"width_{n}", "column_bands", 4093, n, 131071)
              for n in (16383, 16384, 16385, 32767, 32768, 32769, 65537,
                        131071, 524287, 1048583)]
    cases += [
        Scenario("tall_sparse", "rectangular", 1000003, 127, 200003, "random"),
        Scenario("tall_one_column", "rectangular", 2000003, 1, 65539, "random"),
        Scenario("wide_few_rows", "rectangular", 7, 262147, 65539),
        Scenario("one_dense_row", "rectangular", 1, 1021, 1048573),
        Scenario("both_sparse", "rectangular", 1048583, 65537, 2049, "random"),
        Scenario("dense_257", "rectangular", 257, 257, 257 * 257,
                 "balanced", "dense"),
        Scenario("wide_last_column", "empty_bands", 997, 262147, 131071,
                 "random", "last"),
        Scenario("wide_hot_columns", "empty_bands", 997, 524287, 65539,
                 "random", "hotspot"),
    ]
    shapes = ((13, 17, 37), (997, 509, 8197), (5003, 8191, 70001),
              (17011, 16387, 170003), (70001, 70001, 300007),
              (110003, 997, 262147))
    cases += [Scenario(f"random_shape_{i}", "random_shapes", m, n, e, "random")
              for i, (m, n, e) in enumerate(shapes)]
    return cases


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_baseline(args):
    build = args.build_dir.resolve()
    destination = build / "csr_transpose_generalization_baseline"
    destination.mkdir(parents=True, exist_ok=True)
    relative = "src/array/ascend/csr_transpose.cc"
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline_revision}:{relative}"], cwd=REPO)
    if b"COOToCSR(COOTranspose(CSRToCOO(csr, false)))" not in source:
        raise RuntimeError("Selected revision is not the original CSRTranspose chain")
    source_path = destination / "csr_transpose.cc"
    source_path.write_bytes(source)
    values = {}
    for line in (build / "CMakeFiles/dgl.dir/flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            values[key] = shlex.split(value)
    compile_cmd = ["/usr/bin/c++", *values["CXX_DEFINES"],
                   *values["CXX_INCLUDES"], *values["CXX_FLAGS"],
                   "-I" + str(REPO / "src/array/ascend"), "-c", str(source_path),
                   "-o", str(destination / "csr_transpose.cc.o")]
    link_cmd = shlex.split((build / "CMakeFiles/dgl.dir/link.txt").read_text())
    original_object = "CMakeFiles/dgl.dir/src/array/ascend/csr_transpose.cc.o"
    if link_cmd.count(original_object) != 1:
        raise RuntimeError("Expected exactly one CSRTranspose host object")
    link_cmd[link_cmd.index(original_object)] = str(destination / "csr_transpose.cc.o")
    link_cmd[link_cmd.index("-o") + 1] = str(destination / "libdgl.so")
    active_hash = sha256(build / "libdgl.so")
    with (destination / "build.log").open("w") as log:
        subprocess.run(compile_cmd, cwd=build, stdout=log, stderr=log, check=True)
        subprocess.run(link_cmd, cwd=build, stdout=log, stderr=log, check=True)
    if sha256(build / "libdgl.so") != active_hash:
        raise RuntimeError("The active library was unexpectedly modified")
    save_json(destination / "manifest.json", dict(
        baseline_revision=args.baseline_revision, source_sha256=sha256(source_path),
        library_sha256=sha256(destination / "libdgl.so"),
        optimized_library_sha256=active_hash, compile_argv=compile_cmd,
        link_argv=link_cmd, explanation="Only the CSRTranspose host object differs"))
    print(destination / "libdgl.so")


def balanced(m, e):
    counts = torch.full((m,), e // m, dtype=torch.int64)
    counts[:e % m] += 1
    return counts


def make_base(case, seed):
    rng = torch.Generator().manual_seed(seed)
    m, n, e = case.rows, case.cols, case.nnz
    pattern = case.row_pattern
    if pattern == "balanced":
        counts = balanced(m, e)
    elif pattern == "random":
        counts = torch.bincount(torch.randint(m, (e,), generator=rng), minlength=m)
    elif pattern == "powerlaw":
        weights = torch.arange(1, m + 1, dtype=torch.float64).pow(-1.3)
        picks = torch.multinomial(weights, e, replacement=True, generator=rng)
        counts = torch.bincount(picks, minlength=m)[torch.randperm(m, generator=rng)]
    elif pattern in ("first", "last"):
        counts = torch.zeros(m, dtype=torch.int64)
        counts[0 if pattern == "first" else m - 1] = e
    elif pattern == "gaps":
        counts = torch.zeros(m, dtype=torch.int64)
        active = torch.tensor([0, m // 4, m // 2, m - 1])
        counts[active] = balanced(len(active), e)
    elif pattern == "sampled":
        hot_rows, hot_edges = max(1, m // 100), e * 95 // 100
        counts = torch.cat((balanced(hot_rows, hot_edges),
                            balanced(m - hot_rows, e - hot_edges)))
    elif pattern == "alternating":
        counts = torch.zeros(m, dtype=torch.int64)
        counts[::2] = balanced((m + 1) // 2, e)
    else:
        raise ValueError(pattern)
    p = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    j = torch.randint(n, (e,), generator=rng)
    pattern = case.col_pattern
    if pattern in ("ascending", "descending"):
        j = j.sort(descending=pattern == "descending").values
    elif pattern == "hotspot":
        mask = torch.rand(e, generator=rng) < .99
        j[mask] = torch.randint(min(3, n), (int(mask.sum()),), generator=rng)
    elif pattern == "single":
        j.fill_(n // 2)
    elif pattern == "last":
        j.fill_(n - 1)
    elif pattern == "diagonal":
        j = torch.repeat_interleave(torch.arange(m, dtype=torch.int64), counts) % n
    elif pattern == "blocked":
        j = (torch.arange(e, dtype=torch.int64) // 64 * 31) % n
    elif pattern == "dense":
        j = torch.arange(e, dtype=torch.int64) % n
    elif pattern != "random":
        raise ValueError(pattern)
    stats = dict(nonempty_rows=int(counts.ne(0).sum()),
                 max_row_degree=int(counts.max()),
                 touched_columns=int(j.unique().numel()),
                 bands=(n + 16383) // 16384)
    if int(p[-1]) != e:
        raise AssertionError("Invalid generated CSR")
    return p, j, stats


def make_data(e, kind, seed):
    if kind == "none":
        return torch.empty(0, dtype=torch.int64)
    if kind == "random":
        return torch.randperm(e, generator=torch.Generator().manual_seed(seed))
    values = torch.arange(e, dtype=torch.int64)
    return values if kind == "identity" else values.flip(0)


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo) if lo != hi else ordered[lo]


def check_and_measure(case, csr, dtype, args, helpers):
    cpu = torch.device("cpu")
    npu = torch.device(f"npu:{args.device}")
    expected = helpers.transpose(helpers.make_graph(case, csr, dtype, cpu))
    device_csr = tuple(t.to(npu) for t in csr)
    check_graph = helpers.make_graph(case, device_csr, dtype, npu)
    actual = helpers.transpose(check_graph)
    helpers.assert_result(case, dtype, actual, expected)
    del actual, check_graph, expected
    round_ms, samples = [], []
    for _ in range(args.rounds):
        # Shared immutable CSR tensors; every graph has its own empty CSC cache.
        graphs = [helpers.make_graph(case, device_csr, dtype, npu)
                  for _ in range(args.warmup + args.repeat)]
        for graph in graphs:
            if graph.formats()["created"] != ["csr"]:
                raise AssertionError("A benchmark graph already has a converted format")
        for graph in graphs[:args.warmup]:
            helpers.transpose(graph)
        torch.npu.synchronize(npu)
        current = []
        for graph in graphs[args.warmup:]:
            start = time.perf_counter_ns()
            result = helpers.transpose(graph)
            torch.npu.synchronize(npu)
            current.append((time.perf_counter_ns() - start) / 1e6)
            # Output release/garbage collection is outside the measurement.
            del result
        round_ms.append(statistics.mean(current))
        samples.extend(current)
        del graph, graphs
        gc.collect()
    return dict(correct=True, round_ms=round_ms, samples_ms=samples,
                mean_ms=statistics.mean(round_ms), median_ms=statistics.median(samples),
                p95_ms=percentile(samples, .95), min_ms=min(samples), max_ms=max(samples),
                million_edges_per_s=(case.nnz / statistics.mean(round_ms) / 1000))


def write_results(output, report):
    save_json(output / "results.json", report)
    flattened = []
    for row in report["results"]:
        flat = {k: v for k, v in row.items() if k not in ("samples_ms", "round_ms")}
        flat["round_ms"] = json.dumps(row.get("round_ms", []))
        flattened.append(flat)
    if flattened:
        columns = list(dict.fromkeys(k for row in flattened for k in row))
        with (output / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flattened)


def compare(output, report, reference_path):
    reference = json.loads(reference_path.read_text())
    for key in ("warmup", "repeat", "rounds", "seed"):
        if reference[key] != report[key]:
            raise ValueError(f"Inconsistent benchmark configuration: {key}")
    if reference["library_sha256"] == report["library_sha256"]:
        raise ValueError("Both measurements loaded the same library")
    old = {r["key"]: r for r in reference["results"]}
    if set(old) != {r["key"] for r in report["results"]}:
        raise ValueError("The two reports cover different cases")
    rows = []
    for new in report["results"]:
        base = old[new["key"]]
        if new.get("input_sha256") != base.get("input_sha256"):
            raise ValueError("Input hash mismatch: " + new["key"])
        item = {k: v for k, v in new.items() if k not in ("samples_ms", "round_ms")}
        item["baseline_ms"] = base.get("mean_ms")
        item["baseline_correct"] = base.get("correct", False)
        if new.get("correct") and base.get("correct"):
            item["speedup"] = base["mean_ms"] / new["mean_ms"]
            item["classification"] = ("improved" if item["speedup"] >= 1.1 else
                                      "regressed" if item["speedup"] < 1 / 1.1 else "similar")
        else:
            item.update(speedup=None, classification="correctness_failure")
        rows.append(item)
    valid = [r for r in rows if r["speedup"] is not None]
    speeds = [r["speedup"] for r in valid]
    summary = dict(total=len(rows),correct=len(valid),
                   faster=sum(s > 1 for s in speeds),
                   improved=sum(r["classification"] == "improved" for r in rows),
                   similar=sum(r["classification"] == "similar" for r in rows),
                   regressed=sum(r["classification"] == "regressed" for r in rows),
                   geometric_mean_speedup=math.exp(statistics.mean(math.log(s) for s in speeds)) if speeds else None,
                   median_speedup=statistics.median(speeds) if speeds else None,
                   minimum_speedup=min(speeds) if speeds else None)
    save_json(output / "comparison.json", dict(summary=summary, results=rows,
               baseline_report=str(reference_path.resolve())))
    with (output / "comparison.csv").open("w", newline="") as f:
        columns = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# CSRTranspose 扩展场景同输入对照", "",
             "对照对象为原 CSRToCOO/COOToCSR 链路；其含 CPU 前缀和，宽矩阵可能走 CPU 回退。此对照不是 A100。",
             "两边均使用新图，禁用 COOToCSR 缓存，计时包含完整图接口、输出导出和每次最终同步。", "",
             f"共 {summary['total']} 项，双方正确性通过 {summary['correct']} 项；"
             f"提速 ≥10%：{summary['improved']}，±10% 内：{summary['similar']}，"
             f"退化 >10%：{summary['regressed']}。", "",
             f"加速比几何平均：{summary['geometric_mean_speedup']:.3f}×；"
             f"中位数：{summary['median_speedup']:.3f}×；最低：{summary['minimum_speedup']:.3f}×。", "",
             "| 类别 | 项数 | ≥10% 提速 | >10% 退化 | 加速比几何平均 |", "|---|---:|---:|---:|---:|"]
    for family in dict.fromkeys(r["family"] for r in rows):
        group = [r for r in valid if r["family"] == family]
        geo = math.exp(statistics.mean(math.log(r["speedup"]) for r in group)) if group else float("nan")
        lines.append(f"| {family} | {len(group)} | {sum(r['classification']=='improved' for r in group)} | "
                     f"{sum(r['classification']=='regressed' for r in group)} | {geo:.3f}× |")
    lines += ["", "| 场景 | M,N,E | 类型/data | 旧实现 ms | 当前 NPU ms | p95 ms | 加速比 | 判定 |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["speedup"] is None:
            lines.append(f"| {r['key']} | | | | | | | 正确性失败 |")
        else:
            lines.append(f"| {r['name']} | {r['rows']},{r['cols']},{r['nnz']} | {r['dtype']}/{r['data']} | "
                         f"{r['baseline_ms']:.6f} | {r['mean_ms']:.6f} | {r['p95_ms']:.6f} | {r['speedup']:.3f}× | {r['classification']} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("COMPARISON", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--build-baseline", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=REPO / "build")
    parser.add_argument("--baseline-revision", default="HEAD")
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--label", default="optimized")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/outputs/csr_transpose/generalization/optimized"))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--case", default="all", help="Comma separated names, or all")
    parser.add_argument("--dtypes", default="int32,int64")
    parser.add_argument("--data", default="none,random", help="none,identity,reverse,random")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    cases = scenarios()
    if args.list:
        for c in cases:
            print(json.dumps(asdict(c)))
        print(f"{len(cases)} scenarios; default {len(cases)*4} dtype/data combinations")
        return
    if args.build_baseline:
        build_baseline(args)
        return
    if args.warmup < 0 or args.repeat < 1 or args.rounds < 1:
        parser.error("Invalid warmup/repeat/rounds")
    names = set(args.case.split(",")) if args.case != "all" else {c.name for c in cases}
    if names - {c.name for c in cases}:
        parser.error("Unknown scenarios: " + str(names - {c.name for c in cases}))
    dtype_names, data_names = args.dtypes.split(","), args.data.split(",")
    if set(dtype_names) - {"int32", "int64"} or set(data_names) - {"none", "identity", "reverse", "random"}:
        parser.error("Invalid dtype/data selection")
    if args.library_dir:
        os.environ["DGL_LIBRARY_PATH"] = str(args.library_dir.resolve())
    # Set before importing DGL: identities must never hit the old conversion cache.
    os.environ["DGL_COO2CSR_CACHE_DISABLE"] = "1"
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("DGL_FFI", "ctypes")
    os.environ.setdefault("DGL_ASCEND_STREAM_BRIDGE", "1")
    global torch
    import torch
    import torch_npu
    import dgl
    import test_csr_transpose_npu as helpers
    torch.npu.set_device(args.device)
    library = Path(dgl._ffi.base._LIB._name).resolve()
    if args.library_dir and library != (args.library_dir / "libdgl.so").resolve():
        raise RuntimeError(f"Unexpected loaded library: {library}")
    report = dict(label=args.label, warmup=args.warmup, repeat=args.repeat,
                  rounds=args.rounds, seed=args.seed, device=args.device,
                  library=str(library), library_sha256=sha256(library),
                  test_sha256=sha256(__file__), helper_sha256=sha256(helpers.__file__),
                  torch=torch.__version__, torch_npu=torch_npu.__version__,
                  dgl_python=dgl.__file__, cache_disabled=True,
                  source_sha256={name:sha256(REPO / "src/array/ascend" / name)
                                 for name in ("csr_transpose.cc", "csr_transpose_kernel.cpp")},
                  results=[])
    manifest = library.parent / "manifest.json"
    if manifest.exists():
        report["baseline_build"] = json.loads(manifest.read_text())
    print("ENV", json.dumps({k:v for k,v in report.items() if k != "results"}), flush=True)
    failures = 0
    for index, case in enumerate(cases):
        if case.name not in names:
            continue
        seed = args.seed + index * 101
        p, j, stats = make_base(case, seed)
        for data_name in data_names:
            a = make_data(case.nnz, data_name, seed + 17)
            for dtype_name in dtype_names:
                dtype = getattr(torch, dtype_name)
                csr = tuple(t.to(dtype) for t in (p, j, a))
                digest = hashlib.sha256()
                for t in csr:
                    digest.update(t.numpy().tobytes())
                row = dict(**asdict(case), **stats, seed=seed, dtype=dtype_name,
                           data=data_name, key=f"{case.name}/{dtype_name}/{data_name}",
                           input_sha256=digest.hexdigest())
                try:
                    row.update(check_and_measure(case, csr, dtype, args, helpers))
                    print(f"PASS {row['key']} mean_ms={row['mean_ms']:.6f} p95_ms={row['p95_ms']:.6f}", flush=True)
                except Exception as exc:
                    failures += 1
                    row.update(correct=False,error=repr(exc))
                    print(f"FAIL {row['key']}: {exc}", flush=True)
                report["results"].append(row)
                write_results(args.output_dir, report)
                del csr
                gc.collect()
        del p, j, a
    if args.compare:
        compare(args.output_dir, report, args.compare)
    print(f"FINISHED total={len(report['results'])} failures={failures}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
