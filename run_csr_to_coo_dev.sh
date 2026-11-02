#!/bin/bash
set -euo pipefail

cd /workspace/dgl-ascend

export DGL_HOME="$PWD"
export DGL_LIBRARY_PATH="$PWD/build"
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "========== 1. Build =========="
bash script/build_dgl_ascend.sh -t release -j 4 \
    2>&1 | tee /tmp/csr_build.log

echo "========== 2. Check DGL =========="
python3 - <<'PY'
from pathlib import Path
import dgl
from dgl._ffi.base import _LIB

repo = Path("/workspace/dgl-ascend").resolve()
python_path = Path(dgl.__file__).resolve()

# 从 Linux 进程内存映射确认真正加载的动态库
libraries = set()

for line in Path("/proc/self/maps").read_text().splitlines():
    fields = line.split(maxsplit=5)

    if len(fields) == 6 and "libdgl.so" in fields[5]:
        path = fields[5].removesuffix(" (deleted)")
        if path.startswith("/"):
            libraries.add(Path(path).resolve())

print("DGL Python:", python_path)
print("Library handle:", _LIB._name)
print("Loaded libraries:", sorted(map(str, libraries)))

assert python_path.is_relative_to(repo), \
    "DGL Python source path is incorrect"

assert libraries, \
    "No loaded libdgl.so found"

assert all(p.is_relative_to(repo) for p in libraries), \
    "Loaded libdgl.so is not from dgl-ascend"

print("DGL path check: PASS")
PY

echo "========== 3. Correctness =========="
python3 tests/ascend/test_csr_to_coo_npu.py \
    2>&1 | tee /tmp/csr_correctness.log

echo "========== 4. Performance =========="
python3 tests/ascend/test_csr_to_coo_npu.py \
    --performance \
    --perf-case all \
    --warmup 20 \
    --repeat 100 \
    2>&1 | tee /tmp/csr_perf.log

echo "========== Results =========="
grep "PERF_END" /tmp/csr_perf.log

echo "All checks completed."
