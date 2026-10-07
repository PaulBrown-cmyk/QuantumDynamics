#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
root=$PWD
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

python_cmd=
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import h5py, numpy' >/dev/null 2>&1; then
    python_cmd=$candidate
    break
  fi
done
test -n "$python_cmd" || { echo 'h5py/numpy Python required' >&2; exit 1; }
test -x "${QLE_EXE:?QLE_EXE must name parallel-HDF5 executable}"

cd "$work"
OMPI_MCA_btl=self,sm OMP_NUM_THREADS=1 "${MPIEXEC:-mpiexec}" -n 2 \
  "$QLE_EXE" "$root/tests/INPUT.parallel-hdf5.nml" > run.log
"$python_cmd" "$root/tests/check_parallel_hdf5.py" parallel.ensemble.h5
