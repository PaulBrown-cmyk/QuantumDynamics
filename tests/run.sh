#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
root=$PWD
fftw_cppflags=
fftw_ldflags=
if [ -n "${FFTW_PREFIX:-}" ]; then
  fftw_cppflags="-I$FFTW_PREFIX/include"
  fftw_ldflags="-L$FFTW_PREFIX/lib"
fi
cd "$work"
${FC:-gfortran} -x f95-cpp-input -cpp -O2 -fcheck=all -Wall -Wextra \
  "$root/kinds.f08" "$root/constants.f08" "$root/params.f08" \
  "$root/tests/units.f08" -o units
./units

${FC:-gfortran} -x f95-cpp-input -cpp -O2 -fopenmp -fcheck=all -Wall -Wextra \
  $fftw_cppflags \
  "$root/kinds.f08" "$root/constants.f08" "$root/params.f08" "$root/rng.f08" "$root/grid.f08" \
  "$root/potentials.f08" "$root/fftwrap.f08" "$root/langevin.f08" \
  "$root/propagator.f08" "$root/tests/dissipation.f08" \
  $fftw_ldflags -lfftw3_threads -lfftw3 -o regression
OMP_NUM_THREADS=1 ./regression

if [ -n "${QLE_EXE:-}" ] && [ -x "$QLE_EXE" ]; then
  mkdir cli-help
  (
    cd cli-help
    "$QLE_EXE" --help > help.txt
    grep -q 'Usage: qle_1d' help.txt
    if find . -type f \( -name '*.dat' -o -name '*.h5' \) | grep -q .; then
      echo 'help option unexpectedly ran a trajectory' >&2
      exit 1
    fi
  )
  echo 'PASS command-line help safety'

  mkdir ascii-smoke
  (
    cd ascii-smoke
    OMPI_MCA_btl=self OMP_NUM_THREADS=2 "$QLE_EXE" "$root/tests/INPUT.smoke.nml" > run.log
    test -s smoke.traj000001.rank0.s000001.dat
    test -s smoke.traj000001.rank0.s000002.dat
    test -s smoke.traj000001.rank0.s000001.pes.dat
    test -s smoke.traj000001.rank0.s000002.pes.dat
    test -s smoke.traj000001.rank0.obs.dat
    grep -q '# t(fs) =     0.010000' smoke.traj000001.rank0.s000001.dat
    grep -q '# x(Ang)' smoke.traj000001.rank0.s000001.pes.dat
    if grep -Eiq 'nan|inf' ./*.dat; then
      echo 'non-finite ASCII output' >&2
      exit 1
    fi
    awk '
      !/^#/ {
        if (n == 0) first = $1
        if (n == 1) dx = $1-first
        sum += $2+$3
        n++
      }
      END {
        norm = sum*dx
        if (n < 2 || norm < 0.9999999 || norm > 1.0000001) exit 1
      }
    ' smoke.traj000001.rank0.s000002.dat
  )
  echo 'PASS explicit input path and dynamic ASCII PES output'
fi

hdf5_python=
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import h5py, numpy' >/dev/null 2>&1; then
    hdf5_python=$candidate
    break
  fi
done

if [ "${HDF5_ENABLED:-0}" = 1 ] && [ -n "${QLE_EXE:-}" ] && \
   [ -n "$hdf5_python" ]; then
  mkdir hdf5-smoke
  cp "$root/tests/INPUT.smoke-hdf5.nml" hdf5-smoke/INPUT.nml
  (
    cd hdf5-smoke
    OMPI_MCA_btl=self OMP_NUM_THREADS=2 "$QLE_EXE" > run.log
    "$hdf5_python" "$root/tests/check_hdf5.py" smoke_h5.traj000001.rank0.h5
  )
else
  echo 'SKIP HDF5 output check (HDF5 build or h5py unavailable)'
fi
