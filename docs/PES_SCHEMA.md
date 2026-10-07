# External 3D diabatic PES schema

`potential_data.py` reads and writes `QuantumDynamics diabatic PES 3D v1` HDF5
files. Array order is `(z, y, x)`.

Required datasets:

| Dataset | Shape | Required `units` attribute |
|---|---:|---|
| `x_angstrom` | `(nx,)` | `angstrom` |
| `y_angstrom` | `(ny,)` | `angstrom` |
| `z_angstrom` | `(nz,)` | `angstrom` |
| `V11_cm-1` | `(nz, ny, nx)` | `cm^-1` |
| `V22_cm-1` | `(nz, ny, nx)` | `cm^-1` |
| `V12_real_cm-1` | `(nz, ny, nx)` | `cm^-1` |

Optional `V12_imag_cm-1` has the same shape and units. Its omission means zero
imaginary coupling. Coordinate axes must be finite and strictly increasing. All
potential values must be finite. `V21` is defined as the complex conjugate of
`V12`, enforcing a Hermitian local potential matrix.

Use `DiabaticPES3D.to_hdf5()` to produce a conforming file and
`DiabaticPES3D.from_hdf5()` to validate one. `multidimensional_transfer_3d.py
--pes FILE` interpolates onto the propagation grid and converts physical units to
atomic units internally. Requested propagation points must lie inside every source
axis; extrapolation is rejected.

The schema accepts surfaces from electronic-structure packages after the producer
maps adiabatic results to a consistent two-state diabatic representation. Surface
generation and diabatization remain external to QuantumDynamics.
