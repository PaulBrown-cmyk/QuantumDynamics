#!/usr/bin/env python3
"""
ShowDyn

Robust HDF5 trajectory reader/plotter for the QLE 1D solver.

Why you were getting:
  KeyError: 'Unable to synchronously open object (component not found)'

Your script was trying to read hard-coded dataset paths (e.g. "/x", "/t", "/psi", "/V"),
but the HDF5 files written by the Fortran code store data under different group/dataset
names (often per-step groups like "step_000001/..."). So those paths simply don't exist.

This script:
  - prints the HDF5 tree when a requested path is missing
  - auto-discovers likely datasets (x, t, psi, V) by scanning the file
  - supports per-step groups (step_000001, step_000002, ...)
  - handles complex data stored as complex, compound (re/im), or last-dim=2 real arrays
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Matplotlib / LaTeX text setup
# -----------------------------

def configure_matplotlib_text(
    use_tex: bool = True,
    font_family: str = "serif",
    font_size: float = 12.0,
    latex_preamble: Optional[str] = None,
    fallback_if_missing: bool = True,
) -> None:
    """Configure Matplotlib to render labels with LaTeX (usetex) if available.

    Notes:
      * If use_tex=True, a LaTeX installation is required (e.g. TeX Live/MacTeX).
      * If LaTeX is missing, and fallback_if_missing=True, we fall back to Matplotlib mathtext.
    """
    # Base typography controls (work in both usetex and mathtext modes)
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "legend.fontsize": max(1.0, font_size - 1),
        "xtick.labelsize": max(1.0, font_size - 1),
        "ytick.labelsize": max(1.0, font_size - 1),
        "axes.unicode_minus": True,
    })

    if not use_tex:
        mpl.rcParams.update({"text.usetex": False})
        return

    # LaTeX text rendering
    mpl.rcParams.update({"text.usetex": True})
    if latex_preamble:
        mpl.rcParams.update({"text.latex.preamble": latex_preamble})

    # Lightweight verification: try to draw a tiny figure once.
    # If LaTeX is not installed/configured, this will raise at draw time.
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_axis_off()
        ax.text(0.5, 0.5, r"$\mathrm{test}$", ha="center", va="center")
        fig.canvas.draw()
        plt.close(fig)
    except Exception as e:
        plt.close("all")
        if fallback_if_missing:
            print("[ShowDyn] WARNING: LaTeX (usetex) failed; falling back to Matplotlib mathtext.")
            print(f"[ShowDyn]   Reason: {type(e).__name__}: {e}")
            mpl.rcParams.update({"text.usetex": False})
        else:
            raise

# -----------------------------
# HDF5 helpers
# -----------------------------

def _walk_datasets(f: h5py.File) -> List[str]:
    out: List[str] = []
    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append("/" + name)
    f.visititems(_visitor)
    return sorted(out)

def print_h5_tree(fn: str, max_lines: int = 400) -> None:
    with h5py.File(fn, "r") as f:
        paths = _walk_datasets(f)
    print(f"\nHDF5 datasets in {fn} (showing up to {max_lines}):")
    for i, p in enumerate(paths[:max_lines], 1):
        print(f"  {i:4d}. {p}")
    if len(paths) > max_lines:
        print(f"  ... ({len(paths) - max_lines} more)")

def _get_ds(f: h5py.File, path: str) -> h5py.Dataset:
    # h5py accepts both "a/b" and "/a/b"
    p = path[1:] if path.startswith("/") else path
    return f[p]

def _try_paths(f: h5py.File, paths: List[str]) -> Optional[str]:
    for p in paths:
        try:
            _ = _get_ds(f, p)
            return p
        except KeyError:
            pass
    return None

def _find_by_predicate(f: h5py.File, pred: Callable[[str, h5py.Dataset], bool]) -> Optional[str]:
    hit: Optional[str] = None
    def _visitor(name, obj):
        nonlocal hit
        if hit is not None:
            return
        if isinstance(obj, h5py.Dataset):
            p = "/" + name
            try:
                if pred(p, obj):
                    hit = p
            except Exception:
                pass
    f.visititems(_visitor)
    return hit

def _as_complex(a: np.ndarray) -> np.ndarray:
    """
    Convert common HDF5 encodings to complex ndarray:
      - complex dtype already
      - compound dtype with fields like ('r','i') or ('re','im') or ('real','imag')
      - last dimension of size 2 holding (re, im)
    """
    if np.iscomplexobj(a):
        return a

    dt = getattr(a, "dtype", None)
    if dt is not None and dt.fields is not None:
        keys = list(dt.fields.keys())
        # common field names
        candidates = [
            ("r", "i"),
            ("re", "im"),
            ("real", "imag"),
            ("real", "imaginary"),
        ]
        for rk, ik in candidates:
            if rk in keys and ik in keys:
                return a[rk] + 1j * a[ik]
        # fallback: 2-field compound, assume (re, im)
        if len(keys) == 2:
            return a[keys[0]] + 1j * a[keys[1]]

    # last-dim=2 real
    if a.ndim >= 1 and a.shape[-1] == 2 and np.issubdtype(a.dtype, np.floating):
        return a[..., 0] + 1j * a[..., 1]

    raise TypeError(f"Don't know how to convert array with dtype={a.dtype} and shape={a.shape} to complex.")

def _read_array(f: h5py.File, path: str) -> np.ndarray:
    a = np.array(_get_ds(f, path)[...])
    return a

def _read_complex(f: h5py.File, path: str) -> np.ndarray:
    a = _read_array(f, path)
    return _as_complex(a)

def _read_scalar(f: h5py.File, path: str) -> float:
    a = _read_array(f, path)
    return float(np.array(a).reshape(-1)[0])


# -----------------------------
# Auto-discovery logic
# -----------------------------

_STEP_RE = re.compile(r"^/?(step[_\-]?\d+|step\d+)$", re.IGNORECASE)

def _list_step_groups(f: h5py.File) -> List[str]:
    """
    Returns step group paths like "/step_000001".
    """
    groups: List[str] = []
    def _visitor(name, obj):
        if isinstance(obj, h5py.Group):
            p = "/" + name
            base = p.split("/")[-1]
            if _STEP_RE.match(base):
                groups.append(p)
    f.visititems(_visitor)

    def _step_num(p: str) -> int:
        base = p.split("/")[-1]
        m = re.search(r"(\d+)$", base)
        return int(m.group(1)) if m else 0

    groups = sorted(set(groups), key=_step_num)
    return groups

def _guess_x_path(f: h5py.File) -> Optional[str]:
    # Prefer obvious names
    p = _try_paths(f, ["/x", "/grid/x", "/grid1d/x", "/mesh/x", "/coords/x"])
    if p:
        return p
    # Otherwise: first 1D float dataset named "x"
    return _find_by_predicate(
        f,
        lambda path, ds: ds.ndim == 1 and np.issubdtype(ds.dtype, np.floating)
                         and path.lower().split("/")[-1] in ("x", "grid_x", "coords_x")
    )

def _guess_t_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/t", "/time", "/time/t", "/traj/t"]
    if step_group:
        candidates = [f"{step_group}/t", f"{step_group}/time", f"{step_group}/time/t"] + candidates
    p = _try_paths(f, candidates)
    if p:
        return p
    return _find_by_predicate(
        f,
        lambda path, ds: ds.size == 1 and np.issubdtype(ds.dtype, np.floating)
                         and path.lower().split("/")[-1] in ("t", "time")
    )

def _guess_psi_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    """
    Best-effort guess for a *single* wavefunction dataset.

    If your file contains two-state output (psi1/psi2), prefer psi1 here
    (so legacy code still works), but see _guess_psi1_path/_guess_psi2_path.
    """
    candidates = ["/psi", "/psi1", "/state/psi", "/wf/psi", "/wavefunction/psi"]
    if step_group:
        candidates = [
            f"{step_group}/psi",
            f"{step_group}/psi1",
            f"{step_group}/state/psi",
            f"{step_group}/wf/psi",
            f"{step_group}/wavefunction/psi",
        ] + candidates
    p = _try_paths(f, candidates)
    if p:
        return p

    # otherwise: dataset whose leaf is psi/psi1 and is complex-like
    def pred(path, ds):
        leaf = path.lower().split("/")[-1]
        if leaf not in ("psi", "psi1", "wf", "wavefunction"):
            return False
        dt = ds.dtype
        if np.issubdtype(dt, np.complexfloating):
            return True
        if dt.fields is not None:
            return True
        # last dim 2 float -> (re, im)
        if ds.ndim >= 2 and ds.shape[-1] == 2 and np.issubdtype(dt, np.floating):
            return True
        return False

    return _find_by_predicate(f, pred)


def _guess_psi1_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/psi1", "/state1/psi", "/wf/psi1", "/wavefunction/psi1", "/psi_1"]
    if step_group:
        candidates = [f"{step_group}/psi1", f"{step_group}/state1/psi", f"{step_group}/psi_1"] + candidates
    p = _try_paths(f, candidates)
    if p:
        return p
    return _find_by_predicate(
        f,
        lambda path, ds: path.lower().split("/")[-1] in ("psi1", "psi_1")
                         and (np.issubdtype(ds.dtype, np.complexfloating)
                              or ds.dtype.fields is not None
                              or (ds.ndim >= 2 and ds.shape[-1] == 2 and np.issubdtype(ds.dtype, np.floating)))
    )


def _guess_psi2_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/psi2", "/state2/psi", "/wf/psi2", "/wavefunction/psi2", "/psi_2"]
    if step_group:
        candidates = [f"{step_group}/psi2", f"{step_group}/state2/psi", f"{step_group}/psi_2"] + candidates
    p = _try_paths(f, candidates)
    if p:
        return p
    return _find_by_predicate(
        f,
        lambda path, ds: path.lower().split("/")[-1] in ("psi2", "psi_2")
                         and (np.issubdtype(ds.dtype, np.complexfloating)
                              or ds.dtype.fields is not None
                              or (ds.ndim >= 2 and ds.shape[-1] == 2 and np.issubdtype(ds.dtype, np.floating)))
    )
def _guess_V_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    """Best-effort guess for a generic 1D potential array."""
    candidates = ["/V", "/pot/V", "/potential/V", "/Vtot", "/V1", "/V2", "/V11", "/V22"]
    if step_group:
        candidates = [
            f"{step_group}/V",
            f"{step_group}/pot/V",
            f"{step_group}/potential/V",
            f"{step_group}/Vtot",
            f"{step_group}/V1",
            f"{step_group}/V2",
            f"{step_group}/V11",
            f"{step_group}/V22",
        ] + candidates
    p = _try_paths(f, candidates)
    if p:
        return p
    return _find_by_predicate(
        f,
        lambda path, ds: np.issubdtype(ds.dtype, np.floating)
                         and ds.ndim == 1
                         and path.lower().split("/")[-1] in ("v", "v1", "v2", "v11", "v22", "vtot", "potential")
    )


def _guess_V11_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/V11", "/v11"]
    if step_group:
        candidates = [f"{step_group}/V11", f"{step_group}/v11"] + candidates
    return _try_paths(f, candidates)


def _guess_V22_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/V22", "/v22"]
    if step_group:
        candidates = [f"{step_group}/V22", f"{step_group}/v22"] + candidates
    return _try_paths(f, candidates)


def _guess_V12_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/V12", "/v12"]
    if step_group:
        candidates = [f"{step_group}/V12", f"{step_group}/v12"] + candidates
    return _try_paths(f, candidates)


def _guess_V_lower_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/V_lower", "/vlower"]
    if step_group:
        candidates = [f"{step_group}/V_lower", f"{step_group}/vlower"] + candidates
    return _try_paths(f, candidates)


def _guess_V_upper_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/V_upper", "/vupper"]
    if step_group:
        candidates = [f"{step_group}/V_upper", f"{step_group}/vupper"] + candidates
    return _try_paths(f, candidates)


def _guess_Vadiab_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/Vadiab", "/V_adiab"]
    if step_group:
        candidates = [f"{step_group}/Vadiab", f"{step_group}/V_adiab"] + candidates
    return _try_paths(f, candidates)


def _guess_Vdiab_path(f: h5py.File, step_group: Optional[str] = None) -> Optional[str]:
    candidates = ["/Vdiab", "/V_diab"]
    if step_group:
        candidates = [f"{step_group}/Vdiab", f"{step_group}/V_diab"] + candidates
    return _try_paths(f, candidates)

@dataclass
class TrajData:
    x: np.ndarray                 # (nx,)
    t: np.ndarray                 # (nt,)

    # Wavefunctions / densities
    psi: Optional[np.ndarray]     # (nt, nx) complex (single-state output)
    psi1: Optional[np.ndarray]    # (nt, nx) complex (two-state output)
    psi2: Optional[np.ndarray]    # (nt, nx) complex (two-state output)

    rho: np.ndarray               # (nt, nx) total density (always provided)
    rho1: Optional[np.ndarray]    # (nt, nx) state-1 density
    rho2: Optional[np.ndarray]    # (nt, nx) state-2 density

    # Potentials
    V: Optional[np.ndarray]       # generic potential (nt, nx) or (nx,)
    V11: Optional[np.ndarray]     # diabatic well 1 (nt, nx) or (nx,)
    V22: Optional[np.ndarray]     # diabatic well 2 (nt, nx) or (nx,)
    V12: Optional[np.ndarray]     # coupling (nt, nx) or (nx,)
    V_lower: Optional[np.ndarray] # adiabatic lower (nt, nx) or (nx,)
    V_upper: Optional[np.ndarray] # adiabatic upper (nt, nx) or (nx,)
    Vadiab: Optional[np.ndarray]  # (nt, nx, 2) or (nx, 2)
    Vdiab: Optional[np.ndarray]   # (nt, nx, 2) or (nx, 2)


def read_traj_auto(
    fn: str,
    x_path: Optional[str] = None,
    t_path: Optional[str] = None,
    psi_path: Optional[str] = None,
    V_path: Optional[str] = None,
    *,
    psi1_path: Optional[str] = None,
    psi2_path: Optional[str] = None,
    V11_path: Optional[str] = None,
    V22_path: Optional[str] = None,
    V12_path: Optional[str] = None,
    V_lower_path: Optional[str] = None,
    V_upper_path: Optional[str] = None,
    Vadiab_path: Optional[str] = None,
    Vdiab_path: Optional[str] = None,
) -> TrajData:
    """
    Read trajectory from a single .h5 file.

    Supports:
      (A) Whole-trajectory datasets at root
      (B) Per-step groups (step_000001/..., step_000002/..., ...)

    Also supports 2-state output (psi1/psi2) and per-step potentials (V11/V22/V12, V_lower/V_upper, Vadiab/Vdiab).
    """

    def _read_optional_1d(f: h5py.File, p: Optional[str]) -> Optional[np.ndarray]:
        if p is None:
            return None
        a = _read_array(f, p)
        return np.ravel(a)

    def _read_optional_any(f: h5py.File, p: Optional[str]) -> Optional[np.ndarray]:
        if p is None:
            return None
        return _read_array(f, p)

    with h5py.File(fn, "r") as f:
        # Detect per-step groups
        steps = _list_step_groups(f)

        # 1) x (prefer user path; otherwise auto-detect)
        x_path_eff = x_path or _guess_x_path(f)
        if x_path_eff is None:
            raise KeyError("Could not auto-detect x dataset. Run with --print-tree and choose --x-path.")
        try:
            x = _read_array(f, x_path_eff)
        except KeyError:
            print_h5_tree(fn)
            raise

        # 2) If steps exist, read per-step snapshots
        if steps:
            t_vals: List[float] = []

            psi_list: List[np.ndarray] = []
            psi1_list: List[np.ndarray] = []
            psi2_list: List[np.ndarray] = []

            V_list: List[np.ndarray] = []
            V11_list: List[np.ndarray] = []
            V22_list: List[np.ndarray] = []
            V12_list: List[np.ndarray] = []
            Vlower_list: List[np.ndarray] = []
            Vupper_list: List[np.ndarray] = []
            Vadiab_list: List[np.ndarray] = []
            Vdiab_list: List[np.ndarray] = []

            saw_two_state = False

            for sg in steps:
                tp = t_path or _guess_t_path(f, sg)

                # Prefer explicit two-state psi1/psi2 if present
                pp1 = psi1_path or _guess_psi1_path(f, sg)
                pp2 = psi2_path or _guess_psi2_path(f, sg)

                pp_single = psi_path or _guess_psi_path(f, sg)

                if tp is None:
                    print_h5_tree(fn)
                    raise KeyError(f"Missing time dataset under {sg}. Provide --t-path explicitly.")

                try:
                    t_vals.append(_read_scalar(f, tp))

                    if pp1 is not None and pp2 is not None:
                        saw_two_state = True
                        psi1_k = _read_complex(f, pp1)
                        psi2_k = _read_complex(f, pp2)
                        psi1_list.append(np.ravel(psi1_k))
                        psi2_list.append(np.ravel(psi2_k))
                    else:
                        if pp_single is None:
                            print_h5_tree(fn)
                            raise KeyError(f"Missing psi dataset under {sg}. Provide --psi-path (or --psi1-path/--psi2-path).")
                        psi_k = _read_complex(f, pp_single)
                        psi_list.append(np.ravel(psi_k))

                    # Potentials (all optional)
                    vp = V_path or _guess_V_path(f, sg)
                    if vp is not None:
                        V_list.append(np.ravel(_read_array(f, vp)))

                    v11p = V11_path or _guess_V11_path(f, sg)
                    if v11p is not None:
                        V11_list.append(np.ravel(_read_array(f, v11p)))

                    v22p = V22_path or _guess_V22_path(f, sg)
                    if v22p is not None:
                        V22_list.append(np.ravel(_read_array(f, v22p)))

                    v12p = V12_path or _guess_V12_path(f, sg)
                    if v12p is not None:
                        V12_list.append(np.ravel(_read_array(f, v12p)))

                    vlp = V_lower_path or _guess_V_lower_path(f, sg)
                    if vlp is not None:
                        Vlower_list.append(np.ravel(_read_array(f, vlp)))

                    vup = V_upper_path or _guess_V_upper_path(f, sg)
                    if vup is not None:
                        Vupper_list.append(np.ravel(_read_array(f, vup)))

                    vap = Vadiab_path or _guess_Vadiab_path(f, sg)
                    if vap is not None:
                        Vadiab_list.append(_read_array(f, vap))

                    vdp = Vdiab_path or _guess_Vdiab_path(f, sg)
                    if vdp is not None:
                        Vdiab_list.append(_read_array(f, vdp))

                except KeyError:
                    print_h5_tree(fn)
                    raise

            t = np.array(t_vals, dtype=float)

            # Wavefunctions / densities
            psi: Optional[np.ndarray] = None
            psi1: Optional[np.ndarray] = None
            psi2: Optional[np.ndarray] = None
            rho1: Optional[np.ndarray] = None
            rho2: Optional[np.ndarray] = None

            if saw_two_state and psi1_list and psi2_list:
                psi1 = np.vstack([p.reshape(1, -1) for p in psi1_list])
                psi2 = np.vstack([p.reshape(1, -1) for p in psi2_list])
                rho1 = np.abs(psi1) ** 2
                rho2 = np.abs(psi2) ** 2
                rho = rho1 + rho2
            else:
                psi = np.vstack([p.reshape(1, -1) for p in psi_list])
                rho = np.abs(psi) ** 2

            if rho.shape[1] != x.size:
                print(
                    f"WARNING: rho has nx={rho.shape[1]} but x has nx={x.size}. "
                    f"Check that you're reading the right x dataset ({x_path_eff})."
                )

            # Potentials
            V = np.vstack([v.reshape(1, -1) for v in V_list]) if V_list else None
            V11 = np.vstack([v.reshape(1, -1) for v in V11_list]) if V11_list else None
            V22 = np.vstack([v.reshape(1, -1) for v in V22_list]) if V22_list else None
            V12 = np.vstack([v.reshape(1, -1) for v in V12_list]) if V12_list else None
            V_lower = np.vstack([v.reshape(1, -1) for v in Vlower_list]) if Vlower_list else None
            V_upper = np.vstack([v.reshape(1, -1) for v in Vupper_list]) if Vupper_list else None
            Vadiab = np.stack(Vadiab_list, axis=0) if Vadiab_list else None
            Vdiab = np.stack(Vdiab_list, axis=0) if Vdiab_list else None

            return TrajData(
                x=x,
                t=t,
                psi=psi,
                psi1=psi1,
                psi2=psi2,
                rho=rho,
                rho1=rho1,
                rho2=rho2,
                V=V,
                V11=V11,
                V22=V22,
                V12=V12,
                V_lower=V_lower,
                V_upper=V_upper,
                Vadiab=Vadiab,
                Vdiab=Vdiab,
            )

        # 3) Otherwise: whole-trajectory datasets
        t_path_eff = t_path or _guess_t_path(f, None)

        # Wavefunction: prefer two-state if present
        pp1 = psi1_path or _guess_psi1_path(f, None)
        pp2 = psi2_path or _guess_psi2_path(f, None)
        pp_single = psi_path or _guess_psi_path(f, None)

        if t_path_eff is None:
            print_h5_tree(fn)
            raise KeyError("Could not auto-detect t dataset. Provide --t-path explicitly.")

        try:
            t = _read_array(f, t_path_eff)

            psi: Optional[np.ndarray] = None
            psi1: Optional[np.ndarray] = None
            psi2: Optional[np.ndarray] = None
            rho1: Optional[np.ndarray] = None
            rho2: Optional[np.ndarray] = None

            if pp1 is not None and pp2 is not None:
                psi1 = _read_complex(f, pp1)
                psi2 = _read_complex(f, pp2)
                rho1 = np.abs(psi1) ** 2
                rho2 = np.abs(psi2) ** 2
                rho = rho1 + rho2
            else:
                if pp_single is None:
                    print_h5_tree(fn)
                    raise KeyError("Could not auto-detect psi dataset. Provide --psi-path (or --psi1-path/--psi2-path).")
                psi = _read_complex(f, pp_single)
                rho = np.abs(psi) ** 2

            V = _read_optional_any(f, V_path or _guess_V_path(f, None))
            V11 = _read_optional_any(f, V11_path or _guess_V11_path(f, None))
            V22 = _read_optional_any(f, V22_path or _guess_V22_path(f, None))
            V12 = _read_optional_any(f, V12_path or _guess_V12_path(f, None))
            V_lower = _read_optional_any(f, V_lower_path or _guess_V_lower_path(f, None))
            V_upper = _read_optional_any(f, V_upper_path or _guess_V_upper_path(f, None))
            Vadiab = _read_optional_any(f, Vadiab_path or _guess_Vadiab_path(f, None))
            Vdiab = _read_optional_any(f, Vdiab_path or _guess_Vdiab_path(f, None))

            return TrajData(
                x=x,
                t=np.ravel(t).astype(float),
                psi=psi,
                psi1=psi1,
                psi2=psi2,
                rho=np.asarray(rho, dtype=float),
                rho1=None if rho1 is None else np.asarray(rho1, dtype=float),
                rho2=None if rho2 is None else np.asarray(rho2, dtype=float),
                V=V,
                V11=V11,
                V22=V22,
                V12=V12,
                V_lower=V_lower,
                V_upper=V_upper,
                Vadiab=Vadiab,
                Vdiab=Vdiab,
            )
        except KeyError:
            print_h5_tree(fn)
            raise
# -----------------------------
# Plotting / animation
# -----------------------------

def _x_rho_view(d: TrajData) -> Tuple[np.ndarray, np.ndarray]:
    """Return x/rho views with consistent nx."""
    nx = d.rho.shape[1]
    if d.x.size == nx:
        return d.x, d.rho
    n = min(d.x.size, nx)
    return d.x[:n], d.rho[:, :n]


def _pot_at_step(p: Optional[np.ndarray], it: int) -> Optional[np.ndarray]:
    if p is None:
        return None
    if p.ndim == 1:
        return p
    if p.ndim == 2:
        return p[it]
    return None


def plot_snapshot(d: TrajData, it: int) -> None:
    it = int(np.clip(it, 0, d.t.size - 1))
    x, rho = _x_rho_view(d)

    fig, ax = plt.subplots()
    ax.plot(x, rho[it], label=r"$\rho(x,t)$")

    # If we have 2-state densities, show them too
    if d.rho1 is not None and d.rho2 is not None:
        nx = min(x.size, d.rho1.shape[1], d.rho2.shape[1])
        ax.plot(x[:nx], d.rho1[it, :nx], label=r"$\rho_1$")
        ax.plot(x[:nx], d.rho2[it, :nx], label=r"$\rho_2$")

    ax.set_xlabel(r"$x$", size=14)
    ax.set_ylabel(r"$|\psi(x,t)|^2$")
    ax.set_ylim(0.0, float(np.max(rho)) * 1.05)

    # Potentials on a twin axis
    pot_lines: List[Tuple[str, np.ndarray]] = []
    for name, arr in [
        (r"$V_{11}$", d.V11),
        (r"$V_{22}$", d.V22),
        (r"$V_{lower}$", d.V_lower),
        (r"$V_{upper}$", d.V_upper),
        (r"$V$", d.V),
    ]:
        Vt = _pot_at_step(arr, it)
        if Vt is not None:
            pot_lines.append((name, np.ravel(Vt)))

    if pot_lines:
        ax2 = ax.twinx()
        for name, Vt in pot_lines:
            nn = min(x.size, Vt.size)
            ax2.plot(x[:nn], Vt[:nn], label=name)
        ax2.set_ylabel("Potential energy")

        # Scale potential axis to visible range
        Vall = np.concatenate([v[: min(x.size, v.size)] for _, v in pot_lines])
        ax2.set_ylim(float(np.min(Vall)), float(np.max(Vall)))

        # One combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax.legend(loc="best")

    ax.set_title(f"t = {d.t[it]:.6g} (snapshot {it})")
    plt.show()


def animate(d: TrajData, every: int = 1) -> None:
    from matplotlib.animation import FuncAnimation

    x, rho = _x_rho_view(d)

    fig, ax = plt.subplots()
    (ln_rho,) = ax.plot([], [], label=r"$\rho$")
    ln_rho1 = ln_rho2 = None
    if d.rho1 is not None and d.rho2 is not None:
        (ln_rho1,) = ax.plot([], [], label=r"$\rho_1$")
        (ln_rho2,) = ax.plot([], [], label=r"$\rho_2$")

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(x,t)$")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0.0, float(np.max(rho)) * 1.05)

    # Potentials
    ax2 = None
    pot_names: List[str] = []
    pot_arrays: List[np.ndarray] = []
    pot_lines = []

    for name, arr in [
        (r"$V_{11}$", d.V11),
        (r"$V_{22}$", d.V22),
        (r"$V_{lower}$", d.V_lower),
        (r"$V_{upper}$", d.V_upper),
        (r"$V$", d.V),
    ]:
        if arr is not None:
            ax2 = ax2 or ax.twinx()
            pot_names.append(name)
            pot_arrays.append(arr)
            (ln,) = ax2.plot([], [], label=name)
            pot_lines.append(ln)

    if ax2 is not None and pot_arrays:
        ax2.set_ylabel(r"$P(x,t)$")

        # Compute y-lims across all potentials
        pot_vals = []
        for arr in pot_arrays:
            if arr.ndim == 2:
                pot_vals.append(arr[:, : min(arr.shape[1], x.size)].ravel())
            elif arr.ndim == 1:
                pot_vals.append(arr[: min(arr.size, x.size)].ravel())
        if pot_vals:
            Vall = np.concatenate(pot_vals)
            ax2.set_ylim(float(np.min(Vall)), float(np.max(Vall)))

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax.legend(loc="best")

    title = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    idxs = np.arange(0, d.t.size, max(1, every), dtype=int)

    def init():
        ln_rho.set_data([], [])
        if ln_rho1 is not None:
            ln_rho1.set_data([], [])
        if ln_rho2 is not None:
            ln_rho2.set_data([], [])
        for ln in pot_lines:
            ln.set_data([], [])
        title.set_text("")
        artists = [ln_rho, title]
        if ln_rho1 is not None:
            artists.append(ln_rho1)
        if ln_rho2 is not None:
            artists.append(ln_rho2)
        artists.extend(pot_lines)
        return tuple(artists)

    def update(k):
        it = int(idxs[k])
        ln_rho.set_data(x, rho[it])

        if ln_rho1 is not None and d.rho1 is not None:
            nx = min(x.size, d.rho1.shape[1])
            ln_rho1.set_data(x[:nx], d.rho1[it, :nx])
        if ln_rho2 is not None and d.rho2 is not None:
            nx = min(x.size, d.rho2.shape[1])
            ln_rho2.set_data(x[:nx], d.rho2[it, :nx])

        for j, arr in enumerate(pot_arrays):
            Vt = _pot_at_step(arr, it)
            if Vt is not None:
                Vt = np.ravel(Vt)
                nn = min(x.size, Vt.size)
                pot_lines[j].set_data(x[:nn], Vt[:nn])

        title.set_text(f"t = {d.t[it]:.6g} (step {it})")

        artists = [ln_rho, title]
        if ln_rho1 is not None:
            artists.append(ln_rho1)
        if ln_rho2 is not None:
            artists.append(ln_rho2)
        artists.extend(pot_lines)
        return tuple(artists)

    ani = FuncAnimation(fig, update, frames=len(idxs), init_func=init, interval=40, blit=True)
    plt.show()


def compute_reactant_product_probabilities(d: TrajData, x_split: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Integrate total density rho(x,t) left/right of a dividing surface x_split.
      P_R(t) = ∫_{x<=x_split} rho dx
      P_P(t) = ∫_{x> x_split} rho dx
    """
    x, rho = _x_rho_view(d)

    if x_split is None:
        x_split = 0.0 if (np.min(x) <= 0.0 <= np.max(x)) else float(x[x.size // 2])

    left = x <= x_split
    right = ~left
    if not np.any(left) or not np.any(right):
        raise ValueError(f"Bad x_split={x_split}: one side is empty. Choose a split inside the x-grid range.")

    pR = np.trapz(rho[:, left], x[left], axis=1)
    pP = np.trapz(rho[:, right], x[right], axis=1)
    return pR, pP, float(x_split)



def compute_state_average_positions(d: TrajData) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute ⟨x⟩(t) for reactant/product (state-1/state-2) densities when available."""
    x = d.x.reshape(1, -1)

    xavg1: Optional[np.ndarray] = None
    xavg2: Optional[np.ndarray] = None

    if d.rho1 is not None:
        pop1 = np.trapz(d.rho1, d.x, axis=1)
        num1 = np.trapz(d.rho1 * x, d.x, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xavg1 = np.where(pop1 > 0.0, num1 / pop1, np.nan)

    if d.rho2 is not None:
        pop2 = np.trapz(d.rho2, d.x, axis=1)
        num2 = np.trapz(d.rho2 * x, d.x, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xavg2 = np.where(pop2 > 0.0, num2 / pop2, np.nan)

    return xavg1, xavg2

def plot_reactant_product_probabilities(d: TrajData, x_split: Optional[float] = None, save: Optional[str] = None) -> None:
    pR, pP, x0 = compute_reactant_product_probabilities(d, x_split=x_split)
    xavg1, xavg2 = compute_state_average_positions(d)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Probabilities (left/right of dividing surface)
    ax1.plot(d.t, pR, label=f"Reactant (x $\le$ {x0:g})")
    ax1.plot(d.t, pP, label=f"Product (x $\gg$ {x0:g})")
    ax1.plot(d.t, pR + pP, linestyle="--", label="Total")
    ax1.set_ylabel(r"$P(x,t)$", size=14)
    ax1.set_title(r"Reactant-Product probabilities and $\langle\hat{x}\rangle$ vs time")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # Average positions for diabatic components (if available)
    plotted_any = False
    if xavg1 is not None:
        ax2.plot(d.t, xavg1, label=r"$\langle\hat{x}\rangle_1$ (reactant)")
        plotted_any = True
    if xavg2 is not None:
        ax2.plot(d.t, xavg2, label=r"$\langle\hat{x}\rangle_2$ (product)")
        plotted_any = True

    if not plotted_any:
        # Fall back to total density center-of-mass
        x, rho = _x_rho_view(d)
        pop = np.trapz(rho, x, axis=1)
        num = np.trapz(rho * x.reshape(1, -1), x, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xavg = np.where(pop > 0.0, num / pop, np.nan)
        ax2.plot(d.t, xavg, label=r"$\langle\hat{x}\rangle_{total}$")

    ax2.set_xlabel("t", size=14)
    ax2.set_ylabel(r"$\langle\hat{x}\rangle$", size=14)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=400, bbox_inches="tight")
    plt.show()

def main():
    ap = argparse.ArgumentParser()

    # Text rendering / typography
    texg = ap.add_mutually_exclusive_group()
    texg.add_argument("--usetex", dest="usetex", action="store_true",
                      help="Render all text with LaTeX (text.usetex). Requires a LaTeX install.")
    texg.add_argument("--no-usetex", dest="usetex", action="store_false",
                      help="Disable LaTeX and use Matplotlib mathtext.")
    ap.set_defaults(usetex=True)
    ap.add_argument("--font-family", default="serif", choices=["serif", "sans-serif", "monospace"],
                    help="Matplotlib font.family")
    ap.add_argument("--font-size", type=float, default=12.0,
                    help="Base font size for plot text")
    ap.add_argument("--latex-preamble", default=None,
                    help="Extra LaTeX preamble (e.g. \\usepackage{amsmath,amssymb})")
    ap.add_argument("h5", help="Trajectory .h5 file")

    # Core datasets
    ap.add_argument("--x-path", default=None, help="Dataset path for x grid (e.g. /x or /step_000001/x)")
    ap.add_argument("--t-path", default=None, help="Dataset path for time (e.g. /t or /step_000001/t)")
    ap.add_argument("--psi-path", default=None, help="Dataset path for single-state wavefunction (e.g. /psi)")
    ap.add_argument("--psi1-path", default=None, help="Dataset path for state-1 wavefunction (e.g. /psi1 or /step_000001/psi1)")
    ap.add_argument("--psi2-path", default=None, help="Dataset path for state-2 wavefunction (e.g. /psi2 or /step_000001/psi2)")

    # Potentials (optional)
    ap.add_argument("--V-path", default=None, help="Generic potential dataset path (e.g. /V)")
    ap.add_argument("--V11-path", default=None, help="Diabatic well-1 potential path (e.g. /step_000001/V11)")
    ap.add_argument("--V22-path", default=None, help="Diabatic well-2 potential path (e.g. /step_000001/V22)")
    ap.add_argument("--V12-path", default=None, help="Coupling potential path (e.g. /step_000001/V12)")
    ap.add_argument("--V-lower-path", dest="V_lower_path", default=None, help="Lower adiabatic surface path (e.g. /step_000001/V_lower)")
    ap.add_argument("--V-upper-path", dest="V_upper_path", default=None, help="Upper adiabatic surface path (e.g. /step_000001/V_upper)")
    ap.add_argument("--Vadiab-path", default=None, help="Adiabatic surfaces array path (e.g. /step_000001/Vadiab)")
    ap.add_argument("--Vdiab-path", default=None, help="Diabatic surfaces array path (e.g. /step_000001/Vdiab)")

    # Utilities / plotting modes
    ap.add_argument("--print-tree", action="store_true", help="Print dataset paths in the file and exit")
    ap.add_argument("--snapshot", type=int, default=None, help="Plot a single snapshot index")
    ap.add_argument("--animate", action="store_true", help="Animate rho (and potentials if present) over time")
    ap.add_argument("--every", type=int, default=1, help="Use every Nth frame in animation")

    # Populations vs time
    ap.add_argument("--populations", action="store_true", help="Plot reactant/product probabilities vs time (static)")
    ap.add_argument("--x-split", type=float, default=None, help="Dividing surface for reactant/product (default: 0 if in grid else midpoint)")
    ap.add_argument("--save-pop", default=None, help="Save the populations plot to a file (e.g. pop.png, pop.pdf)")

    args = ap.parse_args()

    if args.print_tree:
        print_h5_tree(args.h5, max_lines=2000)
        return

    # Configure Matplotlib text rendering (LaTeX if available)
    configure_matplotlib_text(
        use_tex=args.usetex,
        font_family=args.font_family,
        font_size=args.font_size,
        latex_preamble=args.latex_preamble,
        fallback_if_missing=True,
    )

    d = read_traj_auto(
        args.h5,
        x_path=args.x_path,
        t_path=args.t_path,
        psi_path=args.psi_path,
        V_path=args.V_path,
        psi1_path=args.psi1_path,
        psi2_path=args.psi2_path,
        V11_path=args.V11_path,
        V22_path=args.V22_path,
        V12_path=args.V12_path,
        V_lower_path=args.V_lower_path,
        V_upper_path=args.V_upper_path,
        Vadiab_path=args.Vadiab_path,
        Vdiab_path=args.Vdiab_path,
    )

    if args.populations:
        plot_reactant_product_probabilities(d, x_split=args.x_split, save=args.save_pop)
        return

    if args.snapshot is not None:
        plot_snapshot(d, args.snapshot)
    elif args.animate:
        animate(d, every=max(1, args.every))
    else:
        # default: plot first and last snapshots
        plot_snapshot(d, 0)
        plot_snapshot(d, d.t.size - 1)


if __name__ == "__main__":
    main()

