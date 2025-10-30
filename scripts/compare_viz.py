#!/usr/bin/env python3
# scripts/compare_viz.py
"""
Compare two visualization files (VTK family) field-by-field with tolerances.

Usage:
  # direct files
  python3 scripts/compare_viz.py out/ParaView/step_00000.vtu out/ParaView/step_00100.vtu \
      --fields rho,u,v,p --rtol 1e-10 --atol 1e-12

  # PVD convenience: pass a single .pvd and it will compare first vs last dataset
  python3 scripts/compare_viz.py out/ParaView/case.pvd --fields rho,u,v,p

Exit codes:
  0 -> pass, 1 -> fail, 2 -> input error
"""
from __future__ import annotations
import argparse
import sys
import os
import json
import hashlib
from typing import Dict, List, Tuple
import numpy as np

try:
    import xml.etree.ElementTree as ET
except Exception:
    ET = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two VTK viz files with tolerances.")
    p.add_argument("inputs", nargs="+",
                   help="Two viz files (e.g., first.vtu last.vtu) OR a single .pvd (first vs last inside).")
    p.add_argument("--fields", default="", help="Comma-separated list of fields to compare (empty = all common).")
    p.add_argument("--exclude-fields", default="", help="Comma-separated list of fields to ignore.")
    p.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance.")
    p.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance.")
    p.add_argument("--compare-cells", action="store_true",
                   help="Also compare cell_data (default compares only point_data).")
    p.add_argument("--json-out", default="", help="Optional path to write machine-readable diff JSON.")
    p.add_argument("--quiet", action="store_true", help="Less verbose output.")
    return p.parse_args()


def _hash(a: np.ndarray) -> str:
    # Stable content hash for quick identity checks (after canonicalization)
    h = hashlib.blake2b(digest_size=12)
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def _canonicalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted indices for points by (x,y,z) and the sorted array."""
    if points.ndim != 2:
        raise ValueError("points must be (N,dim)")
    key = np.lexsort(points[:, :3].T[::-1])  # sort by x, then y, then z
    return key, points[key]


def _cell_centers(points: np.ndarray, cells: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    """Compute cell centers as mean of vertices; return array aligned to concatenated cell order."""
    centers = []
    for _cell_type, conn in cells:
        if conn.size == 0:
            continue
        pts = points[conn]
        # mean over node index axis
        c = pts.mean(axis=1)
        centers.append(c)
    if not centers:
        return np.zeros((0, points.shape[1]))
    return np.vstack(centers)


def _canonicalize_cells(mesh, prefer_backend_centers: bool = False):
    """
    Return (sort_index, sorted_centers).
    For PyVista meshes, prefer mesh.cell_centers() to guarantee same order as cell_data.
    For meshio meshes, compute centers from connectivity blocks in given order.
    """
    # PyVista adapter?
    if hasattr(mesh, "_pv"):  # see adapter below
        pv = mesh._pv
        if prefer_backend_centers:
            centers = np.asarray(pv.cell_centers().points)
            if centers.size == 0:
                return np.array([], dtype=int), centers
            key = np.lexsort(centers[:, :3].T[::-1])
            return key, centers[key]

        # fallback: compute with connectivity (rarely needed if centers() is available)
        # (not shown; the centers() path is preferred)

    # meshio-like path
    centers = _cell_centers(np.asarray(mesh.points), mesh.cells)
    if centers.shape[0] == 0:
        return np.array([], dtype=int), centers
    key = np.lexsort(centers[:, :3].T[::-1])
    return key, centers[key]


def _stack_point_data(mesh) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    npts = int(np.asarray(mesh.points).shape[0])

    for name, arr in mesh.point_data.items():
        a = np.asarray(arr)

        # Skip non-numeric or obvious VTK internals
        if a.dtype.kind in {"U", "S", "O"} or name.startswith("vtk"):
            continue

        if a.ndim == 1:
            # Expect (Npts,) -> make (Npts, 1)
            if a.size == npts:
                a = a.reshape(npts, 1)
            else:
                # Not point-sized data; ignore
                continue
        elif a.ndim == 2:
            # Expect (Npts, comp). If (comp, Npts), transpose.
            if a.shape[0] == npts:
                pass
            elif a.shape[1] == npts:
                a = a.T
            else:
                # Mismatch; ignore
                continue
        else:
            # Flatten trailing dims if total size is a multiple of Npts
            if a.size % npts != 0:
                continue
            a = a.reshape(npts, -1)

        out[name] = np.ascontiguousarray(a)

    return out


def _stack_cell_data(mesh) -> Dict[str, np.ndarray]:
    # PyVista adapter: already (Nc, comp) arrays wrapped in a one-element list
    if hasattr(mesh, "_pv"):
        return {n: v[0] if isinstance(v, list) and len(v) == 1 else np.asarray(v)
                for n, v in mesh.cell_data.items()}

    # meshio path: stack per block
    if not getattr(mesh, "cell_data", None):
        return {}
    out: Dict[str, List[np.ndarray]] = {}
    for name, blocks in mesh.cell_data.items():
        stk: List[np.ndarray] = []
        for blk in blocks:
            b = np.asarray(blk)
            if b.ndim == 1:
                b = b.reshape(-1, 1)
            elif b.ndim == 2 and b.shape[0] < b.shape[1]:
                b = b.T
            else:
                b = b.reshape(b.shape[0], -1)
            stk.append(np.ascontiguousarray(b))
        if stk:
            out[name] = np.vstack(stk)
    return out


def _select_common_fields(A: Dict[str, np.ndarray], B: Dict[str, np.ndarray],
                          include: List[str] | None, exclude: List[str]) -> List[str]:
    common = sorted(set(A.keys()) & set(B.keys()))
    if include:
        common = [f for f in common if f in include]
    if exclude:
        ex = set(exclude)
        common = [f for f in common if f not in ex]
    return common


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> Dict[str, float]:
    # a,b: (N, comp) aligned
    if a.shape != b.shape:
        return {"shape_mismatch": 1.0}
    diff = b - a
    abs_err = np.abs(diff)
    # norms per-component collapsed
    l2 = float(np.linalg.norm(diff) / max(1.0, np.linalg.norm(a)))
    linf = float(abs_err.max(initial=0.0))
    mean_abs = float(abs_err.mean())
    # pass/fail: use elementwise tolerance check
    ok = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return {"l2_rel": l2, "linf_abs": linf, "mean_abs": mean_abs, "ok": bool(ok)}


def _load_mesh(path: str) -> 'MeshAdapter':
    import pyvista as pv
    # pyvista wraps vtkXML{P,U}UnstructuredGridReader internally
    mesh = pv.read(path)
    # Create a fake meshio-like object interface so the rest of the script works
    class MeshAdapter:
        def __init__(self, m):
            self._pv = m  # keep pyvista object for backend features
            self.points = np.array(m.points)
            # pyvista stores both point_data and cell_data
            self.point_data = {k: np.array(v) for k, v in m.point_data.items()}
            self.cell_data = {k: [np.array(v)] for k, v in m.cell_data.items()}
            self.cells = []
    return MeshAdapter(mesh)


def _from_pvd(pvd_path: str) -> Tuple[str, str]:
    """Return (first_dataset_path, last_dataset_path) from a .pvd file."""
    if ET is None:
        raise RuntimeError("xml parser unavailable; cannot read .pvd")
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    sets = root.findall(".//DataSet")
    if not sets:
        raise RuntimeError("No DataSet entries in PVD.")
    # preserve file order; some PVDs include timestep attribute but order is fine
    first = sets[0].attrib.get("file")
    last = sets[-1].attrib.get("file")
    base = os.path.dirname(pvd_path)
    return os.path.join(base, first), os.path.join(base, last)


def compare_files(f0: str, f1: str, fields: List[str], exclude: List[str],
                  rtol: float, atol: float, compare_cells: bool, quiet: bool) -> Tuple[bool, Dict]:
    m0 = _load_mesh(f0)
    m1 = _load_mesh(f1)

    # Canonicalize point ordering
    idx0, pts0_sorted = _canonicalize_points(m0.points)
    idx1, pts1_sorted = _canonicalize_points(m1.points)

    # Quick geometry sanity (same bbox within tiny tol)
    bbox0 = np.array([pts0_sorted.min(axis=0), pts0_sorted.max(axis=0)])
    bbox1 = np.array([pts1_sorted.min(axis=0), pts1_sorted.max(axis=0)])
    geom_ok = np.allclose(bbox0, bbox1, rtol=1e-12, atol=1e-12) and (pts0_sorted.shape == pts1_sorted.shape)

    # Point data (aligned by sorted point index)
    pd0_raw = _stack_point_data(m0)
    pd1_raw = _stack_point_data(m1)
    pd0 = {k: v[idx0] for k, v in pd0_raw.items()}
    pd1 = {k: v[idx1] for k, v in pd1_raw.items()}

    # Cells: concatenate & canonicalize on centers
    cd0 = {}
    cd1 = {}
    if compare_cells:
        cd0_raw = _stack_cell_data(m0)
        cd1_raw = _stack_cell_data(m1)
        ckey0, _ = _canonicalize_cells(m0, prefer_backend_centers=True)
        ckey1, _ = _canonicalize_cells(m1, prefer_backend_centers=True)
        cd0 = {k: (v[ckey0] if v.size else v) for k, v in cd0_raw.items()}
        cd1 = {k: (v[ckey1] if v.size else v) for k, v in cd1_raw.items()}
        # ckey0, _ = _canonicalize_cells(m0.points, m0.cells)
        for side, key, cd in (("A", ckey0, cd0_raw), ("B", ckey1, cd1_raw)):
            for nm, arr in cd.items():
                if arr.size and key.size and arr.shape[0] != key.size:
                    raise RuntimeError(
                        f"Cell count mismatch on {side} for '{nm}': "
                        f"cell_data rows={arr.shape[0]} vs centers={key.size}. "
                        "This indicates a backend ordering mismatch."
                    )

    include = [f.strip() for f in fields if f.strip()]
    exclude = [f.strip() for f in exclude if f.strip()]

    results: Dict = {"files": [f0, f1], "geom_ok": bool(geom_ok), "point_data": {}, "cell_data": {}}

    # Which fields
    pfields = _select_common_fields(pd0, pd1, include or None, exclude)
    for name in pfields:
        r = _compare_arrays(pd0[name], pd1[name], rtol, atol)
        r["ncomp"] = int(pd0[name].shape[1])
        r["n"] = int(pd0[name].shape[0])
        r["hash0"] = _hash(pd0[name])
        r["hash1"] = _hash(pd1[name])
        results["point_data"][name] = r

    cfields = []
    if compare_cells:
        cfields = _select_common_fields(cd0, cd1, include or None, exclude)
        for name in cfields:
            r = _compare_arrays(cd0[name], cd1[name], rtol, atol)
            r["ncomp"] = int(cd0[name].shape[1])
            r["n"] = int(cd0[name].shape[0])
            r["hash0"] = _hash(cd0[name])
            r["hash1"] = _hash(cd1[name])
            results["cell_data"][name] = r

    # overall pass
    ok_fields = all(v.get("ok", False) for v in results["point_data"].values())
    if compare_cells:
        ok_fields = ok_fields and all(v.get("ok", False) for v in results["cell_data"].values())
    results["ok"] = bool(geom_ok and ok_fields)

    if not quiet:
        print(f"[compare_viz] Comparing:\n  A: {f0}\n  B: {f1}")
        print(f"  Geometry bbox match: {geom_ok} (A:{bbox0.tolist()} B:{bbox1.tolist()})")
        def _pp(section, data):
            if not data:
                print(f"  {section}: (no common fields)")
                return
            print(f"  {section}:")
            for k, r in data.items():
                status = "OK" if r['ok'] else "FAIL"
                print(f"    {k:20s} {status:4s}  n={r['n']:7d} comp={r['ncomp']}"
                      f"  L2rel={r['l2_rel']:.3e}  Linf={r['linf_abs']:.3e}  mean|e|={r['mean_abs']:.3e}"
                      f"  hashA={r['hash0'][:6]} hashB={r['hash1'][:6]}")
        _pp("point_data", results["point_data"])
        if compare_cells:
            _pp("cell_data", results["cell_data"])
        print(f"Overall: {'PASS' if results['ok'] else 'FAIL'}  (rtol={rtol}, atol={atol})")

    return results["ok"], results


def main():
    args = parse_args()
    inputs = args.inputs

    # PVD convenience: one arg -> first vs last inside .pvd
    if len(inputs) == 1 and inputs[0].lower().endswith(".pvd"):
        pvd = inputs[0]
        try:
            f0, f1 = _from_pvd(pvd)
        except Exception as e:
            print(f"ERROR reading PVD: {e}", file=sys.stderr)
            return 2
    elif len(inputs) == 2:
        f0, f1 = inputs
    else:
        print("ERROR: Provide either two files or a single .pvd.", file=sys.stderr)
        return 2

    if not (os.path.exists(f0) and os.path.exists(f1)):
        print(f"ERROR: files not found: {f0}, {f1}", file=sys.stderr)
        return 2

    fields = [s for s in args.fields.split(",") if s]
    excludes = [s for s in args.exclude_fields.split(",") if s]
    # Auto-exclude common VTK internals that cause shape surprises
    excludes += ["vtkOriginalPointIds", "vtkGhostType", "vtkProcessId"]

    ok, results = compare_files(
        f0, f1, fields=fields, exclude=excludes,
        rtol=args.rtol, atol=args.atol,
        compare_cells=args.compare_cells, quiet=args.quiet
    )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
