"""
Translate CZI frames to their stage positions and write a single OME-Zarr.

Usage
-----
    python translate_to_zarr.py \
        --image_dir   Image \
        --positions_csv Image/Positions_data.csv \
        --output_path  output/P0001_translated.zarr \
        --pattern      P0001 \
        --n_pyramid_layers 4

Each CZI file (one timepoint) is placed onto a common physical canvas according
to its stage position in the CSV, producing a moving field-of-view time series
on one big canvas.

Performance
-----------
Rather than padding every frame out to the full canvas and concatenating a
dense (mostly-zero) array, each frame is written *sparsely* straight into its
region of a zero-filled OME-Zarr canvas. Empty canvas chunks are never stored
(zarr fill_value), so the work scales with the real data (frame_area x
n_timepoints) instead of the whole canvas — typically 10-100x faster and a much
smaller output. Frames are read with the native pylibCZIrw reader (no JVM), the
region writes run as a single parallel dask computation, and the multiscale
pyramid is built by eubi-bridge's tensorstore downscaler (much faster than
dask/coarsen). Paths are normalised to forward slashes.
"""

import asyncio
import csv
import os
from dataclasses import dataclass
from typing import Dict, List

import dask.array as da
import numpy as np

from eubi_bridge.core.data_manager import ArrayManager
from eubi_bridge.core.czi_reader import read_czi
from eubi_bridge.core.writers import (
    _get_or_create_multimeta,
    _zarr_group,
    downscale_with_tensorstore_async,
    wrap_output_path,
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

@dataclass
class FrameMeta:
    filename: str
    x_um: float
    y_um: float


def read_positions(csv_path: str) -> List[FrameMeta]:
    """Read stage positions from a Zeiss-exported CSV (UTF-8 BOM, quoted values)."""
    result = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("FileName::FileName", "").strip()
            if fn:
                result.append(FrameMeta(
                    filename=fn,
                    x_um=float(row["Stage_Position_X::Stage_Position_X"]),
                    y_um=float(row["Stage_Position_Y::Stage_Position_Y"]),
                ))
    return result


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------

async def _load_reference_manager(path: str) -> ArrayManager:
    """Open one CZI via bfio once to read axes / scale / units / shape."""
    outer = ArrayManager(path, metadata_reader="bfio")
    await outer.load_scenes(scene_indices=0)
    inner = next(iter(outer.loaded_scenes.values()))
    inner.fill_default_meta()   # fills missing scale / axes with defaults
    return inner


def _read_frame_array(path: str, t_ax: int) -> da.Array:
    """Read a single CZI frame natively (no JVM) as a lazy T C Z Y X dask array,
    forced to a single timepoint on the T axis."""
    arr = read_czi(path).get_image_dask_data()          # T C Z Y X
    return arr[_axis_slices(arr.ndim, {t_ax: slice(0, 1)})]


def _axis_slices(ndim: int, named: Dict[int, slice]) -> tuple:
    """Build a slice tuple of length ndim; ``named`` maps axis-index -> slice."""
    sl = [slice(None)] * ndim
    for idx, s in named.items():
        sl[idx] = s
    return tuple(sl)


def _fwd(path) -> str:
    """Normalise a filesystem path to forward slashes (we prefer '/')."""
    return str(path).replace(os.sep, "/").replace("\\", "/")


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

async def translate_and_write(
    image_dir: str,
    positions_csv: str,
    output_path: str,
    pattern: str = "",
    n_pyramid_layers: int = 4,
    overwrite: bool = False,
    max_project: bool = False,
) -> None:
    # ── 1. Collect and (optionally) filter frame list ────────────────────
    all_frames = read_positions(positions_csv)
    frames: List[FrameMeta] = (
        [f for f in all_frames if pattern in f.filename]
        if pattern else all_frames
    )
    if not frames:
        raise ValueError(f"No files match pattern '{pattern}' in {positions_csv}")
    frames.sort(key=lambda f: f.filename)
    n_t = len(frames)
    n_layers = max(1, int(n_pyramid_layers))
    print(f"Processing {n_t} frames (pattern='{pattern}'), {n_layers} pyramid level(s)")

    # ── 2. Reference frame → axes / scale / units / spatial shape (one JVM read) ─
    ref_path = os.path.join(image_dir, frames[0].filename)
    print(f"Reading reference metadata from {frames[0].filename} …")
    ref = await _load_reference_manager(ref_path)
    axes: str = ref.axes                       # e.g. 'tczyx'
    scaledict = ref.scaledict                  # e.g. {'t':1, 'z':.5, 'y':.25, 'x':.25}
    units: list = list(ref.units)              # e.g. ['', '', 'µm', 'µm', 'µm']

    t_ax, y_ax, x_ax = axes.index("t"), axes.index("y"), axes.index("x")
    c_ax = axes.index("c") if "c" in axes else None
    z_ax_orig = axes.index("z") if "z" in axes else None

    # dtype + spatial shape come from the SAME native reader used for the frames,
    # so the output array's dtype matches the pixel data we actually write. The
    # bfio metadata reader can report a different pixel type (e.g. uint8 for
    # float data), which would silently truncate the values on write.
    ref_frame = _read_frame_array(_fwd(ref_path), t_ax)
    if max_project and z_ax_orig is not None:
        ref_frame = da.max(ref_frame, axis=z_ax_orig, keepdims=False)
        axes = axes.replace("z", "")
        units.pop(z_ax_orig)
        t_ax = axes.index("t")
        y_ax = axes.index("y")
        x_ax = axes.index("x")
        c_ax = axes.index("c") if "c" in axes else None
        print(f"  Max-projecting along Z → axes now '{axes}'")
    dtype = np.dtype(ref_frame.dtype)
    frame_y, frame_x = int(ref_frame.shape[y_ax]), int(ref_frame.shape[x_ax])
    n_c = int(ref_frame.shape[c_ax]) if c_ax is not None else 1
    scale_y = float(scaledict.get("y", 1.0))
    scale_x = float(scaledict.get("x", 1.0))
    print(f"  axes={axes}  frame={frame_y}x{frame_x}px  "
          f"scale=({scale_y},{scale_x}) µm/px  dtype={dtype}")

    # ── 3. Integer pixel offset of each frame on the canvas ──────────────
    # Negate stage Y: stage Y increases upward, but image rows increase downward.
    locs_um = np.array([(-f.y_um, f.x_um) for f in frames], dtype=float)
    locs_um -= locs_um.min(axis=0)                          # origin -> (0, 0)
    offsets = np.floor(locs_um / np.array([scale_y, scale_x])).astype(int)   # (N, 2)
    canvas_y = int((offsets[:, 0] + frame_y).max())
    canvas_x = int((offsets[:, 1] + frame_x).max())
    print(f"  Canvas (px): Y={canvas_y}  X={canvas_x}")

    # ── 4. Output group + full-resolution (level-0) array + metadata ─────
    outpath = _fwd(wrap_output_path(output_path))
    gr = _zarr_group(outpath, overwrite=overwrite, zarr_format=2)
    meta = _get_or_create_multimeta(gr, axis_order=axes, unit_list=units, version="0.4")

    base_scale = [float(scaledict.get(ax, 1.0)) for ax in axes]
    shp0 = list(ref_frame.shape)
    shp0[t_ax], shp0[y_ax], shp0[x_ax] = n_t, canvas_y, canvas_x
    chunk_yx = 512
    chunks0 = [1] * len(axes)
    if c_ax is not None:
        chunks0[c_ax] = n_c
    chunks0[y_ax] = min(chunk_yx, canvas_y)
    chunks0[x_ax] = min(chunk_yx, canvas_x)
    shp0 = tuple(int(v) for v in shp0)
    chunks0 = tuple(int(v) for v in chunks0)

    # Use eubi's nested "/" chunk layout (0/0/0/1/2), not zarr v2's default "."
    # separator — so level 0 matches the levels the downscaler writes.
    level0 = gr.create_array(
        "0", shape=shp0, chunks=chunks0, dtype=dtype, fill_value=0,
        chunk_key_encoding={"name": "v2", "configuration": {"separator": "/"}},
    )
    meta.add_dataset(path="0", scale=base_scale)
    meta.autocompute_omerometa(n_c, dtype)
    meta.save_changes()

    # ── 5. Sparse region writes into level 0 (single parallel dask store) ─
    print(f"Writing {n_t} frames sparsely into {outpath}/0 …")
    sources, targets, regions = [], [], []
    for i, fm in enumerate(frames):
        frame = _read_frame_array(_fwd(os.path.join(image_dir, fm.filename)), t_ax)
        if max_project and z_ax_orig is not None:
            frame = da.max(frame, axis=z_ax_orig, keepdims=False)
        oy, ox = int(offsets[i, 0]), int(offsets[i, 1])
        region = _axis_slices(frame.ndim, {
            t_ax: slice(i, i + 1),
            y_ax: slice(oy, oy + frame_y),
            x_ax: slice(ox, ox + frame_x),
        })
        sources.append(frame)
        targets.append(level0)
        regions.append(region)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  queued frame {i + 1}/{n_t}: {fm.filename}")
    # Disjoint T slices per frame → no two writes share a chunk, so lock=False.
    da.store(sources, targets, regions=regions, lock=False)

    # ── 6. Build pyramid levels 1..N with eubi's tensorstore downscaler ──
    #      Much faster than dask/coarsen: it reads the sparse level-0 canvas
    #      (empty chunks read back as fill) and writes the smaller levels.
    if n_layers > 1:
        scale_factors = tuple(2 if ax in ("y", "x") else 1 for ax in axes)
        print(f"Downscaling {n_layers - 1} level(s) via tensorstore "
              f"(factors={scale_factors}) …")
        await downscale_with_tensorstore_async(
            base_store=f"{outpath}/0",
            scale_factor=scale_factors,
            n_layers=n_layers,
            downscale_method="mean",
            min_dimension_size=1,
            chunks=chunks0,
            shards=chunks0,
            max_concurrency=4,
            region_size_mb=256.0,
        )
    print(f"Done. Written to {outpath}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    image_dir: str,
    positions_csv: str,
    output_path: str,
    pattern: str = "",
    n_pyramid_layers: int = 4,
    overwrite: bool = False,
    max_project: bool = False,
) -> None:
    """
    Translate CZI time-series frames to stage positions and write one OME-Zarr.

    Parameters
    ----------
    image_dir : str
        Directory containing the .czi files.
    positions_csv : str
        Path to the CSV file with FileName, Stage_Position_X, Stage_Position_Y.
    output_path : str
        Output .zarr path.
    pattern : str
        Optional substring to filter filenames (e.g. 'P0001').
        If empty, all rows in the CSV are processed.
    n_pyramid_layers : int
        Number of multiscale levels to generate, including full resolution
        (default 4).
    overwrite : bool
        Overwrite existing output (default False).
    max_project : bool
        If True, collapse the Z axis by max-projection before writing.
        The output will have no Z dimension (default False).
    """
    asyncio.run(translate_and_write(
        image_dir=image_dir,
        positions_csv=positions_csv,
        output_path=output_path,
        pattern=pattern,
        n_pyramid_layers=n_pyramid_layers,
        overwrite=overwrite,
        max_project=max_project,
    ))


def _cli() -> None:
    """Entry point for the ``translate-to-zarr`` command."""
    import fire
    fire.Fire(main)


if __name__ == "__main__":
    _cli()
