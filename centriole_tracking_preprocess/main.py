import copy, glob, tempfile, shutil, warnings

import dask.array as da
import fire

from ome_zarr_pyramid import Pyramid, LabelPyramid, PyramidCollection, Converter
from ome_zarr_pyramid import basic, aggregative
from dataclasses import dataclass
import json, os
import numpy as np, zarr
import pprint
import itertools
import csv


def _read_csv_to_dicts(file_path, delimiter = ','):
    dict_list = []
    with open(file_path, mode='r', encoding="utf-8-sig") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter = delimiter)
        for row in csvreader:
            dict_list.append(dict(row))
    return dict_list


def _get_translation_meta(dict_list):
    translations = []
    @dataclass
    class TranslationMeta:
        filename: str
        x: float
        y: float
        @property
        def translation(self):
            return {'x': self.x, 'y': self.y}
    for d in dict_list:
        if d['FileName::FileName'] != '':
            namebase = os.path.splitext(d['FileName::FileName'])[0]
            xtrans = d['Stage_Position_X::Stage_Position_X']
            ytrans = d['Stage_Position_Y::Stage_Position_Y']
            translations.append(TranslationMeta(namebase, x = xtrans, y = ytrans))
    return translations


def maxproject_reframe_concatenate(workdir: str,
                                   positions_file_path: str = '',
                                   input_files_dir: str = '',
                                   pattern: str = f"P0001",
                                   filetype: str = f"czi",
                                   n_jobs: int = 8,
                                   use_projections: bool = True,
                                   overwrite_ome_zarr_dir = False
                                   ):
    cvt = Converter(n_jobs = n_jobs)
    ops = basic.BasicOperations(n_jobs = n_jobs)
    ome_zarr_dir = f"{workdir}/input_ome_zarr_{pattern}"
    max_proj_dir = f"{workdir}/max_proj_{pattern}"
    output_dir = f"{workdir}/output_{pattern}"

    if len(positions_file_path) == 0:
        positions_file_path = f"{workdir}/Positions_data.csv"

    if len(input_files_dir) == 0:
        input_files_dir = f"{workdir}/input_files"

    if os.path.exists(ome_zarr_dir):
        if overwrite_ome_zarr_dir:
            warnings.warn(f"The folder {os.path.basename(ome_zarr_dir)} exists already. Removing existing directory.")
            shutil.rmtree(ome_zarr_dir)
        else:
            warnings.warn(f"The folder {os.path.basename(ome_zarr_dir)} exists already. Using existing OME-Zarrs.")

    if not os.path.exists(ome_zarr_dir):
        cvt.to_omezarrs(input_dir = input_files_dir, output_dir = ome_zarr_dir, pattern = f"*{pattern}*{filetype}*",
                        chunk_x = 96, chunk_y = 96, chunk_z = 6, resolutions = 3, compression = 'zlib')
        print('Conversions complete.')

    ome_zarrs = sorted(glob.glob(ome_zarr_dir + '/' + f"*{pattern}*"))

    if not use_projections:
        pyrs = [Pyramid().from_zarr(item) for item in ome_zarrs]
    else:
        pyrs = []
        for i, item in enumerate(ome_zarrs):
            pyr = Pyramid().from_zarr(item)
            name = os.path.basename(item)
            namebase = os.path.splitext(name)[0]
            proj_path = os.path.join(max_proj_dir, namebase + '_max_proj.zarr')
            proj = ops.max(pyr, axis = 'z', out = proj_path)
            pyrs.append(proj)
        print(f"Maxiumum intensity projections complete.")

    pyr = pyrs[0]
    scaleyx = np.array(pyr.scales['0'])[pyr.index('yx')]
    shapeyx = np.array(pyr.shape)[pyr.index('yx')]

    dicts = _read_csv_to_dicts(positions_file_path)
    translations = _get_translation_meta(dicts)
    translations = [translation for translation in translations if pattern in translation.filename]
    locs = np.array([(float(trans.y), float(trans.x)) for trans in translations])
    normalized_locs = locs - locs.min(axis = 0)

    min_positions = normalized_locs / scaleyx # min positions in voxels
    max_positions = min_positions + shapeyx # max positions in voxels

    frameyx = np.around(max_positions.max(axis = 0)).astype(int)
    frame = np.array(pyr.shape)
    frame[pyr.index('yx')] = frameyx
    frame[pyr.index('t')] = len(normalized_locs)
    pyrnew = Pyramid()
    pyrnew.parse_axes(pyr.axis_order)
    zarr_meta = {'store': output_dir,
                 'dimension_separator': pyr.dimension_separator,
                 'compressor': pyr.compressor,
                 'dtype': pyr.dtype
                 }
    arr = da.zeros(frame, chunks = pyr.chunks, dtype = pyr.dtype)
    pyrnew.add_layer( arr,
                      pth = '0',
                      scale = pyr.scales['0'],
                      zarr_meta = zarr_meta,
                      axis_order = pyr.axis_order,
                      unitlist = pyr.unit_list
                      )
    print(f"Empty frame created.")

    pyrnew.to_zarr(output_dir)
    pyrnew = Pyramid().from_zarr(output_dir)
    for i, (loc, pyr) in enumerate(zip(min_positions, pyrs)):
        yloc0, xloc0 = np.around(loc).astype(int)
        yloc1 = yloc0 + pyr.shape[pyr.index('y')]
        xloc1 = xloc0 + pyr.shape[pyr.index('x')]
        pyrnew[0][i: i+1, ..., yloc0:yloc1, xloc0:xloc1] = pyr[0]
        # print(f"Timepoint {i} has been inserted with max value {np.max(pyrnew[0][0])}")
    print(f"Concatenation of the Zarr arrays is complete.")
    return pyrnew


def maxproject_reframe_concatenate_exe():
    _ = fire.Fire(maxproject_reframe_concatenate)
    return

# if __name__ == '__main__':
#     fire.Fire(maxproject_reframe_concatenate)
