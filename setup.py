# -*- coding: utf-8 -*-
"""
@author: bugra
"""

import setuptools
# from ome_zarr_pyramid.process.process_utilities import get_functions_with_params

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

def parse_requirements(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [line.strip() for line in fid.readlines() if line]
    return requires

def readme():
   with open('README.txt') as f:
       return f.read()

# requirements = parse_requirements('requirements.txt')

setuptools.setup(
    name = 'centriole_tracking_preprocess',
    version = '0.0.1',
    author = 'Bugra Özdemir',
    author_email = 'bugraa.ozdemir@gmail.com',
    description = 'A package for preprocessing of feedback microscopy time-course data for tracking.',
    long_description = readme(),
    long_description_content_type = "text/markdown",
    # url = 'https://github.com/Euro-BioImaging/ome_zarr_pyramid',
    # license = 'MIT',
    packages = setuptools.find_packages(),
    include_package_data=True,
    # install_requires = requirements,
    entry_points={'console_scripts': [
                                      "maxproject_reframe_concatenate = centriole_tracking_preprocess.main:maxproject_reframe_concatenate_exe"
                                    ]
                  }
    )
