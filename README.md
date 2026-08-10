# centriole_tracking_preprocess

This is a Python package for translating data to Zarr format.

## Installation

You can install the package using pip. Make sure you have Python 3.11 or 3.12 installed.

```bash
mamba create -n centriole_env openjdk=11.* maven python=3.12 pyqt6=6.8.1
mamba activate centriole_env
pip install centriole_tracking_preprocess
```

## Usage

To use the package, you can run the command line interface:

```bash
translate_to_zarr [OPTIONS]
```

Replace `[OPTIONS]` with the appropriate command line options for your use case.

## Development

To contribute to the project, clone the repository and install the required dependencies. You can run the tests using:

```bash
pytest
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.