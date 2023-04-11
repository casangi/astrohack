# astroHACK

[![Python 3.8 3.9 3.10](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/astrohack/badge/?version=latest)](https://astrohack.readthedocs.io/en/latest/?badge=latest)


astroHack (Holography Antenna Commissioning Kit) is a Python package that produces antenna aperture images and panel adjustment corrections from calibrated holography measurement sets. Initially, it will support holography data from the VLA and ALMA, with the future goal of supporting the ngVLA. Much of the core functionality from the following AIPS tasks has been ported: UVHOL, HOLOG, and PANEL. astroHack enables parallel execution by using Dask and efficient single-threaded performance by making use of Numba.

> 📝 astroHACK is under active development! Breaking API changes are still happening on a regular basis, so proceed with caution.

# Installing
It is recommended to use the [conda](https://docs.conda.io/projects/conda/en/latest/) environment manager to create a clean, self-contained runtime where astrohack and all its dependencies can be installed:
```sh
conda create --name astrohack python=3.8 --no-default-packages
conda activate astrohack

```
> 📝 On macOS it is required to pre-install `python-casacore` using `conda install -c conda-forge python-casacore`.

Making astroHACK available for download from conda-forge directly is pending, so until then the current recommendation is to sully that pristine environment by calling pip [from within conda](https://www.anaconda.com/blog/using-pip-in-a-conda-environment), like this:
```sh
pip install astrohack
```
