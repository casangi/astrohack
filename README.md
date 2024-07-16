![astrohack](docs/_media/astrohack_logo.png)

[![Python 3.9 3.10 3.11](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/downloads/release/python-380/)
![Linux Tests](https://github.com/casangi/astrohack/actions/workflows/python-testing-linux.yml/badge.svg)
![macOS Tests](https://github.com/casangi/astrohack/actions/workflows/python-testing-macos.yml/badge.svg)
![Published](https://github.com/casangi/astrohack/actions/workflows/pythonpublish.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/astrohack/badge/?version=stable)](https://astrohack.readthedocs.io/en/stable/?badge=stable)

astroHack (Holography Antenna Commissioning Kit) is a Python package under development by NRAO's [CASA](https://casa.nrao.edu) team to support holography and antenna position correction measurements. It currently supports pointed and on-the-fly holographies for both ALMA and the VLA as well as antenna position corrections for the VLA. The future goal of astrohack is to support the commissioning of the Next Generation Very Large Array (ngVLA). Much of the core functionality of astroHACK is inspired by the code of the following AIPS tasks: UVHOL, HOLOG and PANEL for holography and LOCIT for the antenna position corrections. AstroHACK enables parallel execution by using Dask and efficient single-threaded performance by making use of Numba.

> 📝 astroHACK is under active development! Breaking API changes are still happening on a regular basis, so proceed with caution.

# Installing
It is recommended to use the [conda](https://docs.conda.io/projects/conda/en/latest/) environment manager to create a clean, self-contained runtime where astrohack and all its dependencies can be installed:
```sh
conda create --name astrohack python=3.11 --no-default-packages
conda activate astrohack

```
> 📝 On macOS it is required to pre-install `python-casacore` using `conda install -c conda-forge python-casacore`.

Making astroHACK available for download from conda-forge directly is pending, so until then the current recommendation is to sully that pristine environment by calling pip [from within conda](https://www.anaconda.com/blog/using-pip-in-a-conda-environment), like this:
```sh
pip install astrohack
```

# Tutorials

Besides the API for the user facing functions there are three tutorials that can be followed that demonstrate the capabilities of astrohack:
- [VLA Holography tutorial](https://astrohack.readthedocs.io/en/stable/tutorial_vla.html)
- [Holography visualization tutorial](https://astrohack.readthedocs.io/en/stable/visualization_tutorial.html)
- [Antenna position correction tutorial](https://astrohack.readthedocs.io/en/stable/locit_tutorial.html)
