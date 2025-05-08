.. image:: https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue
   :target: https://www.python.org/downloads/release/python-3130/

.. image:: https://github.com/casangi/astrohack/actions/workflows/python-testing-linux.yml/badge.svg?branch=main
   :target: https://github.com/casangi/astrohack/actions/workflows/python-testing-linux.yml?query=branch%3Amain

.. image:: https://github.com/casangi/astrohack/actions/workflows/python-testing-macos.yml/badge.svg?branch=main
   :target: https://github.com/casangi/astrohack/actions/workflows/python-testing-macos.yml?query=branch%3Amain
	   
.. image:: https://github.com/casangi/astrohack/actions/workflows/run-ipynb.yml/badge.svg?branch=main
   :target: https://github.com/casangi/astrohack/actions/workflows/run-ipynb.yml?query=branch%3Amain
	   

.. image:: https://codecov.io/gh/casangi/astrohack/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/casangi/astrohack/branch/main/astrohack


.. image:: https://readthedocs.org/projects/astrohack/badge/?version=latest
   :target: https://astrohack.readthedocs.io

.. image:: https://img.shields.io/pypi/v/astrohack.svg
   :target: https://pypi.python.org/pypi/astrohack/

Holography Antenna Commissioning Kit
====================================

.. image:: https://github.com/casangi/astrohack/blob/astrohack-dev/docs/_media/astrohack_logo.png?raw=true
   :align: left

AstroHACK (Holography and Antenna Comissioning Kit) is a python
package intended for NRAO telescope support activities, such as
holography and antenna position corrections.

Currently AstroHACK supports:

- `Astronomical holographies from the VLA <https://astrohack.readthedocs.io/en/stable/tutorial_vla.html>`_.
- Astronomical holographies from ALMA.
- `Antenna position corrections for the VLA <https://astrohack.readthedocs.io/en/stable/locit_tutorial>`_.
- `Near-field ALMA holographies <./https://astrohack.readthedocs.io/en/stable/AstroHACK-for-NF-ALMA.html>`_ have
  basic support, such as the creation of Aperture images, but the
  correction of near-field effects is not yet supported.

In the near term we are working to support the `ngVLA antenna
prototype <https://public.nrao.edu/ngvla/>`_ comissioning activities.
The prototype antenna is currently (March 2025) under construction at
the VLA site.

The holography tasks in AstroHACK port much of the functionality of
AIPS holography tasks (UVHOL, HOLOG and PANEL) to a python framework
that enables us to take advantage of modern parallelization and code
acceleration technologies, such as `Dask <https://www.dask.org/>`_ and
`Numba <https://numba.pydata.org/>`_. Making the reduction of
holography data much faster and streamlined for the user.


> 📝 astroHACK is under active development! Breaking API changes are
still happening on a regular basis, so proceed with caution.
