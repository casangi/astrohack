ALMA Near Field holographies support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current support for ALMA near field holographies is limited.  It is
possible to produce aperture images but the near field phase effects
are not yet fully corrected, making it impossible to produce panel
adjustments from them.



ALMA near field ASDM to .holog.zarr filler
==========================================

Information on the usage of the ALMA NF ASDM filler can be found in
the `etc section of our github page <https://github.com/casangi/astrohack/tree/astrohack-dev/etc/alma-nf-filler>`_.

In case of pip install you have to visit the `etc section of our github
page <https://github.com/casangi/astrohack/tree/astrohack-dev/etc/alma-nf-filler>`_ and download the .py files from there.

In case of instalation from source the .py files can be found in 
\`<astrohackdir>/etc/alma-nf-filler'.

After using the filler you will be left with a .holog.zarr file that
contains the data necessary to produce the gridded beam image and the
apertures.

Producing apertures
===================

In astrohack the function that produces apertures from .holog.zarr
files is called holog. This function grids the beam image from the
data and then Fourier transforms (FT) it to produce the apertures.  In
the case of near field holographies after the first FT subsequent
transformations are applied to correct for the non-fresnel terms
followed by a step of correction of the aperture phase from the NF
effects.

The apertures can be produced and then subsequently plot with a simple
python script. In this script we call holog with the parameter
**alma_osf_pad** set to the pad on which the antenna was positioned
during the holography measurement. We set to parallel to False as
holog parallelizes holographies per antenna and ALMA nf holographies
contain a single antenna.

After the call to holog we are left with an object that contains the
gridded beam images as well as the apertures. This object is reflect
on disk by a .image.zarr file. From this object we can acces the
plotting functions, plot\textunderscore beams and plot\textunderscore
apertures.

.. code:: python

    from astrohack import holog

    image_mds = holog(holog_name='almanf.holog.zarr',
    		  image_name='almanf.image.zarr',
    		  alma_osf_pad='TF03',
    		  parallel=False,
    		  overwrite=True # denpending on your preferences
    		  )

    # Plotting gridded beam images
    image_mds.plot_beams('almanf_plots', angle_unit='asec', display=False)
    # Plotting aperture images
    image_mds.plot_apertures('almanf_plots', polarization_state=['I'], display=False)

If after a while you want to interact once more with the image_mds
object you can use the open_image function.

.. code:: python

    from astrohack import open_image

    image_mds = open_image('almanf.image.zarr')

For more information on the API of holog and other astrohack functions
can be found `here
<https://astrohack.readthedocs.io/en/stable/api.html>`_. More detailed
instruction on how to use the plotting functions can be found on our
`visualization tutorial
<https://astrohack.readthedocs.io/en/stable/visualization_tutorial.html>`_.

Caveat
======

Currently the correction of the apertures for the NF effects is not
working as it should. Aperture phases contain significant ringing that
is not yet understood.
