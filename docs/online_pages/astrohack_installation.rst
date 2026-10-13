Astrohack Installation
~~~~~~~~~~~~~~~~~~~~~~

Installation under Anaconda
###########################

When installing AstroHACK in an `Anaconda
<https://docs.conda.io/projects/conda/en/latest/>`_ environment it is
recommended to start with a fresh environment, Preferably under
python 3.13, as it is the most recent, and also fastest, version of
python supported by AstroHACK. A fresh environment is recommended as
to avoid conflicting dependencies with other packages. To create such
an environment:

.. code-block:: sh
		
   $ conda create --name astrohack python=3.13 --no-default-packages
   $ conda activate astrohack

Astrohack reads MeasurementSets and CASA calibration tables through
`casacoretables <https://github.com/nrao/casacoretables>`_, a self-contained
build of casacore's table system that is installed automatically as a
dependency on both Linux and macOS. No separate ``python-casacore`` install is
required (this used to be a manual step on macOS, and is no longer needed).

Astrohack is not yet available for download directly from conda-forge,
therefore we suggest to install AstroHACK by using pip:

.. code-block:: sh

   $ pip install astrohack

Source code installation
########################

If you would like or need to be following the latest developments of AstroHACK, it is also possible to install AstroHACK from source by downloading
the `source code
<https://github.com/nrao/astrohack/archive/refs/heads/astrohack-dev.zip>`_
directly from github or using ``git clone``.

.. code-block:: sh

   $ cd <your/preferred/installation/location>
   $ git clone git@github.com:nrao/astrohack.git
   $ cd astrohack


With the zip extracted or the cloned repository via git you can then navigate to
astrohack's root directory and make a local editable (-e) pip installation:

.. code-block:: sh
		
   $ cd <Astrohack_root_dir>
   $ pip install -e .

Updating a local git installation
---------------------------------

To update a local git installation it is necessary to use git.
The default installation follows the ``astrohack-dev`` branch, i.e. the main development branch of AstroHACK, which unless you are working with active development is the branch that should be followed. The updating process is the following:

.. code-block:: sh

   $ cd <Astrohack_root_dir>
   $ git pull

If you are required to follow a specific development branch, there is one extra step:

.. code-block:: sh

   $ cd <Astrohack_root_dir>
   $ git switch <current-development-branch-name>
   $ git pull

Installation inside CASA
########################

Astrohack can now be installed inside CASA! this is possible due to a new
python package (`casacoretables <https://pypi.org/project/casacoretables/>`_)
that reimplements the access to CASA tables so that they can be accessed
inside a CASA environment without conflicts with CASA.

For installation inside CASA to work it is required that the CASA version
being used is based on python 3.12 (CASA version >= 6.7). The installation can then be done as is described above, but remembering to execute the pip commands inside casa, regular pip install:

.. code-block:: sh

   $ casa
   CASA <1>: pip install astrohack

Source code editable install:

.. code-block:: sh

   $ cd <Astrohack_root_dir>
   $ casa
   CASA <1> pip install -e .

Running CASA + AstroHACK @ NRAO
###############################

There is now a distributed way of running CASA + AstroHACK at NRAO workstations (at least in NM, CV and GBO not tested). This distributed CASA + AstroHACK bundle can be accessed through the ``casa-astrohack`` command.
As with other distributed CASA versions there are a few versions that can be accessed. The default one, stable, should suffice for most user and uses the latest release of AstroHACK. The other verions should be used with care, or only if recommended by a developer.

Installation or execution problems
##################################

If the user encounters any issues during installation and/or execution
of AstroHACK they should leave an issue here on github or write an
e-mail to Victor de Souza at NRAO.
