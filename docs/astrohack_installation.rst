Astrohack Installation
~~~~~~~~~~~~~~~~~~~~~~

We provide installation instructions for two types of installations,
one inside an Anaconda environment the other by using a provided
installation script that will create a python virtual environment and
install astrohack in that environment as well as a few convenience
functions to simplify the usage of astrohack.

Installation in an Anaconda environment is the recommended way for
most users. If that is not possible, such as is the case for some at
NRAO, the installation script is the suggested way to go.


Installation under Anaconda
###########################

When installing Astrohack in an `Anaconda
<https://docs.conda.io/projects/conda/en/latest/>`_ environment it is
recommended to start with a fresh environment, Preferabilly under
python3.12, as it is the most recent, and also fastest, version of
python supported by astrohack. A fresh environment is recommended as
to avoid conflicting dependencies with other packages. To create such
an environment:

.. code-block:: sh
		
   $ conda create --name astrohack python=3.12 --no-default-packages
   $ conda activate astrohack

On macos it is required to pre-install `python-casacore
<https://github.com/casacore/python-casacore>`_, before installing
astrohack:

.. code-block:: sh
		
   $ conda install -c conda-forge python-casacore

Astrohack is not yet available for download directly from conda-forge,
therefore we suggest to install astrohack by using pip:

.. code-block:: sh
		
   $ pip install astrohack

It is also possible to install astrohack from source by downloading
the `source code
<https://github.com/casangi/astrohack/archive/refs/heads/astrohack-dev.zip>`_
directly from github. With the zip extracted you can then navigate to
the root directory and make a local pip installation:

.. code-block:: sh
		
   $ cd <Astrohack_root_dir>
   $ pip install -e .
		

Installation in NRAO machines using the provided installation script
####################################################################

For convenience we have created an installation script for astrohack
that installs it in a python virtual environment (venv).  This venv is
expected to be created using python 3.11 and is usually installed at
**~/.local/python/venvs**.  The last thing that the installation script
does is to create convenience functions that are highly recommended as
they simplify the use of the venv.

WARNING: VENVS ARE NOT PORTABLE, these instructions must be executed
on the machine the user intends to use for astrohack reduction.

Downloading and installing
----------------------------

The installation script can be downloaded `here
<https://github.com/casangi/astrohack/raw/main/etc/installation/astrohack-install.sh>`_. To
run the installation script it is necessary to make it executable, and
then it can be executed:

.. code-block:: sh
		
   $ chmod u+x astrohack_install.sh
   $ ./astrohack_install.sh

There will be 3 questions during execution:

1. Installation location, for default just press **<enter>**.

2. Python3.11 executable for venv creation, for default just press **<enter>**.

3. The name for the venv, for default just press **<enter>**.

After this questions the script will first create the venv, followed
by updating pip and then it installs astrohack.  When all of this is
done (should take a few minutes to download and install all astrohack
dependencies), the script will then ask if the user wants to install
the convenience functions.  Unless the user is reinstalling with the
same venv name and location the user should always install the
convenience functions.

When the installation is done the user can check that it worked by
invoking the environment:

.. code-block:: sh
		
   $ activate_astrohack

If the venv is correctly installed the prompt will change by the
addition of the venv name. For a simple test, the user can open an
Ipython session inside the venv and try to import astrohack:

.. code-block:: python
		
   from astrohack import locit

If the installation happened without any problems the user will see no
error messages.  To exit the venv the user should use the deactivate
command:

.. code-block:: sh
		
   $ deactivate

After installing the user can then go on to `Obtaining Antenna Position
corrections <./Using-Astrohack-Virtual-Environment-for-antenna-position-corrections>`_.

Updating astrohack inside the VENV
------------------------------------

Every now and then a new release of astrohack will come along with new
functionalities and/or bug fixes. To update astrohack it is necessary
to first get into the venv and then we can call pip to update
astrohack:

.. code-block:: sh
		
   $ activate_astrohack
   $ pip install astrohack --upgrade

If the user is not going to use astrohack straight away it is
recommended to deactivate the venv after doing the update:

.. code-block:: sh
		
   $ deactivate

Installation or execution problems
------------------------------------

If the user encounters any issues during installation and/or execution
of astrohack they should leave an issue here on github or write an
e-mail to Victor de Souza.
