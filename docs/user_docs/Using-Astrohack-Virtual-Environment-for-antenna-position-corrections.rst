    :Author: Victor de\ :sub:`Souza`\ \ :sub:`Magalhaes`\

.. contents::

1 Foreword
----------

This guide assumes that astrohack was installed in a virtual
environment (venv) with the convenience functions as described `here <./Installing-astrohack-in-a-virtual-environment>`_
and that the data has been downloaded as an ASDM.

2 Downloading reduction scripts
-------------------------------

The first step in the data reduction is to download the scripts to be
used in the data reduction process, this can be done by using one of the
convenience functions created by the installation script:
\`\`\`bash
$ get\ :sub:`locit`\ \ :sub:`scripts`\
\`\`\`
This will download 3 scripts to the current working directory:'

- pre\ :sub:`locit`\ \ :sub:`script.py`\ : this is the CASA script for data calibration.

- exec-locit.py : This is the script that runs locit and produces the
  relevant plots.

- export\ :sub:`to`\ \ :sub:`parminator.py`\ : This is the script that exports locit
  results to a parminator compliant file.

It is expected that this scripts will change due to new requests made
by VLA's Data analysts and/or due to bugs found by them. Hence it is
good practice to redownload them each time a reduction is to be done
to garantee that the user has the latest version of the script.

3 Running pre\ :sub:`locit`\ \ :sub:`script.py`\ in CASA
--------------------------------------------------------

This script contains a basic command line interface to control its
parameters, because of this it must be run in CASA in the command line
mode, i.e.:

\`\`\`bash
$ casa -c pre\ :sub:`locit`\ \ :sub:`script.py`\ -h
\`\`\`

The output of the pre\ :sub:`locit`\ \ :sub:`script.py`\ script called with the **-h** flag
is the help which should look like this:

\`\`\`
$ casa -c pre-locit-script.py baseline.ms teste1 -h

optional configuration file config.py not found, continuing CASA startup without it

Using matplotlib backend: TkAgg
Telemetry initialized. Telemetry will send anonymized usage statistics to NRAO.
You can disable telemetry by adding the following line to the config.py file in your rcdir (e.g. ~/.casa/config.py):
telemetry\ :sub:`enabled`\ = False
--> CrashReporter initialized.
CASA 6.5.5.21 -- Common Astronomy Software Applications [6.5.5.21]


------------

usage: pre-locit-script.py [-h] [-m] [-f FRINGE\ :sub:`FIT`\ \ :sub:`SOURCE`\] [-r REFERENCE\ :sub:`ANTENNA`\] [-s SCANS\ :sub:`TO`\ \ :sub:`FLAG`\] input\ :sub:`dataset`\ output\ :sub:`name`\

CASA pre-locit script
Execute fringe fit, averaging and phase cal to produce the cal table to ingested by astrohack's locit

positional arguments:
  input\ :sub:`dataset`\         Input dataset usually an ASDM
  output\ :sub:`name`\           Base name for output name produced by this script

optional arguments:
  -h, --help            show this help message and exit
  -m, --is\ :sub:`ms`\           Input file is a MS rather than an ASDM
  -f FRINGE\ :sub:`FIT`\ \ :sub:`SOURCE`\, --fringe\ :sub:`fit`\ \ :sub:`source`\ FRINGE\ :sub:`FIT`\ \ :sub:`SOURCE`\
                        Fringe fit source, default is 0319+415
  -r REFERENCE\ :sub:`ANTENNA`\, --reference\ :sub:`antenna`\ REFERENCE\ :sub:`ANTENNA`\
                        Choose reference antenna for fringe fit and phase cal, default is ea23
  -s SCANS\ :sub:`TO`\ \ :sub:`FLAG`\, --scans\ :sub:`to`\ \ :sub:`flag`\ SCANS\ :sub:`TO`\ \ :sub:`FLAG`\
                        Comma separated list of scans to flag, default is no scan to flag
\`\`\`

The user can then run the script by specifying the ASDM name (or a MS
name if importasdm has already been run in advance by using flag **-m**)
and an **output\ :sub:`name`\**. This **output\ :sub:`name`\** will be the basename for the
products created by this script.  The script uses **0319+415** as the
default fringe fit source, and antenna **ea23** as the default reference
antenna, if either of those are not present on the dataset the script
will throw an error before executing any CASA step.

A typical execution call will look like this:

\`\`\`bash 
$ casa -c pre\ :sub:`locit`\ \ :sub:`script.py`\ my\ :sub:`dl`\ \ :sub:`asdm`\ baseline-241125 
\`\`\`

Where **my\ :sub:`dl`\ \ :sub:`asdm`\** is the name of the downloaded ASDM and
**baseline-241125** is the basename for the output files created by the
script.

Execution can be quite long as fringe fit may take close to 1 hour to
execute depending on the size of the pointing dataset.

4 Running exec\ :sub:`locit.py`\
--------------------------------

For running **exec\ :sub:`locit.py`\** we need to first activate the astrohack
venv:

\`\`\`bash
$ activate\ :sub:`astrohack`\
\`\`\`

With the venv activated we can then call **exec\ :sub:`locit.py`\** with the **-h**
flag to have a look at its help:

\`\`\`
$ python exec\ :sub:`locit.py`\ -h
usage: exec\ :sub:`locit.py`\ [-h] [-d] [-a ANTENNAS] [-c COMBINATION] [-p POLARIZATION] [-k] [-e ELEVATION\ :sub:`LIMIT`\] [-f FIT\ :sub:`ENGINE`\] caltable

Execute locit with a phase cal table produced by CASA

This script executes a subset of locit's features, for a more detailed tutorial see:
`https://astrohack.readthedocs.io/en/stable/locit_tutorial.html <https://astrohack.readthedocs.io/en/stable/locit_tutorial.html>`_

positional arguments:
  caltable              Phase cal table

options:
  -h, --help            show this help message and exit
  -d, --display\ :sub:`plots`\   Display plots during script execution
  -a ANTENNAS, --antennas ANTENNAS
                        Comma separated list of antennas to be processed, default is all antennas
  -c COMBINATION, --combination COMBINATION
                        How to combine different spws to for locit processing, valid values are: no, simple or difference, default is simple
  -p POLARIZATION, --polarization POLARIZATION
                        Which polarization hands to be used for locit processing, for the VLA options are: both, L or R, default is both
  -k, --fit\ :sub:`kterm`\       Fit antennas K term (i.e. Offset between azimuth and elevation axes)
  -e ELEVATION\ :sub:`LIMIT`\, --elevation\ :sub:`limit`\ ELEVATION\ :sub:`LIMIT`\
                        Lowest elevation of data for consideration in degrees, default is 10
  -f FIT\ :sub:`ENGINE`\, --fit\ :sub:`engine`\ FIT\ :sub:`ENGINE`\
                        Choose the fitting engine, default is "scipy" other available engine is "linear algebra"
\`\`\`

Several options are available, but usually only **--antennas** and
**--combination** will be used as they control the antennas for which we
want position correction solutions and how to combine the different
spectral windows to obtain a solution.  The flag **-d** can be used to
display the plots as the script is executing.

Below is an example call to **exec\ :sub:`locit.py`\** where we specify only a
few antennas for which we want antenna position corrections and that
we want to combine the spectral windows using the phase difference
between them.

\`\`\`bash
$ python exec\ :sub:`locit.py`\ baseline-241125-pha.cal -a 'ea06,ea13,ea27' -c difference
\`\`\`

5 Exporting results to parminator
---------------------------------

After the user is satisfied with the results they can export the results
to a parminator file by calling the **export\ :sub:`to`\ \ :sub:`parminator.py`\** script.
Like the other scripts it has a help that can be accessed with the
**-h** flag:

\`\`\`
$ python export\ :sub:`to`\ \ :sub:`parminator.py`\ -h
usage: export\ :sub:`to`\ \ :sub:`parminator.py`\ [-h] [-t CORRECTION\ :sub:`THRESHOLD`\] [-a ANTENNAS] position\ :sub:`file`\ parminator\ :sub:`file`\

Export position corrections to parminator

This script executes a subset of locit's features, for a more detailed tutorial see:
`https://astrohack.readthedocs.io/en/stable/locit_tutorial.html <https://astrohack.readthedocs.io/en/stable/locit_tutorial.html>`_

positional arguments:
  position\ :sub:`file`\         position.zarr file produced by locit
  parminator\ :sub:`file`\       Name for the output parminator file

options:
  -h, --help            show this help message and exit
  -t CORRECTION\ :sub:`THRESHOLD`\, --correction\ :sub:`threshold`\ CORRECTION\ :sub:`THRESHOLD`\
                        Threshold for including corrections in meters, default is 0.01
  -a ANTENNAS, --antennas ANTENNAS
                        Comma separated list of antennas to be processed, default is all antennas
\`\`\`

A typical call to **export\ :sub:`to`\ \ :sub:`parminator.py`\** shall look like this:

\`\`\`bash
$ python export\ :sub:`to`\ \ :sub:`parminator.py`\ baseline-241125.position.zarr 241125-baseline.par -t 0.05 -a 'ea13,ea27'
\`\`\`

In this call we have chosen a threshold for corrections of 5 cm and to
only export corrections for antennas ea13 and ea27 which will be
exported to a file called 241125-baseline.par.
