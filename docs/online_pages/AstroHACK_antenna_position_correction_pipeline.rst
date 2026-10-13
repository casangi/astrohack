Antenna position Correction pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AstroHACK provides an executable script for obtaining antenna position corrections, which is installed by pip somewhere in the PATH. This script has 5 main stages:

#. ASDM import to ms (if data set has not yet been imported to an MS).

#. A CASA pre-locit stage, called calibration in the pipeline, where visibilities are channel averaged and phase solutions are computed. If using the ``--use-fringefit-locit`` option, instead of phase solutions the pipeline will use CASA`s fringefit to compute delay solutions for all the sources in the dataset.

#. A locit stage where phase solutions are extracted from a gain table and then processed to obtained antenna position corrections. If using the ``--use-fringefit-locit`` option, the pipeline will use ``fringefit_locit`` instead of ``extract_locit`` and ``locit`` to compute the antenna position corrections.

#. An export stage where data products such as plots and fitting reports are created.

#. The creation of a report grouping all the data products.

This pipeline has been written under the assumption that the user will be running it in CASA or in an environment that provides the casatasks, casaplotms and casatools modules. The instructions below assume that the pipeline is being run inside CASA.

Pipeline interface
==================

The pipeline has been written with a simple command line interface that expects one mandatory argument from the user, the name of the dataset to be processed (be it an MS or an ASDM) and if starting from calibration a second mandatory argument, a reference antenna. Several execution customization options are also available a simple help can be accessed with the ``-h`` flag:

.. code-block::

    CASA <1>: !baseline-reduction-pipeline -h

    ###################################################################################################################################################################################
    ###  Welcome to the AstroHACK baseline reduction pipeline                                                                                                                       ###
    ###################################################################################################################################################################################

    usage: baseline-reduction-pipeline [-h] [-r ROOT_NAME] [-s SPW] [-a ANTENNA] [-n NCORES] [-m MEMORY_PER_CORE] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-o] [-y]
                                       [--reimport-asdm] [--starting-stage {calibration,locit,exports,report}] [--dpi DPI] [-f FRINGEFIT_SOURCE] [-i INTENT] [-e ELEVATION_LIMIT]
                                       [-p {both,L,R}] [-c {simple,difference}] [-k] [-dr] [-l DELAY_LIMITS] [--use-fringefit-locit] [--exclude-scans EXCLUDE_SCANS]
                                       filename [refant]

    Baseline reduction pipeline

    positional arguments:
      filename              Path to the input dataset to process.
      refant                Reference antenna for calibration

    options:
      -h, --help            show this help message and exit
      -r ROOT_NAME, --root-name ROOT_NAME
                            Root name for the products of the pipeline, default is ms_name without extension
      -s SPW, --spw SPW     Select SPWs for processing, for a list use comma separated values with no spaces, e.g.: '0,1,2', default is all
      -a ANTENNA, --antenna ANTENNA
                            Select antennas for processing, for a list use comma separated values with no spaces, e.g.: 'ea01,ea02', default is all
      -n NCORES, --ncores NCORES
                            Number of cores to use, default is 4
      -m MEMORY_PER_CORE, --memory-per-core MEMORY_PER_CORE
                            Memory per core to use, the memory unit should be explicitly given, i.e. 4 means for 4B, default is 10GB
      --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Logging level to use, default is WARNING
      -o, --overwrite       Overwrite existing files if found
      -y, --assume-yes      Assume yes on proceed.
      --reimport-asdm       Forcefully re-import the asdm file is the ms already exists (default: False)
      --starting-stage {calibration,locit,exports,report}
                            Starting stage in which to start processing (default: calibration).
      --dpi DPI             Dots Per Inch for plotting, default is 300
      -f FRINGEFIT_SOURCE, --fringefit-source FRINGEFIT_SOURCE
                            Fringe fit source, default is 0319+415
      -i INTENT, --intent INTENT
                            Intent for pointing observations.
      -e ELEVATION_LIMIT, --elevation-limit ELEVATION_LIMIT
                            Lowest elevation of data for consideration in degrees, default is 10.0
      -p {both,L,R}, --polarization {both,L,R}
                            Which polarization hands to be used for locit processing, default is both
      -c {simple,difference}, --combination {simple,difference}
                            How to combine different spws for locit processing, default is simple
      -k, --fit-kterm       Fit antennas K term (i.e. Offset between azimuth and elevation axes)
      -dr, --fit-delay-rate
                            Fit delay rate
      -l DELAY_LIMITS, --delay-limits DELAY_LIMITS
                            Symmetrical limit for delay plots in nanoseconds, default is 0.1 which results in the limits being [-0.1, 0.1]
      --use-fringefit-locit
                            Use fringefit_locit to determine very large errors (order of meters) in antenna positions (EXPERIMENTAL)
      --exclude-scans EXCLUDE_SCANS
                            Select scans to exclude from locit delay fitting, for a list use comma separated values with no spaces, e.g.: '5,37', default is None


Pipeline initialization
=======================

With a reference antenna chosen it is now time to run the antenna position correction pipeline.
The first step of the pipeline is to check whether the data is an ASDM or an MS and if it is an ASDM if it needs to be imported into an MS. With an MS in hands the pipeline proceeds to fetching some metadata from it and then prints a summary of what it has found and which parameters it will use for calibration and further data reduction, e.g.:

.. code-block::

    CASA <3>: !baseline-reduction-pipeline short_x.ms ea13 -f 2148+611

    #####################################################################################################################################
    ###  Welcome to the AstroHACK baseline reduction pipeline                                                                         ###
    #####################################################################################################################################

    2026-10-07 17:53:05	INFO	msmetadata_cmpt.cc::open	Performing internal consistency checks on short_x.ms...

    Baseline determination parameters:
        filename            => short_x.ms
        refant              => ea13
        root_name           => None
        spw                 => all
        antenna             => all
        ncores              => 4
        memory_per_core     => 10GB
        log_level           => WARNING
        overwrite           => False
        assume_yes          => False
        reimport_asdm       => False
        starting_stage      => calibration
        dpi                 => 300
        fringefit_source    => 2148+611
        intent              => CALIBRATE_POINTING#ON_SOURCE
        elevation_limit     => 10.0
        polarization        => both
        combination         => simple
        fit_kterm           => False
        fit_delay_rate      => False
        delay_limits        => [-0.1, 0.1]
        use_fringefit_locit => False
        exclude_scans       => None
        is_asdm             => False
        msname              => short_x.ms
        pointing_only_ms    => short_x.pnt.ms
        freq_averaged_ms    => short_x.avg.ms
        fringefit_caltable  => short_x.sbd
        phase_caltable      => short_x.pha.gcal
        antpos_caltable     => short_x.antpos
        locit_name          => short_x.locit.zarr
        position_name       => short_x.position.zarr
        exports_name        => short_x.exports
        report_name         => short_x-report.html
        parallel            => True
        n_chan              => 64


    Proceed? <(Y)es/(N)o>:


The check before proceeding can be suppressed by adding the ``-y`` option to the call, e.g.:

.. code-block::

    CASA <4>: !baseline-reduction-pipeline short_x.ms ea13 -y


After initialization there are two possible processing flows in the pipeline, one using regular locit, i.e. the classical approach dependent on gain phases; the other using fringefit_locit, which is less accurate but is not constrained by λ ambiguities.

Regular locit flow
==================

The first steps are taken using CASA tasks:

#. Split the data to contain only the pointing scans with ``split``.

#. Perform a ``fringefit`` of one bright source in the pointing only MS to obtain a delay estimate for each spectral window.

#. Apply the fringefit computed delays with ``applycal``.

#. Average all channels in the now phase aligned spectral windows with ``split``.

#. Obtain phase solutions for all sources using ``gaincal(calmode="p")``

#. Apply the phase solutions to the channel averaged ms with ``applycal`` and then plot then with ``plotms`` for user inspection (they are now expected to be clustered around 0).

After these steps using casa tasks the pipeline then proceeds to the AstroHACK steps:

#. `extract_locit <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_locit/index.html>`_: extract phase gains from the gain table and stored then in a convenient format for further processing.

#. `locit <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/locit/index.html>`_: Process phases from all spectral windows either combined or through their differences to produce antenna position solutions.

When solutions are not good, i.e. there is no considerable improvement of antenna phase RMS after the new antenna positions, it is sometimes useful to try combining the spectral windows using the difference method instead of the the default simple combination. The default method simply try to solve the antenna position corrections by using all available data points from all the spectral windows which may not be sufficient if the antenna position error is larger than or comparable to the wavelength of the observation. The difference method then comes in by taking the difference in phase between 2 spectral windows to leverage measuring the delay over a smaller frequency, the frequency difference between the 2 spectral windows, allowing us to measure much larger delays. To use the difference method, without having to start from scratch:

.. code-block::

    CASA <5>: !baseline-reduction-pipeline short_x.ms -y -c difference --starting-stage locit


fringefit_locit flow
====================

The ``fringefit_locit`` flow contains only 3 steps, however it usually takes much longer than the regular flow as using ``fringefit`` to compute delays for all baselines, sources and spectral windows can take a long time. THis method is also less precise than the classical method as the delay solutions usually have a smaller SNR, the advantage of this method is that it is not restricted by λ ambiguities and can be used when the position of the antenna is very poorly known (e.g. 10s of meters of error). The processing steps are:

#. Split the data to contain only the pointing scans with ``split``.

#. Perform a ``fringefit`` over all sources, with the baselines between the chosen antennas and the refernce antenna and for all selected spectral windows.

#. run `fringefit_locit <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/fringefit_locit/index.html>`_ on the caltable generated by ``fringefit``.

Fro both flows, in case of failures or there is a desire to re run the pipeline from a particular stage, the user can then use option ``--starting-stage``.
For more details on the antenna position corrections processing stages there is the more detailed `locit tutorial <https://astrohack.readthedocs.io/en/stable/tutorials/locit_tutorial.html>`_.

Export & report stages
======================

After the AstroHACK position file is created, both flows follow the same path, the pipeline proceeds to execute the exporting functions from the associated Python classes:

#. `AstrohackLocitFile.plot_source_positions <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/locit_mds/index.html>`_: Single plot showing the positions in the sky of the sources used for obtaining antenna position corrections.

#. `AstrohackLocitFile.plot_array_configuration <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/locit_mds/index.html>`_: Single plot displaying the array configuration at observation time.

#. `AstrohackPositionFile.export_locit_fit_results <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a single table with all antenna position corrections.

#. `AstrohackPositionFile.export_results_to_parminator <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a parminator file with proposed antenna position corrections to bea applied at the correlator.

#. `AstrohackPositionFile.plot_position_corrections <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a single plot with arbitrarily scaled antenna position corrections over the plot of the array configuration to have a graphical representation of antenna corrections.

#. `AstrohackPositionFile.plot_delays <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/position_mds/index.html>`_: Produce a plot per antenna showing the measured delays, the modeled delays and the residual delays.

For the classical method there is an extra step. After creating the AstroHACK plots the pipeline then:

#. Produce an antenna position correction calibration table using CASA's ``gencal``.

#. Apply the antenna position corrections to the channel averaged MS using ``applycal``.

#. Produce plots of the phase over time for the raw and baseline corrected data.

After the production of these export products the pipeline then creates a standalone HTML report with all of them that can then be stored or shared without the need to carry any extra data, an example of such a report can be seen `here <../example-baseline-short_x-report.html>`_.


