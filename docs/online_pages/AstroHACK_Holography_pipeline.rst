Holography data reduction pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AstroHACK provides an executable script for the reduction of holography data from the VLA, which is installed by pip somewhere in the PATH. This script has 5 main stages:

#. ASDM import to ms (if data set has not yet been imported to an MS).

#. Calibration of the holography data using CASA tasks (delay, bandpass and phase).

#. Holography processing with AstroHACK (extract_pointing, extract_holog, holog (optionally combine), panel).

#. Data exports generation (Beam plots, aperture plots, zernike fit plots, gains, screw adjustments, etc).

#. HTML report creation (by combining all plots and text products onto a single standalone HTMl to be shared or stored).

This pipeline has been written under the assumption that the user will be running it in CASA or in an environment that provides the casatasks and casatools modules. If casatasks and casatools are not available the pipeline can be run from the extract_pointing stage assuming that the MS is already properly calibrated. The instructions below assume that the pipeline is being run inside CASA.

Pipeline interface
##################

The pipeline has been written with a simple command line interface that expects one mandatory argument from the user, the name of the dataset to be processed (be it an MS or an ASDM). If starting from the calibration stage a reference antenna is also a mandatory argument. Several execution customization options are available, a simple help can be accessed with the ``-h`` flag:

.. code-block::

    CASA <1>: !holography-reduction-pipeline -h

    ###################################################################################################################################################################################
    ###  Welcome to the AstroHACK holography reduction pipeline                                                                                                                     ###
    ###################################################################################################################################################################################

    usage: holography-reduction-pipeline [-h] [-r ROOT_NAME] [-s SPW] [-a ANTENNA] [-n NCORES] [-m MEMORY_PER_CORE] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-o] [-y]
                                         [--reimport-asdm] [--starting-stage {calibration,extract_pointing,extract_holog,holog,panel,exports,report}] [--dpi DPI] [-d DATA_COLUMN]
                                         [--plot-pointing] [--exclude-bad-antennas EXCLUDE_BAD_ANTENNAS] [-q QUACK_NCHAN] [-f HOLOGRAPHY_FIELD]
                                         [--baseline-average-nearest BASELINE_AVERAGE_NEAREST] [--pointing-interpolation-method {linear,gaussian}] [--grid-size GRID_SIZE]
                                         [--cell-size CELL_SIZE] [--grid-interpolation-mode GRID_INTERPOLATION_MODE] [--padding-factor PADDING_FACTOR]
                                         [--zernike-n-order ZERNIKE_N_ORDER] [-z] [-c] [--clip-type {sigma,none,absolute,relative,noise_threshold}] [--clip-level CLIP_LEVEL]
                                         [--panel-model {mean,rigid,flexible,corotated_scipy,corotated_lst_sq,corotated_robust,xy_paraboloid,rotated_paraboloid,full_paraboloid_lst_sq}]
                                         [--panel-margins PANEL_MARGINS] [-u SCREW_UNIT]
                                         filename [refant]

    Holography reduction pipeline

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
                            Memory per core to use, if no unit is given GB is assumed, default is 10GB
      --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Logging level to use, default is WARNING
      -o, --overwrite       Overwrite existing files if found
      -y, --assume-yes      Assume yes on proceed.
      --reimport-asdm       Forcefully re-import the asdm file is the ms already exists (default: False)
      --starting-stage {calibration,extract_pointing,extract_holog,holog,panel,exports,report}
                            Starting stage in which to start processing (default: calibration).
      --dpi DPI             Dots Per Inch for plotting, default is 300
      -d DATA_COLUMN, --data-column DATA_COLUMN
                            Data column to be extracted from MS, default is CORRECTED_DATA
      --plot-pointing       Plot antenna pointing, default is False
      --exclude-bad-antennas EXCLUDE_BAD_ANTENNAS
                            Exclude antennas with bad data, for a list use comma separated values with no spaces, e.g.: 'ea18,ea01', default is None.
      -q QUACK_NCHAN, --quack-nchan QUACK_NCHAN
                            Number of channels to quack at the edge of the spectral window (default is 4)
      -f HOLOGRAPHY_FIELD, --holography-field HOLOGRAPHY_FIELD
                            Field Id or name containing holography data (default is to determine it from data)
      --baseline-average-nearest BASELINE_AVERAGE_NEAREST
                            Number of baselines to average for each mapping antenna (default is 1)
      --pointing-interpolation-method {linear,gaussian}
                            Interpolation method to use for matching pointing and visibilities, default is linear
      --grid-size GRID_SIZE
                            Choose grid size for beam image in pixels, the default is to guess it from the data.
      --cell-size CELL_SIZE
                            Choose cell size for beam image in radians, the default is to guess it from the data.
      --grid-interpolation-mode GRID_INTERPOLATION_MODE
                            Choose interpolation mode for beam image (default is gaussian)
      --padding-factor PADDING_FACTOR
                            Padding factor applied to the beam image before the FFT to produce the apertures (default is 10)
      --zernike-n-order ZERNIKE_N_ORDER
                            Zernike N polynomial order to be fitted to the apertures (default is 4)
      -z, --use-zernike-phase-fitting
                            Use zernike phase fitting instead of regular perturbation phase fitting (ngVLA uses this even when option is not given).
      -c, --combine         Combine SPWs to improve SNR (all selected SPWs will be combined)
      --clip-type {sigma,none,absolute,relative,noise_threshold}
                            Choose the type of clipping algorithm to use in the apertures before fitting panels (default is sigma)
      --clip-level CLIP_LEVEL
                            Choose the level for the chosen clipping algorithm, panel help for more details (default is 3)
      --panel-model {mean,rigid,flexible,corotated_scipy,corotated_lst_sq,corotated_robust,xy_paraboloid,rotated_paraboloid,full_paraboloid_lst_sq}
                            Choose the panel model to use (default is flexible)
      --panel-margins PANEL_MARGINS
                            Define the fraction of the edge of the panels to be excluded from fitting (default is 0.05)
      -u SCREW_UNIT, --screw-unit SCREW_UNIT
                            Unit to present screw adjustments (default is mils)


Calibration stage
#################

With a reference antenna chosen it is now time to run the holography pipeline.
The first step of the pipeline is to check whether the data is an ASDM or an MS and if it is an ASDM if it needs to be imported into an MS. With an MS in hands the pipeline proceeds to fetching some metadata from it and then prints a summary of what it has found and which parameters it will use for calibration and further data reduction, e.g.:

.. code-block::

    CASA <2>: !holography-reduction-pipeline otf-ku.ms ea05

    #####################################################################################################################################
    ###  Welcome to the AstroHACK holography reduction pipeline                                                                       ###
    #####################################################################################################################################

    2026-10-07 16:46:46	INFO	msmetadata_cmpt.cc::open	Performing internal consistency checks on otf-ku.ms...
    2026-10-07 16:46:49	INFO	MSMetaData::_computeScanAndSubScanProperties 	Computing scan and subscan properties...

    Holography reduction parameters:
        filename                      => otf-ku.ms
        refant                        => ea05
        root_name                     => None
        spw                           => all
        antenna                       => all
        ncores                        => 4
        memory_per_core               => 10GB
        log_level                     => WARNING
        overwrite                     => False
        assume_yes                    => False
        reimport_asdm                 => False
        starting_stage                => calibration
        dpi                           => 300
        data_column                   => CORRECTED_DATA
        plot_pointing                 => False
        exclude_bad_antennas          => None
        quack_nchan                   => 4
        holography_field              => 3
        baseline_average_nearest      => 1
        pointing_interpolation_method => linear
        grid_size                     => None
        cell_size                     => None
        grid_interpolation_mode       => gaussian
        padding_factor                => 10
        zernike_n_order               => 4
        use_zernike_phase_fitting     => False
        combine                       => False
        clip_type                     => sigma
        clip_level                    => 3
        panel_model                   => flexible
        panel_margins                 => 0.05
        screw_unit                    => mils
        is_asdm                       => False
        msname                        => otf-ku.ms
        delay_cal_name                => otf-ku.dcal
        bandpass_cal_name             => otf-ku.bcal
        gain_cal_name                 => otf-ku.gcal
        point_name                    => otf-ku.point.zarr
        holog_name                    => otf-ku.holog.zarr
        image_name                    => otf-ku.image.zarr
        combine_name                  => otf-ku.combine.zarr
        panel_name                    => otf-ku.panel.zarr
        exports_name                  => otf-ku.exports
        report_name                   => otf-ku-report.html
        quacked_spw_selection         => 0~7:4~60
        quacked_base_band_0_selection => 0~3:4~60
        quacked_base_band_1_selection => 4~7:4~60
        delay_spwmap                  => [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(0), np.int64(0)]
        full_spwmap                   => [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9)]
        calibration_scans             => 2,5,26,47,68,89,92,113,134,155,176,179,200,221,242,263,266,287,304
        holography_scans              => 8,12,16,20,24,29,33,37,41,45,50,54,58,62,66,71,75,79,83,87,95,99,103,107,111,116,120,124,128,132,137,141,145,149,153,158,162,166,170,174,182,186,190,194,198,203,207,211,215,219,224,228,232,236,240,245,249,253,257,261,269,273,277,281,285,290,294,298,302
        parallel                      => True


    Proceed? <(Y)es/(N)o>:


The check before proceeding can be suppressed by adding the ``-y`` option to the call, e.g.:

.. code-block::

    CASA <3>: !holography-reduction-pipeline otf-ku.ms ea05 -y

The code will then proceed through the calibration steps:

#. Delay calibration with ``gaincal(gaintype="K")``, which is done separately for the two basebands.

#. Bandpass calibration with ``bandpass``.

#. Amplitude and Phase calibration with ``gaincal(calmode="AP")``.

#. Application of all the previously computed calibration tables with ``applycal``.

Holography processing
###################

After calibration the pipeline then proceeds to run AstroHACK's functions:

#. `extract_pointing <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_pointing/index.html>`_: Extract pointing data from the MS onto a ``.point.zarr`` file that is arranged in a convenient way for further processing.

#. `extract_holog <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/extract_holog/index.html>`_: Identify moving antennas from the pointing data, then extract visibilities from the ms for these antennas and finally match the pointing data to the visibilities.

#. `holog <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/holog/index.html>`_: Grid the visibility data onto beam images, then perform a zero padded 2D FFT of the beam images to obtain aperture images, these aperture images are then fitted with zernike polynomials to obtain a description of the macro-scale perturbations of the aperture and then finally a phase model is subtracted from the aperture to account from minor optical misalignments, such as reference pointing errors and being slightly out of focus.

#. Optionally `combine <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/combine/index.html>`_ can be run after holog (``-c``, ``--commbine`` option) to combine the different spectral windows in the aperture plane, this is usually done to increase SNR.

#. `panel <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/holog/index.html>`_: This is the last step of the holography data reduction. Here each pixel of the stokes I image of an aperture is attributed to a panel, and then a surface is fit to each panel to determine how much each screw adjuster needs to be moved in each panel to improve the surface accuracy of the antenna.


By default the AstroHACK stages are run in parallel, (ncores =4), this can be changed by explicitly giving a number of cores e.g. ``--ncores 5``. For a serial run, one should use ``--ncores 0`` or ``--ncores 1``. In case of failures or there is a desire to re-run the pipeline from a particular stage, the user can then use option ``--starting-stage``.
For more details on the holography processing stages there is the more detailed `VLA holography tutorial <https://astrohack.readthedocs.io/en/stable/tutorials/vla_holography_tutorial.html>`_.

Exports and Report stages
#########################

After the AstroHACK data files are created, the pipeline then proceeds to execute the exporting functions from the associated Python classes:

#. `AstrohackPointFile.plot_array_configuration <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/point_mds/index.html#plot_array_configuration>`_: Single plot displaying the array configuration at observation time.

#. `AstrohackImageFile.plot_beams <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/image_mds/index.html>`_: Plots of the gridded beam images.

#. `AstrohackImageFile.plot_zernike_model <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/image_mds/index.html>`_: plots of the Zernike models of the apertures and the associated residuals.

#. `AstrohackImageFile.export_phase_fit_results <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/image_mds/index.html>`_: Produce a small table of the phase fit results when the regular perturbations phase fit mechanism is used.

#. `AstrohackImageFile.export_zernike_fit_results <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/image_mds/index.html>`_: Produce a table of the fitted Zernike polynomial coefficients for each correlation.

#. `AstrohackPanelFile.plot_antennas <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/panel_mds/index.html>`_: Plots of the antenna apertures, amplitudes, masks, phase, surface deviation, estimated surface deviation corrections and estimated surface correction residuals.

#. `AstrohackPanelFile.export_screws <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/panel_mds/index.html>`_: Export a table with proposed screw adjustments as well as a map of the antenna surface with color coded screws to help identifying screws for which there are bigger corrections to be applied.

#. `AstrohackPanelFile.export_gain_tables <https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/io/panel_mds/index.html>`_: Create a small table containing estimated antenna gain losses for several wavelengths before and after screw adjustments.

After the production of these export products the pipeline then creates a standalone HTML report with all of them that can then be stored or shared without the need to carry any extra data, an example of such a report can be seen `here <../example-holography-ku-band-report.html>`_.