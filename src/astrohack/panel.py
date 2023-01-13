from astrohack._classes.antenna_surface import AntennaSurface
import os

def _panel_chunk(basename, amp, dev, telescope, cutoff=0.21, pkind=None, savemask=False,
                 saveplots=True, exportcorrected=False, unit='miliinches'):
    surface = AntennaSurface(amp, dev, telescope, cutoff, pkind)
    surface.compile_panel_points()
    surface.fit_surface()
    surface.correct_surface()

    os.makedirs(name=basename, exist_ok=True)
    basename += '/'

    surface.export_screw_adjustments(basename+'screws.txt', unit=unit)

    if savemask:
        surface.plot_surface(filename=basename+'mask.png', mask=True, screws=True)
    if saveplots:
        surface.plot_surface(filename=basename+'surface.png')
    if exportcorrected:
        surface.export_corrected(basename+'corrected.fits')

    ingains, ougains = surface.gains()
    inrms, ourms = surface.get_rms()
    report = open(basename+'report.txt', 'w')

    report.write("Gains before correction: Real: {0:7.3} dB, Theoretical: {1:7.3} dB\n".format(*ingains))
    report.write("RMS before correction: {0:7.3} mm\n".format(inrms))
    report.write('\n')
    report.write("Gains after correction: Real: {0:7.3} dB, Theoretical: {1:7.3} dB\n".format(*ougains))
    report.write("RMS after correction: {0:7.3} mm\n".format(ourms))
    report.close()
    return
