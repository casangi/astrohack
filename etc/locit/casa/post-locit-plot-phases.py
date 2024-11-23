
def _get_phases(msnam, ant, ref):
    import numpy as np
    from casacore import tables as ctables
    antsel = 4
    refsel = 25
    
    table_obj = ctables.table(msnam, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    
    time = table_obj.getcol("TIME")
    corr = table_obj.getcol("CORRECTED_DATA")
    orig = table_obj.getcol("DATA")
    ant1 = table_obj.getcol("ANTENNA1")
    ant2 = table_obj.getcol("ANTENNA2")
    flag = np.invert(table_obj.getcol("FLAG"))

    refsel = ant2 == refsel
    ant1 = ant1[refsel]
    time = time[refsel]
    corr = corr[refsel]
    orig = orig[refsel]
    flag = flag[refsel]

    antsel = ant1 == antsel
    time = time[antsel]
    corr = corr[antsel]
    orig = orig[antsel]
    flag = flag[antsel]

    time = (time-time[0])/3600.

    return time, _centered_phases(orig), _centered_phases(corr), flag

def _plot_phases(msnam, ant, ref):
    time, raw_pha, cor_pha, flag = _get_phases(msnam, ant, ref)
    
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8,5))
    corrs = ['RR', 'RL',  'LR', 'LL']

    for icorr in range(4):
        if icorr > 1:
            ix = 1
            iy = icorr - 2
        else:
            ix = 0
            iy = icorr
        _plot_single_phase(axes[ix, iy], time, raw_pha[:,:,icorr],
                           cor_pha[:,:,icorr], flag[:,:,icorr], corrs[icorr])

    suptitle = f'Phases for {ant}&{ref}'
    figname = f'phase_{ant}.png'
    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.savefig(figname, dpi=300)
        
def _plot_single_phase(ax, time, raw, cor, flag, corr):
    import numpy as np
    time = time[flag[:,0]]
    raw = raw[flag]
    cor = cor[flag]
    raw_rms = np.std(raw)
    cor_rms = np.std(cor)
    mksz = 0.2
    ax.plot(time, raw, label='Raw', color='black', ls='', marker='o',
            markersize=mksz)
    ax.plot(time, cor, label='Corrected', color='red', ls='', marker='o',
            markersize=mksz)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Phase [deg]')
    ax.set_ylim([-180, 180])
    ax.set_title(f'{corr} Raw = {raw_rms:.1f}, Corrected = {cor_rms:.1f}')
    ax.legend()

def _centered_phases(cplx):
    import numpy as np
    phase = np.angle(cplx)
    avg = np.mean(phase)
    centered = phase - avg
    wrapped = (centered + np.pi) % (2*np.pi) - np.pi
    return wrapped*180/np.pi
    

msname = 'redux-scan-63-avg.ms'
ref_antenna = 'ea28'
antenna = 'ea06'

_plot_phases(msname, antenna, ref_antenna)
