import matplotlib.pyplot as plt
import numpy as np
from astrohack._utils._constants import fontsize, markersize
from astrohack._utils._plot_commons import _create_figure_and_axes, _close_figure
from astropy.time import Time


def _calc_index(n, m):
    if n >= m:
        return n % m
    else:
        return n


def _extract_indices(l, m, squared_radius):
    indices = []

    assert l.shape[0] == m.shape[0], "l, m must be same size."

    for i in range(l.shape[0]):
        squared_sum = np.power(l[i], 2) + np.power(m[i], 2)
        if squared_sum <= squared_radius:
            indices.append(i)

    return np.array(indices)


def _matplotlib_calibration_inspection_function(data, delta=0.01, pol='RR', width=1000, height=450):
    pixels = 1 / plt.rcParams['figure.dpi']
    UNIX_CONVERSION = 3506716800

    radius = np.power(data.grid_parms['cell_size'] * delta, 2)
    pol_index = np.squeeze(np.where(data.pol.values == pol))

    l = data.DIRECTIONAL_COSINES.values[..., 0]
    m = data.DIRECTIONAL_COSINES.values[..., 1]

    assert l.shape[0] == m.shape[0], "l, m dimensions don't match!"

    indices = _extract_indices(
        l=l,
        m=m,
        squared_radius=radius
    )

    vis = data.isel(time=indices).VIS
    times = Time(vis.time.data - UNIX_CONVERSION, format='unix').iso

    fig, axis = _create_figure_and_axes([width * pixels, height * pixels], [2, 1])

    chan = np.arange(0, data.chan.data.shape[0])

    for i in range(times.shape[0]):
        axis[0].plot(chan, vis[i, :, pol_index].real, marker='o', label=times[i])
        axis[0].set_title('Calibration Check: polarization={p}'.format(p=data.pol.values[pol_index]))
        axis[0].set_ylabel("Visibilities (real)")
        axis[0].set_xlabel("Channel")

        axis[0].legend()

        axis[1].plot(chan, vis[i, :, pol_index].imag, marker='o', label=times[i])
        axis[1].set_ylabel("Visibilities (imag)")
        axis[1].set_xlabel("Channel")

        axis[1].legend()


def _plotly_calibration_inspection_function(data, delta=0.01, pol='RR', width=1000, height=450):
    import plotly.graph_objects as go
    import plotly.express as px

    from plotly.subplots import make_subplots

    UNIX_CONVERSION = 3506716800

    pol_index = np.squeeze(np.where(data.pol.values == pol))
    radius = np.power(data.grid_parms['cell_size'] * delta, 2)

    l = data.DIRECTIONAL_COSINES.values[..., 0]
    m = data.DIRECTIONAL_COSINES.values[..., 1]

    assert l.shape[0] == m.shape[0], "l, m dimensions don't match!"

    indices = _extract_indices(
        l=l,
        m=m,
        squared_radius=radius
    )

    vis = data.isel(time=indices).VIS
    times = Time(vis.time.data - UNIX_CONVERSION, format='unix').iso

    chan = np.arange(0, data.chan.data.shape[0])
    fig = make_subplots(rows=2, cols=1, start_cell="top-left")

    for i in range(times.shape[0]):
        index = _calc_index(i, 10)
        fig.add_trace(
            go.Scatter(
                x=chan,
                y=vis[i, :, pol_index].real,
                marker={
                    'color': px.colors.qualitative.D3[index],
                    'line': {
                        'width': 3,
                        'color': px.colors.qualitative.D3[index]
                    }
                },
                mode='lines+markers',
                name=times[i],
                legendgroup=times[i],
                meta=[times[i]],
                hovertemplate="<br>".join([
                    '<b>time: %{meta[0]}</b><extra></extra>',
                    'chan:%{x}',
                    'vis: %{y}'
                ])
            ), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=chan,
                y=vis[i, :, pol_index].imag,
                marker={
                    'color': px.colors.qualitative.D3[index],
                    'line': {
                        'width': 3,
                        'color': px.colors.qualitative.D3[index]
                    }
                },
                mode='lines+markers',
                name=times[i],
                legendgroup=times[i],
                showlegend=False,
                meta=[times[i]],
                hovertemplate="<br>".join([
                    '<b>time: %{meta[0]}</b><extra></extra>',
                    'chan:%{x}',
                    'vis: %{y}'
                ])
            ), row=2, col=1
        )

        fig['layout'] = {
            'height': height,
            'width': width,
            'title': 'Calibration Check: polarization={p}'.format(p=data.pol.values[pol_index]),
            'paper_bgcolor': '#FFFFFF',
            'plot_bgcolor': '#FFFFFF',
            'font_color': '#323130',
            'yaxis': {
                'title': 'Visibilities (real)',
                'linecolor': '#626567',
                'linewidth': 2,
                'zeroline': False,
                'mirror': True,
                'showline': True,
                'anchor': 'x',
                'domain': [0.575, 1.0],
                # 'showspikes': True,
                # 'spikemode': 'across',
                # 'spikesnap': 'cursor',
            },
            'yaxis2': {
                'title': 'Visibilities (imag)',
                'linecolor': '#626567',
                'linewidth': 2,
                'zeroline': False,
                'mirror': True,
                'showline': True,
                'anchor': 'x2',
                'domain': [0.0, 0.425]
            },
            'xaxis': {
                'title': 'Channel',
                'zeroline': False,
                'linecolor': ' #626567',
                'linewidth': 2,
                'mirror': True,
                'showline': True,
                'anchor': 'y',
                'domain': [0.0, 1.0],
                # 'showspikes': True,
                # 'spikemode': 'across',
                # 'spikesnap': 'cursor',
            },
            'xaxis2': {
                'title': 'Channel',
                'zeroline': False,
                'linecolor': ' #626567',
                'linewidth': 2,
                'mirror': True,
                'showline': True,
                'anchor': 'y2',
                'domain': [0.0, 1.0]
            }
        }

    fig.show()


def _calibration_plot_chunk(param_dict):
    data = param_dict['xds_data']
    delta = param_dict['delta']
    complex_split = param_dict['complex_split']
    display = param_dict['display']
    figuresize = param_dict['figure_size']
    destination = param_dict['destination']
    dpi = param_dict['dpi']
    thisfont = 1.2 * fontsize

    UNIX_CONVERSION = 3506716800

    radius = np.power(data.grid_parms['cell_size'] * delta, 2)

    l_axis = data.DIRECTIONAL_COSINES.values[..., 0]
    m_axis = data.DIRECTIONAL_COSINES.values[..., 1]

    assert l_axis.shape[0] == m_axis.shape[0], "l, m dimensions don't match!"

    indices = _extract_indices(l=l_axis, m=m_axis, squared_radius=radius)

    if complex_split == "cartesian":
        vis_dict = {
            "data": [
                data.isel(time=indices).VIS.real,
                data.isel(time=indices).VIS.imag
            ],
            "polarization": [0, 3],
            "label": ["REAL", "IMAG"]
        }
    else:
        vis_dict = {
            "data": [
                data.isel(time=indices).apply(np.abs).VIS,
                data.isel(time=indices).apply(np.angle).VIS
            ],
            "polarization": [0, 3],
            "label": ["AMP", "PHASE"]
        }

    times = np.unique(Time(vis_dict["data"][0].time.data - UNIX_CONVERSION, format='unix').iso)

    fig, axis = _create_figure_and_axes(figuresize, [4, 1], sharex=True)

    chan = np.arange(0, data.chan.data.shape[0])

    length = times.shape[0]

    for i, vis in enumerate(vis_dict["data"]):
        for j, pol in enumerate(vis_dict["polarization"]):
            for time in range(length):
                k = 2 * i + j
                axis[k].plot(chan, vis[time, :, pol], marker='o', label=times[time], markersize=markersize)
                axis[k].set_ylabel(f'Vis ({vis_dict["label"][i]}; {data.pol.values[pol]})', fontsize=thisfont)
                axis[k].tick_params(axis='both', which='major', labelsize=thisfont)

    axis[3].set_xlabel("Channel", fontsize=thisfont)
    axis[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncols=4, mode="expand", borderaxespad=0.,
                   fontsize=fontsize)

    fig.suptitle(
        f'Data Calibration Check: [{param_dict["this_ddi"]}, {param_dict["this_map"]}, {param_dict["this_ant"]}]',
        ha='center', va='center', x=0.5, y=0.95, rotation=0, fontsize=1.5 * thisfont)
    plotfile = f'{destination}/holog_diagnostics_{param_dict["this_map"]}_{param_dict["this_ant"]}_' \
               f'{param_dict["this_ddi"]}.png'
    _close_figure(fig, None, plotfile, dpi, display, tight_layout=False)
