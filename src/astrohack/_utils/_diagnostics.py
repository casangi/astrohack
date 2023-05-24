import numpy as np
    
import matplotlib.pyplot as plt

from astropy.time import Time

def _calc_index(n, m):
    if n >= m:
        return n % m
    else:
        return n

def _extract_indicies(l, m, squared_radius):
    indicies = []

    assert l.shape[0] == m.shape[0], "l, m must be same size."

    for i in range(l.shape[0]):
        squared_sum = np.power(l[i], 2) + np.power(m[i], 2)
        if squared_sum <= squared_radius:
            indicies.append(i)
            
    return np.array(indicies)

def _matplotlib_calibration_inspection_function(data, delta=0.01, pol='RR', width=1000, height=450):
  import matplotlib.pyplot as plt

  pixels = 1/plt.rcParams['figure.dpi']
  plt.rcParams['figure.figsize'] = [width*pixels, height*pixels]
    
  UNIX_CONVERSION = 3506716800
    
  radius = np.power(data.grid_parms['cell_size']*delta, 2)
  pol_index = np.squeeze(np.where(data.pol.values==pol))
    
  l = data.DIRECTIONAL_COSINES.values[..., 0] 
  m = data.DIRECTIONAL_COSINES.values[..., 1]
    
  assert l.shape[0] == m.shape[0], "l, m dimensions don't match!"
    
  indicies = _extract_indicies(
        l = l, 
        m = m, 
        squared_radius=radius
  )
    
  vis = data.isel(time=indicies).VIS
  times = Time(vis.time.data - UNIX_CONVERSION, format='unix').iso
    
  fig, axis = plt.subplots(2, 1)
    
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
    
    pol_index = np.squeeze(np.where(data.pol.values==pol))
    radius = np.power(data.grid_parms['cell_size']*delta, 2)
    
    l = data.DIRECTIONAL_COSINES.values[..., 0] 
    m = data.DIRECTIONAL_COSINES.values[..., 1]
    
    assert l.shape[0] == m.shape[0], "l, m dimensions don't match!"
    
    indicies = _extract_indicies(
        l = l, 
        m = m, 
        squared_radius=radius
    )
    
    vis = data.isel(time=indicies).VIS
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
        
        fig['layout']={
            'height':height,
            'width': width,
            'title': 'Calibration Check: polarization={p}'.format(p=data.pol.values[pol_index]),
            'paper_bgcolor':'#FFFFFF',
            'plot_bgcolor':'#FFFFFF',
            'font_color': '#323130',
            'yaxis':{
                'title':'Visibilities (real)',
                'linecolor':'#626567',
                'linewidth': 2,
                'zeroline':False,
                'mirror': True,
                'showline':True,
                'anchor': 'x', 
                'domain': [0.575, 1.0],
                #'showspikes': True,
                #'spikemode': 'across',
                #'spikesnap': 'cursor',
            },
            'yaxis2':{
                'title':'Visibilities (imag)',
                'linecolor':'#626567',
                'linewidth': 2,
                'zeroline':False,
                'mirror': True,
                'showline':True,
                'anchor': 'x2',
                'domain': [0.0, 0.425]
            },
            'xaxis':{
                'title':'Channel',
                'zeroline':False,
                'linecolor':' #626567',
                'linewidth': 2,
                'mirror': True,
                'showline':True,
                'anchor': 'y', 
                'domain': [0.0, 1.0],
                #'showspikes': True,
                #'spikemode': 'across',
                #'spikesnap': 'cursor',
            },
            'xaxis2':{                
                'title':'Channel',
                'zeroline':False,
                'linecolor':' #626567',
                'linewidth': 2,
                'mirror': True,
                'showline':True,
                'anchor': 'y2', 
                'domain': [0.0, 1.0]
            }
        }
    
    fig.show()

def _calibration_plot_chunk(param_dict):

    data = param_dict['data']
    delta  = param_dict['delta']
    data_type = param_dict['type']
    save_plot = param_dict['save']
    display = param_dict['display']
    width = param_dict['width']
    height = param_dict['height']
    out_folder = param_dict['out_folder']
    
    pixels = 1/plt.rcParams['figure.dpi']
    plt.rcParams['figure.figsize'] = [width*pixels, height*pixels]
    
    UNIX_CONVERSION = 3506716800
    
    radius = np.power(data.grid_parms['cell_size']*delta, 2)
    
    l = data.DIRECTIONAL_COSINES.values[..., 0] 
    m = data.DIRECTIONAL_COSINES.values[..., 1]
    
    assert l.shape[0] == m.shape[0], "l, m dimensions don't match!"
    
    indicies = _extract_indicies(
        l = l, 
        m = m, 
        squared_radius=radius
    )
    
    if data_type == "real":
        vis_dict = {
            "data": [
                data.isel(time=indicies).VIS.real,
                data.isel(time=indicies).VIS.imag
            ],
            "polarization": [0, 3],
            "label":[
                "REAL", 
                "IMAG"
            ]
        }
    else:
        vis_dict = {
            "data": [
                data.isel(time=indicies).apply(np.abs).VIS,
                data.isel(time=indicies).apply(np.angle).VIS
            ],
            "polarization": [0, 3],
            "label":[
                "AMP", 
                "PHASE"
            ]
        }
    
    times = np.unique(Time(vis_dict["data"][0].time.data - UNIX_CONVERSION, format='unix').iso)
    
    fig, axis = plt.subplots(4, 1, sharex=True)
    
    chan = np.arange(0, data.chan.data.shape[0])

    fig.suptitle(
        'Data Calibration Check: [dd_{ddi}, {map_id}, {ant_id}]'.format(ddi=data.ddi, map_id=data.holog_map_key, ant_id=data.antenna_name), 
        ha='left', 
        va='center', 
        x=0.04, 
        y=0.5, 
        rotation=90
    )
    
    length = times.shape[0]
    
    for i, vis in enumerate(vis_dict["data"]):
        for j, pol in enumerate(vis_dict["polarization"]):
            for time in range(length):
                k = 2*i + j
                axis[k].plot(chan, vis[time, :, pol], marker='o', label=times[time])
                axis[k].set_ylabel("Vis ({component}; {pol})".format(component=vis_dict["label"][i], pol=data.pol.values[pol]))
    
    
    axis[3].set_xlabel("Channel")
    axis[0].legend(
        bbox_to_anchor=(0., 1.02, 1., .102), 
        loc='lower left',
        ncols=4, 
        mode="expand", 
        borderaxespad=0.
    )
    
    if save_plot: fig.savefig("{out_folder}/ddi_{ddi}_{map_id}_{ant_id}.png".format(out_folder=out_folder, ddi=data.ddi, map_id=data.holog_map_key, ant_id=data.antenna_name))

    if not display: plt.close(fig)