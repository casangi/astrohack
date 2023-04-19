import param
import scipy.constants
import panel as pn
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

from astrohack.dio import AstrohackDataFile

from panel.template.theme import DarkTheme

pn.extension('plotly', sizing_mode='stretch_width')

class ApertureViewer(param.Parameterized):
    antenna = param.ObjectSelector(label="Antenna")
    ddi = param.ObjectSelector(label="DDI")
    type = param.ObjectSelector(label="Type")

    
    def __init__(self, data_file, **kwargs):
        super().__init__(**kwargs)
        self.data_dict = data_file
        
        if self.data_dict.image.is_open() == False:
            self.data_dict.image.open()

        self.param.antenna.objects = list(self.data_dict.image.keys())
        self.antenna = list(self.data_dict.image.keys())[0]
        
        index = list(self.data_dict.image.keys())[0]
        
        self.param.ddi.objects = list(self.data_dict.image[index].keys())
        self.ddi = list(self.data_dict.image[index].keys())[0]

        self.param.type.objects =["Aperture", "Amplitude", "Angle"]
        self.type = "Aperture"

        self._plot_function = {
            'Aperture':self._make_aperture_plot,
            'Amplitude':self._make_amplitude_plot,
            'Angle':self._make_angle_plot
        }

        self.layout = pn.WidgetBox(
                pn.Row(
                    pn.Card(self.param, width=250, height=925), 
                    pn.Card(self._make_plot)
                )
            )

        
    @param.depends('antenna', watch=True)
    def _update_ddi(self):
        index = self.antenna
        self.param.ddi.objects = list(self.data_dict.image[index].keys())
        self.ddi = list(self.data_dict.image[index].keys())[0]

    @param.depends('antenna', 'ddi')
    def _make_aperture_plot(self):
        fig = Figure(figsize=(12, 12))

        wavelength = scipy.constants.speed_of_light/self.data_dict.image[self.antenna][self.ddi].chan.values[0]
        
        u = self.data_dict.image[self.antenna][self.ddi].u.values*wavelength
        v = self.data_dict.image[self.antenna][self.ddi].v.values*wavelength

        ax = fig.subplots()
        image = ax.imshow(
            self.data_dict.image[self.antenna][self.ddi].apply(np.abs).APERTURE.values.mean(axis=0)[0, 0, ...],
            extent=[u.min(), u.max(), v.min(), v.max()]
        )

        fig.colorbar(image, ax=ax)
        mpl_pane = pn.pane.Matplotlib(fig, dpi=144, tight=True)

        return mpl_pane
    
    @param.depends('antenna', 'ddi')
    def _make_amplitude_plot(self):        
        fig = Figure(figsize=(12, 12))
        
        wavelength = scipy.constants.speed_of_light/self.data_dict.image[self.antenna][self.ddi].chan.values[0]
        
        u = self.data_dict.image[self.antenna][self.ddi].u_prime.values*wavelength
        v = self.data_dict.image[self.antenna][self.ddi].v_prime.values*wavelength

        ax = fig.subplots()
        image = ax.imshow(
            self.data_dict.image[self.antenna][self.ddi].apply(np.abs).AMPLITUDE.values.mean(axis=0)[0, 0, ...],
            extent=[u.min(), u.max(), v.min(), v.max()]
        )
        
        fig.colorbar(image, ax=ax)
        mpl_pane = pn.pane.Matplotlib(fig, dpi=144, tight=True)

        return mpl_pane
    
    @param.depends('antenna', 'ddi')
    def _make_angle_plot(self):        
        fig = Figure(figsize=(12, 12))
        
        wavelength = scipy.constants.speed_of_light/self.data_dict.image[self.antenna][self.ddi].chan.values[0]
        
        u = self.data_dict.image[self.antenna][self.ddi].u_prime.values*wavelength
        v = self.data_dict.image[self.antenna][self.ddi].v_prime.values*wavelength

        ax = fig.subplots()
        image = ax.imshow(
            self.data_dict.image[self.antenna][self.ddi].apply(np.abs).ANGLE.values.mean(axis=0)[0, 0, ...],
            extent=[u.min(), u.max(), v.min(), v.max()]
        )

        fig.colorbar(image, ax=ax)

        mpl_pane = pn.pane.Matplotlib(fig, dpi=144, tight=True)

        return mpl_pane

    @param.depends('type')
    def _make_plot(self):
        return self._plot_function[self.type]


    def notebook(self):
        return self.layout
        
    def app(self, theme="light"):
        if theme=="dark":
            golden = pn.template.GoldenTemplate(title='', theme=DarkTheme)
        else:
            golden = pn.template.GoldenTemplate(
                title='', 
                header_background='black', 
                logo='https://public.nrao.edu/wp-content/themes/nrao/img/NRAO_logo_text.png'
            )
            
            golden.sidebar.append(self.param.antenna)
            golden.sidebar.append(self.param.ddi)
            
            golden.main.append(
                pn.Card(self._make_aperture_plot, name="aperture")
            )
            
            golden.main.append(
                pn.Card(self._make_amplitude_plot, name="amplitude")
            )   
            
            golden.main.append(
                pn.Card(self._make_angle_plot, name="angle")
            )

        return golden.show()
