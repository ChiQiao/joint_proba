import pickle
import time
from tkinter import Tk, filedialog

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
import traitlets
from IPython.display import display
import functools

from tail_extrap import multivariate

debug_view = widgets.Output(layout={'border': '1px solid black'})
layout_section = {'margin': '12px 2px 12px 2px'}

class Interactive:
    def __init__(self, mv):
        self.mv = mv

        self.uni_fit_button = widgets.Button() # In tab1

        self.save_button = SaveFileButton(
            description='Save session as...',
            file_type=[("pickle archive", ".pkl")],
        )

        self.tab0 = Tab_config(self.mv, self.uni_fit_button)
        self.tab1 = Tab_univariate(self.mv, self.uni_fit_button)
        self.tab2 = Tab_contour(self.mv)
        # self.tab3 = Tab_export(self.mv)
        
        tab = widgets.Tab(children=[
            self.tab0.tab, self.tab1.tab, self.tab2.tab
            ])
        tab.set_title(0, 'Data config')
        tab.set_title(1, 'Univariate fitting')
        tab.set_title(2, 'Contour construction')
        # tab.set_title(3, 'Export result')

        self.save_button.on_click(self.save_session)
        self.tab0.update_button.on_click(
            functools.partial(self.tab0_update_clicked, mv=mv))
        self.tab0.confirm_box.ok_button.on_click(
            functools.partial(self.tab0_update_confirmed, mv=self.mv))

        display(
            widgets.VBox(children=[
                self.save_button,
                tab,
                debug_view
            ])
        )

    def save_session(self, change):
        # local function fitting_func in multivariate._CondY.fit cannot be
        #   pickled. As an workaround, mv.condY_cont_dists_bulk is removed and
        #   refit when loaded next time
        if hasattr(self.mv, 'condY_cont_dists_bulk'):
            delattr(self.mv, 'condY_cont_dists_bulk')
        with open(self.save_button.file_name + '.pkl', 'wb') as f:
            pickle.dump(self.mv, f)

    def tab0_update_clicked(self, change, mv):
        if hasattr(mv, 'x_dist'):
            # Fitting result exists, confirm cleanup
            self.tab0.confirm_box.show()
            self.tab0.update_button.disabled = True
        else:
            # No fitting result exists
            self.tab0_update_confirmed(change=None, mv=mv)

    def tab0_update_confirmed(self, change, mv):
        # Clean up fitting results
        mv.condY_x = str_to_condY(self.tab0.condY_text.value)
        for attr in ['x_dist', 'y_dist', 'condY_disc_dists']:
            if hasattr(mv, attr):
                delattr(mv, attr)
        mv.ct = {}

        # Update tab0 display
        self.tab0.confirm_box.hide()
        self.tab0.update_button.disabled = False
        self.tab0.refresh_plot(mv)
        button_visual(self.tab0.update_button)

        # Reset tab1 display
        self.tab1.uni_fit_button.description = 'Start Fitting' 
        set_widget_visibility(self.tab1.hide_list, 'hidden')
        
        # Update tab2 display
        # TODO
    
    @classmethod
    def from_archive(cls, archive_path):
        with open(archive_path, 'rb') as f:
            mv = pickle.load(f)
        # Re-fit mv.condY_cont_dists_bulk as it was removed when pickling
        mv.condY_cont_dists_bulk = mv._fit_condY_cont(mv.condY_para_bulk_df)
        return cls(mv)

    @classmethod
    def from_df(cls, df, col_x=0, col_y=1):
        mv = multivariate.Multivariate(df, col_x=col_x, col_y=col_y)
        # Initialize session setting
        mv.ss = {
            # 'condY_x': mv.condY_x,
            'x_dist': {
                'maxima_extract': 'Annual Maxima',
                'maxima_fit': 'Gumbel Chart',
                'bulk_fit': 'Empirical',
                'outlier_detect': 'None',
            },
            'y_dist': {
                'maxima_extract': 'Annual Maxima',
                'maxima_fit': 'Gumbel Chart',
                'bulk_fit': 'Empirical',
                'outlier_detect': 'None',
            },
            'condY_disc_dists': {
                'maxima_extract': 'Annual Maxima',
                'maxima_fit': 'Gumbel Chart',
                'bulk_fit': 'Empirical',
                'outlier_detect': 'None',
            },
        }
        # Initialize contour results
        mv.ct = {}

        return cls(mv)


class Tab_config:
    def __init__(self, mv, uni_fit_button):

        # Update button

        self.update_button = widgets.Button(
            description='Update',
            disabled=False,
            tooltip='Save settings and update figure',
        )
        self.confirm_box = ConfirmDialog(
            text='Update CondY_X will erase all the fitting results. Continue?'
        )
        self.update_section = widgets.VBox(
            children=[self.update_button, self.confirm_box.box],
        )

        self.confirm_box.hide()
        self.confirm_box.cancel_button.on_click(self.cancel_clicked)

        # CondY_X section

        self.condY_label = widgets.Label(
            value='$x$ for evaluating $f(y|x)$: ',
        )
        self.condY_text = widgets.Text(
            value=condY_to_str(mv.condY_x),
            placeholder='start : interval : end',
            layout=widgets.Layout(width='40%'),
        )
        self.condY_section = widgets.HBox(
            children=[self.condY_label, self.condY_text],
            layout = layout_section,
        )

        # Diagnostic plot

        layout_disp = {'height': '400px'}
        layout_disp.update(layout_section)
        self.data_display = widgets.Output(layout=layout_disp)

        self.tab = widgets.VBox(children=[
            self.update_section, self.condY_section, self.data_display,
        ])

        self.refresh_plot(mv)

    def cancel_clicked(self, change):
        self.confirm_box.hide()
        self.update_button.disabled = False

    def refresh_plot(self, mv):
        '''Re-generate the plot in data_display using mv'''
        self.data_display.clear_output(wait=True)
        with self.data_display:
            plt.figure(dpi=100)
            plt.plot(
                mv.x_data, mv.y_data, 'o', 
                markersize=3, alpha=0.2, color=[0.5, 0.5, 0.5])
            ylm = plt.ylim()
            plt.plot(
                np.vstack([mv.condY_x, mv.condY_x]), 
                np.matlib.repmat(
                    np.array(ylm).reshape(2, 1), 1, len(mv.condY_x)),
                '--', color=[1, 0.5, 0.5])
            plt.ylim(ylm)
            plt.xlabel(mv.x_name)
            plt.ylabel(mv.y_name)        
            plt.grid(True)
            plt.legend(['Raw data', 'CondY_X'], loc='upper left')
            plt.show()


class Tab_univariate:
    def __init__(self, mv, uni_fit_button):

        # Fitting section

        self.uni_fit_button = uni_fit_button
        self.progress_bar = widgets.IntProgress(
            min=0, max=5, 
            layout=widgets.Layout(width='10%', visibility='hidden'))
        self.progress_label = widgets.Label(
            layout=widgets.Layout(visibility='hidden')
        )

        self.fit_section = widgets.HBox(
            children=[self.uni_fit_button, self.progress_bar, self.progress_label],
        )
        
        self.uni_fit_button.on_click(
            functools.partial(self.update, mv=mv))

        # Config section

        self.dist_dropdown = widgets.Dropdown(
            options=[
                ('Marginal X', 'x_dist'), 
                ('Marginal Y', 'y_dist'),
                ('Conditional Y', 'condY_disc_dists')], 
            stylestyle={'description_width': 'initial'},
            layout=widgets.Layout(width='120px')
        )
        self.condY_slider = widgets.SelectionSlider(
            options=[None],
            description=' for $x$ = ',
            continuous_update=False,
            readout=True,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%', visibility='hidden')
        )
        self.condY_prev = widgets.Button(
            description='\u25C0', 
            tooltip='Select conditional Y for the previous value of x',
            layout=widgets.Layout(width='50px', visibility='hidden')
        )
        self.condY_next = widgets.Button(
            description='\u25B6', 
            tooltip='Select conditional Y for the previous value of x',
            layout=widgets.Layout(width='50px', visibility='hidden'),
        )
        
        self.maxima_extract_label = widgets.Label(
            value='Maxima extraction',
            layout=widgets.Layout(width='25%'),
        )
        self.maxima_extract_dropdown = widgets.Dropdown(
            options=['Annual Maxima'],
            value=mv.ss[self.dist_dropdown.value]['maxima_extract'],
            layout=widgets.Layout(width='25%'),
        )

        self.maxima_fit_label = widgets.Label(
            value='Maxima fitting',
            layout=widgets.Layout(width='25%'),
        )
        self.maxima_fit_dropdown = widgets.Dropdown(
            options=['Gumbel Chart'],
            value=mv.ss[self.dist_dropdown.value]['maxima_fit'],
            layout=widgets.Layout(width='25%'),
        )

        self.bulk_fit_label = widgets.Label(
            value='Bulk fitting',
            layout=widgets.Layout(width='25%'),
        )
        self.bulk_fit_dropdown = widgets.Dropdown(
            options=['Empirical', 'Parametric'],
            value=mv.ss[self.dist_dropdown.value]['bulk_fit'],
            layout=widgets.Layout(width='25%'),
        )

        self.outlier_detect_label = widgets.Label(
            value='Outlier detection',
            layout=widgets.Layout(width='25%'),
        )
        self.outlier_detect_dropdown = widgets.Dropdown(
            options=['None', 'RANSAC Regression', 'Huber Regression'],
            value=mv.ss[self.dist_dropdown.value]['outlier_detect'],
            layout=widgets.Layout(width='25%'),
        )

        self.config_section = widgets.VBox(
            children=[
                widgets.HBox(
                    children=[
                        self.dist_dropdown, self.condY_slider,
                        self.condY_prev, self.condY_next
                    ],
                    layout={'margin': '2px 2px 10px 2px'}
                ), 
                widgets.HBox(children=[
                    self.maxima_extract_label, self.maxima_fit_label, 
                    self.bulk_fit_label, self.outlier_detect_label
                ]), 
                widgets.HBox(children=[
                    self.maxima_extract_dropdown, self.maxima_fit_dropdown, 
                    self.bulk_fit_dropdown, self.outlier_detect_dropdown
                ]), 
            ],
            layout=layout_section,
        )

        self.update_condY_slider(mv)
        self.dist_dropdown.observe(
            functools.partial(self.refresh_plot, mv=mv), names='value')
        self.condY_slider.observe(
            functools.partial(self.refresh_plot, mv=mv), names='value')
        self.condY_prev.on_click(self.condY_slider_prev)
        self.condY_next.on_click(self.condY_slider_next)

        # Diagnostic plot

        layout_disp = {'height': '450px'}
        layout_disp.update(layout_section)
        self.data_display = widgets.Output(layout=layout_disp)
        
        self.tab = widgets.VBox(children=[
            self.fit_section, 
            self.config_section,
            self.data_display,
        ])

        self.hide_list = [self.config_section, self.data_display,
            self.condY_next, self.condY_prev, self.condY_slider]
        
        if not hasattr(mv, 'x_dist'):
            self.uni_fit_button.description = 'Start Fitting'
            set_widget_visibility(self.hide_list, 'hidden')
        else:
            self.uni_fit_button.description = 'Update'
            self.refresh_plot(change=None, mv=mv)
    
    def update_condY_slider(self, mv):
        '''Update the option for condY_slider'''
        condY_slider_dict = {f'{condY_x:.1f}': idx 
            for idx, condY_x in enumerate(mv.condY_x)}
        self.condY_slider.options = condY_slider_dict

    def condY_slider_prev(self, change):
        self.condY_slider.value = max([0, self.condY_slider.value - 1])

    def condY_slider_next(self, change):
        self.condY_slider.value = min(
            [len(self.condY_slider.options) - 1, self.condY_slider.value + 1])
    
    def fit_all(self,mv):
        ''' Fit each univariate distribution '''
        self.data_display.clear_output()
        self.progress_bar.layout.visibility = 'visible'
        self.progress_label.layout.visibility = 'visible'
        self.progress_bar.value = 0
        
        self.progress_label.value = 'Fitting marginal X'
        mv._fit_marginalX(**mv.ss['x_dist'])
        self.progress_bar.value += 1

        self.progress_label.value = 'Fitting marginal Y'
        mv._fit_marginalY(**mv.ss['y_dist'])
        self.progress_bar.value += 1

        self.progress_label.value = 'Fitting discrete conditional Y'
        mv._fit_condY_disc(**mv.ss['condY_disc_dists'])
        self.progress_bar.value += 1

        self.progress_label.value = 'Fitting median of conditional Y'
        mv._get_condY_median()
        self.progress_bar.value += 1

        self.progress_label.value = 'Fitting continuous conditional Y using bulk'
        df = mv._get_condY_para_bulk()
        mv.condY_cont_dists_bulk = mv._fit_condY_cont(df)
        mv.condY_para_bulk_df = df # Save df as condY_cont_dists_bulk will be removed
        self.progress_bar.value += 1

        self.update_condY_slider(mv)
        set_widget_visibility(self.hide_list, 'visible')
        self.progress_bar.layout.visibility = 'hidden'
        self.progress_label.layout.visibility = 'hidden'
        self.uni_fit_button.description = 'Update'
        self.refresh_plot(change=None, mv=mv)
        
    def fit_single(self, mv):
        '''Fit a specific univariate distribution defined by dist_dropdown'''
        self.data_display.clear_output()

        # Record current setting
        mv.ss[self.dist_dropdown.value]['maxima_extract'] = \
            self.maxima_extract_dropdown.value
        mv.ss[self.dist_dropdown.value]['maxima_fit'] = \
            self.maxima_fit_dropdown.value
        mv.ss[self.dist_dropdown.value]['bulk_fit'] = \
            self.bulk_fit_dropdown.value
        mv.ss[self.dist_dropdown.value]['outlier_detect'] = \
            self.outlier_detect_dropdown.value
        
        if self.dist_dropdown.value == 'x_dist':
            mv._fit_marginalX(**mv.ss[self.dist_dropdown.value])
        elif self.dist_dropdown.value == 'y_dist':
            mv._fit_marginalY(**mv.ss[self.dist_dropdown.value])
        else:
            mv._fit_condY_disc(**mv.ss[self.dist_dropdown.value])
        self.refresh_plot(change=None, mv=mv)

    def refresh_plot(self, change, mv):
        '''Save fitting config and regenerate diagnostic plot'''
        self.maxima_extract_dropdown.value = \
            mv.ss[self.dist_dropdown.value]['maxima_extract']
        self.maxima_fit_dropdown.value = \
            mv.ss[self.dist_dropdown.value]['maxima_fit']
        self.bulk_fit_dropdown.value = \
            mv.ss[self.dist_dropdown.value]['bulk_fit']
        self.outlier_detect_dropdown.value = \
            mv.ss[self.dist_dropdown.value]['outlier_detect']
        
        # Update condY_slider and data_display
        if self.dist_dropdown.value == 'condY_disc_dists':
            self.condY_slider.layout.visibility = 'visible'
            self.condY_prev.layout.visibility = 'visible'
            self.condY_next.layout.visibility = 'visible'
            dist = getattr(mv, self.dist_dropdown.value)[self.condY_slider.value]
        else:
            self.condY_slider.layout.visibility = 'hidden'
            self.condY_prev.layout.visibility = 'hidden'
            self.condY_next.layout.visibility = 'hidden'
            dist = getattr(mv, self.dist_dropdown.value)
        self.data_display.clear_output(wait=True)
        with self.data_display:
            display(dist.diag_fig)
            
    def update(self, change, mv):
        '''Operation for the uni_fit_button'''
        if self.uni_fit_button.description == 'Start Fitting': 
            self.fit_all(mv)
        else: 
            self.fit_single(mv)
        button_visual(self.uni_fit_button)
    

class Tab_contour:
    def __init__(self, mv):

        # Fitting status

        self.fit_button = widgets.Button(
            description='Start Fitting',
            tooltip='Fit contour for the current MRP'
        )
        self.progress_bar = widgets.IntProgress(
            min=0, max=5, 
            layout=widgets.Layout(width='10%', visibility='hidden'))
        self.progress_label = widgets.Label(
            layout=widgets.Layout(visibility='hidden')
        )
        self.confirm_box = ConfirmDialog(
            text='MRP exists, overwrite?'
        )
        self.confirm_box.hide()

        self.fit_section = widgets.VBox(
            children=[
                widgets.HBox(children=[
                    self.fit_button, self.progress_bar, self.progress_label]), 
                self.confirm_box.box
            ],
        )

        self.fit_button.on_click(functools.partial(self.fit_clicked, mv=mv))
        self.confirm_box.ok_button.on_click(functools.partial(self.fit_confirmed, mv=mv))
        self.confirm_box.cancel_button.on_click(self.cancel_clicked)
        
        # MRP selection

        self.mrp_from_new = widgets.Checkbox(
            description='Create new MRP of: ',
            value=True,
            indent=False,
            layout={'width': 'max-content'},
        )
        self.mrp_from_exist = widgets.Checkbox(
            description='Overwirte existing MRP of: ',
            value=False,
            indent=False,
            layout={'width': 'max-content'},
        )
        self.mrp_new = widgets.IntText(
            value=1,
            layout={'width': '100px'},
        )
        self.mrp_exist_select = widgets.Dropdown(
            options=list(mv.ct.keys()),
            layout={'width': '100px'},
        )
        
        self.mrp_section = widgets.HBox(
            children=[
                widgets.VBox(
                    children=[self.mrp_from_new, self.mrp_from_exist],
                ),
                widgets.VBox(children=[self.mrp_new, self.mrp_exist_select],
                )
            ],
            layout=layout_section,
        )
        
        if not self.mrp_exist_select.options: 
            self.mrp_from_exist.disabled = True
        self.mrp_from_new.observe(self.update_mrp_from_exist, names='value')
        self.mrp_from_exist.observe(self.update_mrp_from_new, names='value')
        self.mrp_exist_select.observe(self.update_diag, names='value')

        # Contour distribution selection

        self.contour_dropdown = widgets.Dropdown(
            options=['Lower contour', 'Upper contour'],
            layout=widgets.Layout(width='120px'),
        )
        self.select_button = widgets.Button(
            description='Select',
            tooltip='Use the current distribution for the contour',
            layout=widgets.Layout(margin='2px 10px 2px 10px', width='100px'),
        )
        self.dist_slider = widgets.SelectionSlider(
            description='using distribution: ',
            options=['None'],
            continuous_update=False,
            readout=True,
            layout=widgets.Layout(width='40%'),
            style={'description_width': 'initial'},
        )
        self.dist_prev = widgets.Button(
            description='\u25C0', 
            tooltip='Show results of the next distribution',
            layout=widgets.Layout(width='50px'),
        )
        self.dist_next = widgets.Button(
            description='\u25B6', 
            tooltip='Show results of the previous distribution',
            layout=widgets.Layout(width='50px'),
        )
        self.dist_err = widgets.Label()
        
        self.dist_section = widgets.VBox(
            children=[
                widgets.HBox(children=[
                    self.contour_dropdown, self.dist_slider, 
                    self.dist_prev, self.dist_next, self.select_button
                ]), 
                widgets.HBox(children=[
                    self.dist_err
                ]), 
            ],
            layout=layout_section,
        )

        self.contour_dropdown.observe(
            functools.partial(self.update_diag, mv=mv, mrp=self.get_mrp()), names='value')
        self.dist_slider.observe(
            functools.partial(self.update_diag, mv=mv, mrp=self.get_mrp()), names='value')
        self.dist_prev.on_click(self.dist_slider_prev)
        self.dist_next.on_click(self.dist_slider_next)
        self.select_button.on_click(
            functools.partial(self.update_selection, mv=mv))

        # Diagnostic plots

        layout_plot = {'width': '33%', 'height': '300px'}
        self.repara_plot = widgets.Output(layout=layout_plot)
        self.para_plot = widgets.Output(layout=layout_plot)
        self.contour_plot = widgets.Output(layout=layout_plot)

        self.plot_section = widgets.HBox(
            children=[self.repara_plot, self.para_plot, self.contour_plot],
        )

        self.tab = widgets.VBox(children=[
            self.fit_section, self.mrp_section, 
            self.dist_section, self.plot_section,
        ])

        self.hide_list = list(self.dist_section.children) + \
            list(self.plot_section.children)
        set_widget_visibility(self.hide_list, 'hidden')

    def get_mrp(self) -> int:
        if self.mrp_from_new.value:
            return self.mrp_new.value
        else:
            return self.mrp_exist_select.value

    def cancel_clicked(self, change):
        self.confirm_box.hide()
        self.fit_button.disabled = False

    def fit_clicked(self, change, mv):
        mrp = self.get_mrp()
        if self.mrp_from_new and mrp in self.mrp_exist_select.options:
            # New mrp is in the existing mrp list
            self.confirm_box.show()
            self.fit_button.disabled = True
        else:
            self.fit_confirmed(change=None, mv=mv)

    def fit_confirmed(self, change, mv):
        self.confirm_box.hide()
        self.fit_button.disabled = False
        set_widget_visibility(self.hide_list, 'hidden')
        self.progress_bar.layout.visibility = 'visible'
        self.progress_label.layout.visibility = 'visible'
        self.progress_bar.value = 0

        mrp = self.get_mrp()
        ct = {'mrp': mrp} # Initialize contour result
        self.progress_label.value = 'Calculating marginal MRP value for X & Y'
        ct['x_mrp'] = mv.x_dist.predict(mrp=mrp)
        ct['y_mrp'] = mv.y_dist.predict(mrp=mrp)
        self.progress_bar.value += 1

        self.progress_label.value = 'Calculating jagged contour'
        ct['jagged'] = mv._get_jaggaed_contour(mrp)
        self.progress_bar.value += 1

        self.progress_label.value = 'Calculating lower contour with MLE fitting'
        ct['lower'], ct['df_lower'] = mv._smooth_contour_lower(ct)
        self.progress_bar.value += 1

        self.progress_label.value = 'Calculating upper contour with reparameterization'
        ct['upper'], ct['df_upper'], ct['condY_cont_dists_tail'] = \
            mv._smooth_contour_upper(ct, range_ratio=10)
        self.progress_bar.value += 1

        self.progress_label.value = 'Combining final contour'
        ct['final_x'], ct['final_y'] = mv._smooth_contour_combine(ct)
        self.progress_bar.value += 1
            
        mv.ct[mrp] = ct # Record contour result

        # Update display
        self.mrp_exist_select.options = list(mv.ct.keys())
        self.mrp_from_exist.disabled = False
        self.progress_bar.layout.visibility = 'hidden'
        self.progress_label.layout.visibility = 'hidden'
        set_widget_visibility(self.hide_list, 'visible')
        self.update_diag(change=None, mv=mv, mrp=mrp)
        
    def update_diag(self, change, mv, mrp):
        ct = mv.ct[mrp]
        if self.contour_dropdown.value == 'Lower contour':
            self.dist_slider.options = list(ct['df_lower'].index)
            self.repara_plot.layout.width='0%'
            self.update_lower_diag(mv, ct)
        else:
            self.dist_slider.options = list(ct['df_upper'].index)
            self.repara_plot.layout.width='33%'
            self.update_upper_diag(mv, ct)
        
    def update_selection(self, change, mv):
        ct = mv.ct[self.get_mrp()]
        if self.contour_dropdown.value == 'Lower contour':
            ct['lower'] = ct['df_lower'].loc[self.dist_slider.value, 'y_bot']
        else:
            ct['upper'] = ct['df_upper'].loc[self.dist_slider.value, 'y_top']
        ct['final_x'], ct['final_y'] = mv._smooth_contour_combine(ct)
        
    def plot_validations(self, mv, ct):
        plt.plot(mv.x_data, mv.y_data, '.', color=[0.5, 0.5, 0.5], alpha=0.1,  markersize=10)
        plt.plot(ct['jagged']['x'], ct['jagged']['y_bot'], 'b.-')
        plt.plot(ct['jagged']['x'], ct['jagged']['y_top'], 'b.-')
        plt.plot([ct['x_mrp'], ct['x_mrp']], [0, ct['y_mrp']], 'b--')
        plt.plot([0, ct['x_mrp']], [ct['y_mrp'], ct['y_mrp']], 'b--')
        plt.plot(mv.x_dist.sample_coor, mv.median_pred, 'b-.')
        plt.grid(True)
        plt.xlim([0, ct['x_mrp'] * 1.1])
        plt.ylim([
            0, 1.1 * max([ct['y_mrp'], ct['jagged']['y_top'].max()])
        ])
        plt.xlabel(mv.x_name)
        plt.ylabel(mv.y_name)

    def update_lower_diag(self, mv, ct):
        self.dist_err.value = 'Error: ' \
            f"{ct['df_lower'].loc[self.dist_slider.value, 'err']:.2f}  " \
            '(RMS error compared to the jagged lower contour)'

        self.para_plot.clear_output(wait=True)
        with self.para_plot:
            mv.condY_cont_dists_bulk[self.dist_slider.value].plot_diagnosis()
            plt.title('')
            plt.xlabel(mv.x_name)
            plt.show()

        self.contour_plot.clear_output(wait=True)
        with self.contour_plot:
            self.plot_validations(mv, ct)
            plt.plot(
                mv.x_dist.sample_coor, 
                ct['df_lower'].loc[self.dist_slider.value, 'y_bot'],
                'r-', LineWidth=2)
            plt.show()

    def update_upper_diag(self, mv, ct):
        self.dist_err.value = 'Error: ' \
            f"{ct['df_upper'].loc[self.dist_slider.value, 'err']:.2f}  " \
            r'(25% RMS error compared to the jagged upper contour ' +\
            r'+ 75% absolute error compared to MRP of marginal y)'

        self.repara_plot.clear_output(wait=True)
        with self.repara_plot:
            mv.plot_repara_result(ct, self.dist_slider.value)

        self.para_plot.clear_output(wait=True)
        with self.para_plot:
            ct['condY_cont_dists_tail'][self.dist_slider.value].plot_diagnosis()
            plt.title('')
            plt.xlabel(mv.x_name)
            plt.show()

        self.contour_plot.clear_output(wait=True)
        with self.contour_plot:
            self.plot_validations(mv, ct)
            plt.plot(
                mv.x_dist.sample_coor, 
                ct['df_upper'].loc[self.dist_slider.value, 'y_top'],
                'r-', LineWidth=2)
            plt.show()

    def update_mrp_from_exist(self, change):
        self.mrp_from_exist.value = not self.mrp_from_new.value
        self.update_diag(change=None)

    def update_mrp_from_new(self, change):
        self.mrp_from_new.value = not self.mrp_from_exist.value
        
    def dist_slider_prev(self, change):
        idx = self.dist_slider.options.index(self.dist_slider.value)
        idx = max([0, idx - 1])
        self.dist_slider.value = self.dist_slider.options[idx]

    def dist_slider_next(self, change):
        idx = self.dist_slider.options.index(self.dist_slider.value)
        idx = min([len(self.dist_slider.options) - 1, idx + 1])
        self.dist_slider.value = self.dist_slider.options[idx]
    

class SaveFileButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog.
    Modified from https://codereview.stackexchange.com/questions/162920/file-selection-button-for-jupyter-notebook
    """

    def __init__(self, file_type=None, **kwargs):
        super(SaveFileButton, self).__init__(**kwargs)
        self.file_type = file_type
        self.add_traits(file_name=traitlets.traitlets.Unicode())
        self.on_click(self.select_file)

    def select_file(self, b):
        """Generate instance of tkinter.filedialog"""
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        b.file_name = filedialog.asksaveasfilename(filetypes=self.file_type)


class ConfirmDialog:
    def __init__(self, text=None):
        self.text = widgets.Label(value=text)
        self.ok_button = widgets.Button(
            description='OK', layout={'width': '80px'})
        self.cancel_button = widgets.Button(
            description='Cancel', layout={'width': '80px'})
        self.box = widgets.VBox(
            children=[
                self.text, 
                widgets.HBox(children=[self.ok_button, self.cancel_button]),
            ],
            layout={
                'border': 'solid 1px', 
                'padding': '5px 5px 5px 5px',
                'align_items': 'center',
                'width': '40%',
            }
        )

    def show(self):
        self.box.layout.visibility = 'visible'
        self.box.layout.height = None

    def hide(self):
        self.box.layout.visibility = 'hidden'
        self.box.layout.height = '0px'


def button_visual(button_widget):
    button_widget.style.button_color = 'lightgreen'
    button_widget.icon = 'check'
    time.sleep(1)
    button_widget.style.button_color = None
    button_widget.icon = ''
    
def condY_to_str(condY_x: list) -> str:
    '''Convert a list into the format of "start : interval : end" for display
    '''
    return (f'{condY_x[0]:.1f} : '
        f'{condY_x[1] - condY_x[0]:.1f} : '
        f'{condY_x[-1]:.1f}')

def str_to_condY(s: str) -> list:
    '''Convert condY_x expression from text to list
        s has the format of "start : interval : end" or "start : end"
        assuming an interval of 1
    '''
    condY_x = list(map(float, s.split(':')))
    if len(condY_x) == 2:
        condY_x = np.arange(condY_x[0], condY_x[1] * 1.0001, 1)
    elif len(condY_x) == 3:
        # add a small value to "end" so that it is included
        condY_x = np.arange(condY_x[0], condY_x[2] * 1.0001, condY_x[1])
    else:
        raise ValueError('Please check format of CondY_X')
        
    return condY_x

def set_widget_visibility(widget_list, visibility):
    # print('********** set_widget_visibility called ************')
    '''Hide all related widgets for fitting config'''
    for widget in widget_list:
        # print(type(widget))
        setattr(widget.layout, 'visibility', visibility)
