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

# from tail_extrap import multivariate
import multivariate
import copy

font_size = 14
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('figure', titlesize=font_size)
layout_section = {'margin': '12px 2px 12px 2px'}

# TODO: Dataset C: 1:5 (6?) leads to optimization error, adjust by 0.5 solves
# the problem; 0.5:1:6.5 leads to median intersect with mrp_y

class Interactive:
    def __init__(self, mv):
        self.mv = mv

        self.uni_fit_button = widgets.Button() # In tab1

        self.save_button = SaveFileButton(
            description='Save session as...',
            tooltip='Save current session as a pickle file',
            file_type=[("pickle archive", ".pkl")],
            default_type='.pkl'
        )

        self.tab0 = Tab_config(self.mv, self.uni_fit_button)
        self.tab1 = Tab_univariate(self.mv, self.uni_fit_button)
        self.tab2 = Tab_contour(self.mv)
        
        tab = widgets.Tab(children=[
            self.tab0.tab, self.tab1.tab, self.tab2.tab
            ])
        tab.set_title(0, 'Data config')
        tab.set_title(1, 'Univariate fitting')
        tab.set_title(2, 'Contour construction')

        self.save_button.on_click(self.save_session)
        self.tab0.update_button.on_click(
            functools.partial(self.tab0_update_clicked, mv=mv))
        self.tab0.confirm_box.ok_button.on_click(
            functools.partial(self.tab0_update_confirmed, mv=self.mv))

        display(
            widgets.VBox(children=[
                self.save_button,
                tab,
            ])
        )
    # TODO: fig resolution, optimize chache, Update tab1 clears tab2, add tail
    # threshold for marginal x and y
    def save_session(self, change):
        # local function fitting_func in multivariate._CondY.fit cannot be
        #   pickled. As an workaround, mv.condY_cont_dists_bulk is removed and
        #   refit when loaded next time
        mv_temp = copy.copy(self.mv)
        if hasattr(mv_temp, 'condY_cont_dists_bulk'):
            delattr(mv_temp, 'condY_cont_dists_bulk')
        if self.save_button.file_name != '':
            with open(self.save_button.file_name, 'wb') as f:
                pickle.dump(mv_temp, f)
            button_visual(self.save_button, 'Saved')

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
        button_visual(self.tab0.update_button, 'Updated')

        # Reset tab1 display
        self.tab1.uni_fit_button.description = 'Start Fitting' 
        set_widget_visibility(self.tab1.hide_list, 'hidden')
        
        # Update tab2 display
        self.tab2.mrp_exist_select.options = []
        self.tab2.mrp_from_new.value = True
    
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

        self.data_display = widgets.Output(layout=layout_section)

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
            plt.figure(figsize=(8, 6))
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

        self.data_display = widgets.Output(layout={'height': '450px'})
        self.plot_section = widgets.HBox(
            children=[self.data_display],
            layout=layout_section,
        )
        
        self.tab = widgets.VBox(children=[
            self.fit_section, 
            self.config_section,
            self.plot_section,
        ])

        self.hide_list = [self.config_section, self.plot_section,
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
            button_visual(self.uni_fit_button, 'Updated')
    

class Tab_contour:
    def __init__(self, mv):
        self.mv = mv

        # Fitting status

        self.fit_button = widgets.Button(
            description='Start Fitting',
            tooltip='Fit contour for the current MRP'
        )
        self.export_button = SaveFileButton(
            description='Export Contour',
            tooltip='Export the current contour to a csv file',
            file_type=[("CSV", ".csv")],
            default_type='.csv'
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
                    self.fit_button, self.export_button, 
                    self.progress_bar, self.progress_label]), 
                self.confirm_box.box
            ],
        )

        self.fit_button.on_click(self.fit_clicked)
        self.confirm_box.ok_button.on_click(self.fit_confirmed)
        self.confirm_box.cancel_button.on_click(self.cancel_clicked)
        self.export_button.on_click(self.export_contour)
        
        # MRP selection

        self.mrp_from_new = widgets.Checkbox(
            description='Create new MRP: ',
            value=not bool(mv.ct),
            indent=False,
            layout={'width': 'max-content'},
        )
        self.mrp_from_exist = widgets.Checkbox(
            description='Existing MRP: ',
            value=bool(mv.ct),
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
                self.mrp_from_new, self.mrp_new, 
                widgets.Box(layout={'width': '40px'}),
                self.mrp_from_exist, self.mrp_exist_select
            ],
            layout=layout_section,
        )
        
        if not self.mrp_exist_select.options: 
            self.mrp_from_exist.disabled = True
        self.mrp_from_new.observe(self.mrp_from_new_clicked, names='value')
        self.mrp_from_exist.observe(self.mrp_from_exist_clicked, names='value')
        self.mrp_exist_select.observe(self.update_mrp_select, names='value')

        # Contour distribution selection

        self.contour_dropdown = widgets.Dropdown(
            options=['Upper contour', 'Lower contour'],
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
            layout=widgets.Layout(width='60%'),
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
            self.update_contour_dropdown, names='value')
        self.dist_slider.observe(self.update_diag, names='value')
        self.dist_prev.on_click(self.dist_slider_prev)
        self.dist_next.on_click(self.dist_slider_next)
        self.select_button.on_click(self.update_dist_selection)

        # Diagnostic plots

        self.contour_plot = widgets.Output(
            layout={'width': '55%', 'height': '450px', 'align_items': 'center'})
        self.para_plot = widgets.Output()
        self.repara_plot = widgets.Output()
        
        self.plot_section = widgets.HBox(
            children=[
                self.contour_plot,
                widgets.VBox(
                    children=[self.para_plot, self.repara_plot],
                    layout={'width': '45%', 'align_items': 'center'}
                ),
            ],
            layout=layout_section
        )

        self.tab = widgets.VBox(children=[
            self.fit_section, self.mrp_section, 
            self.dist_section, self.plot_section,
        ])

        self.hide_list = [self.dist_section, self.plot_section]
        self.cur_mrp = self.get_mrp()

        if not mv.ct:
            # No existing contour result
            set_widget_visibility(self.hide_list, 'hidden')
            self.export_button.disabled = True
        else:
            # Has existing contour result
            self.update_contour_dropdown(change=None)
            self.update_diag(change=None)
            self.export_button.disabled = False

    def export_contour(self, change):
        '''Export the current contour to a csv file'''
        if self.export_button.file_name != '':
            print('a')
            df = pd.DataFrame({
                self.mv.x_name: self.mv.ct[self.cur_mrp]['final_x'], 
                self.mv.y_name: self.mv.ct[self.cur_mrp]['final_y']})
            df.to_csv(self.export_button.file_name, index=False)
            button_visual(self.export_button, 'Export Sucessful')

    def get_mrp(self) -> int:
        '''Return currently selected MRP'''
        if self.mrp_from_new.value:
            return self.mrp_new.value
        else:
            return self.mrp_exist_select.value

    def cancel_clicked(self, change):
        '''Hide confirm_box and enable fit_button'''
        self.confirm_box.hide()
        self.fit_button.disabled = False

    def fit_clicked(self, change):
        '''Either fit directly (call fit_confirmed) or show confirm_box'''
        mrp = self.get_mrp()
        if self.mrp_from_new and mrp in self.mrp_exist_select.options:
            # New mrp is in the existing mrp list
            self.confirm_box.show()
            self.fit_button.disabled = True
        else:
            self.fit_confirmed(change=None)

    def fit_confirmed(self, change):
        '''Fit contour for the current MRP'''
        mv = self.mv
        self.confirm_box.hide()
        self.fit_button.disabled = False
        set_widget_visibility(self.hide_list, 'hidden')
        self.progress_bar.layout.visibility = 'visible'
        self.progress_label.layout.visibility = 'visible'
        self.progress_bar.value = 0

        mrp = self.get_mrp()
        self.cur_mrp = mrp
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
            
        ct = self.cache_tail_para(ct)
        ct['dist_name_lower'] = list(ct['df_lower'].index)[0]
        ct['dist_name_upper'] = list(ct['df_upper'].index)[0]
        mv.ct[mrp] = ct # Record contour result
        
        # Update display
        self.mrp_exist_select.options = list(mv.ct.keys())
        self.mrp_exist_select.value = self.cur_mrp
        self.mrp_from_exist.value = True
        self.mrp_from_exist.disabled = False
        self.progress_bar.layout.visibility = 'hidden'
        self.progress_label.layout.visibility = 'hidden'
        set_widget_visibility(self.hide_list, 'visible')
        self.update_diag(change=None)
    
    def cache_tail_para(self, ct):
        '''Record data to avoid calling multivariate._CondY.plot_diagnosis()'''
        x_sample = np.linspace(self.mv.x_dist.sample_coor[0], ct['x_mrp'], 200)
        ct['tail_diag'] = {}
        ct['tail_diag']['x_sample'] = x_sample
        for dist_name, condY in ct['condY_cont_dists_tail'].items():
            ct['tail_diag'][dist_name] = {}
            for idx in range(condY.params_raw.shape[1]):
                para_name = condY.params_name[idx]
                ct['tail_diag'][dist_name][para_name] = {}
                ct['tail_diag'][dist_name][para_name]['para_raw'] = condY.params_raw[:, idx]
                ct['tail_diag'][dist_name][para_name]['x_raw'] = condY.x
                ct['tail_diag'][dist_name][para_name]['para_fit'] = [condY._coef_func[idx](x) for x in x_sample]
        ct.pop('condY_cont_dists_tail')

        return ct  

    def plot_tail_para(self, dist_name):
        x_sample = self.mv.ct[self.cur_mrp]['tail_diag']['x_sample']
        cur_diag = self.mv.ct[self.cur_mrp]['tail_diag'][dist_name]
        y_lb, y_ub = [], []# Lower and upper bounds of raw parameter
        for para_name, para_res in cur_diag.items():
            h = plt.plot(para_res['x_raw'], para_res['para_raw'], 'x')
            plt.plot(x_sample, para_res['para_fit'],
                    '-', color=h[0].get_color(), label=para_name)
            y_ub.append(max(para_res['para_raw']))
            y_lb.append(min(para_res['para_raw']))
        plt.xlabel(self.mv.x_name)
        plt.ylabel('$f(y|x)$ parameters')
        
        # Adjust ylim incase fitting result blows up
        ylm = list(plt.ylim())
        ylm[0] = max([ylm[0], min(y_lb) * 0.5])
        ylm[1] = min([ylm[1], max(y_ub) * 2])
        plt.ylim(ylm)
        
        plt.grid(True)
        plt.legend(loc='best')
        plt.title('CondY parameters fitting')
        plt.show()

    def update_diag(self, change):
        '''Update plot_section'''
        if self.contour_dropdown.value == 'Lower contour':
            self.update_lower_diag()
        else:
            self.update_upper_diag()
    
    def update_contour_dropdown(self, change):
        '''Update dist_slider'''
        mv = self.mv
        ct = mv.ct[self.cur_mrp]
        if self.contour_dropdown.value == 'Lower contour':
            self.dist_slider.options = list(ct['df_lower'].index)
            self.dist_slider.value = ct['dist_name_lower']
        else:
            self.dist_slider.options = list(ct['df_upper'].index)
            self.dist_slider.value = ct['dist_name_upper']

    def update_dist_selection(self, change):
        '''Update which distribution to use for the contour'''
        mv = self.mv
        ct = mv.ct[self.get_mrp()]
        if self.contour_dropdown.value == 'Lower contour':
            (ct['dist_name_lower'], ) = self.dist_slider.value,
            ct['lower'] = ct['df_lower'].loc[self.dist_slider.value, 'y_bot']
        else:
            (ct['dist_name_upper'], ) = self.dist_slider.value,
            ct['upper'] = ct['df_upper'].loc[self.dist_slider.value, 'y_top']
        ct['final_x'], ct['final_y'] = mv._smooth_contour_combine(ct)
        self.update_diag(change=None)
        
    def plot_validations(self):
        '''Plot basic information for the diagnostic plot'''
        mv = self.mv
        ct = mv.ct[self.cur_mrp]
        plt.plot(mv.x_data, mv.y_data, '.', color=[0.5, 0.5, 0.5], 
            alpha=0.1,  markersize=10)
        plt.plot(ct['jagged']['x'], ct['jagged']['y_bot'], 'b.-')
        plt.plot(ct['jagged']['x'], ct['jagged']['y_top'], 'b.-')
        plt.plot([ct['x_mrp'], ct['x_mrp']], [0, ct['y_mrp']], 'b--')
        plt.plot([0, ct['x_mrp']], [ct['y_mrp'], ct['y_mrp']], 'b--')
        plt.plot(mv.x_dist.sample_coor, mv.median_pred, 'b-.')
        plt.plot(ct['final_x'], ct['final_y'], '--', 
            LineWidth=2, color=[0.5, 0.5, 0.5])
        plt.grid(True)
        plt.xlim([0, ct['x_mrp'] * 1.1])
        plt.ylim([
            0, 1.1 * max([ct['y_mrp'], ct['jagged']['y_top'].max()])
        ])
        plt.xlabel(mv.x_name)
        plt.ylabel(mv.y_name)
        plt.title('Final contour')

    def update_lower_diag(self):
        '''Update display based on the current distribution for lower contour'''
        mv = self.mv
        ct = mv.ct[self.cur_mrp]
        self.dist_err.value = 'Error: ' \
            f"{ct['df_lower'].loc[self.dist_slider.value, 'err']:.2f}  " \
            '(RMS error compared to the jagged lower contour)'

        self.repara_plot.clear_output(wait=False)

        self.para_plot.clear_output(wait=True)
        with self.para_plot:
            plt.figure(figsize=(5, 3))
            mv.condY_cont_dists_bulk[self.dist_slider.value].plot_diagnosis()
            plt.title('')
            plt.xlabel(mv.x_name)
            plt.ylabel('$f(y|x)$ parameters')
            plt.title('CondY parameters fitting')
            plt.show()

        self.contour_plot.clear_output(wait=True)
        with self.contour_plot:
            plt.figure(figsize=(7, 7))
            self.plot_validations()
            plt.plot(
                mv.x_dist.sample_coor, 
                ct['df_lower'].loc[self.dist_slider.value, 'y_bot'],
                'r-', LineWidth=2)
            plt.show()

    def update_upper_diag(self):
        '''Update display based on the current distribution for upper contour'''
        mv = self.mv
        ct = mv.ct[self.cur_mrp]
        self.dist_err.value = 'Error: ' \
            f"{ct['df_upper'].loc[self.dist_slider.value, 'err']:.2f}  " \
            r'(25% RMS error compared to the jagged upper contour ' +\
            r'+ 75% absolute error compared to MRP of marginal y)'

        self.repara_plot.clear_output(wait=True)
        with self.repara_plot:
            plt.figure(figsize=(5, 3))
            mv.plot_repara_result(ct, self.dist_slider.value)
            plt.title('Tail re-parameterization')
            plt.show()

        self.para_plot.clear_output(wait=True)
        with self.para_plot:
            plt.figure(figsize=(5, 3))
            self.plot_tail_para(dist_name=self.dist_slider.value)
            
        self.contour_plot.clear_output(wait=True)
        with self.contour_plot:
            plt.figure(figsize=(7, 7))
            self.plot_validations()
            plt.plot(
                mv.x_dist.sample_coor, 
                ct['df_upper'].loc[self.dist_slider.value, 'y_top'],
                'r-', LineWidth=2)
            plt.show()
    
    def mrp_from_exist_clicked(self, change):
        if self.mrp_from_exist.value:
            self.update_mrp_from_exist()
        else:
            self.update_mrp_from_new()

    def mrp_from_new_clicked(self, change):
        if self.mrp_from_new.value:
            self.update_mrp_from_new()
        else:
            self.update_mrp_from_exist()

    def update_mrp_from_exist(self):
        '''Update when "Existing MRP" is selected'''
        self.mrp_from_new.value = False
        self.export_button.disabled = False
        set_widget_visibility(self.hide_list, 'visible')
        self.cur_mrp = self.mrp_exist_select.value
        self.update_diag(change=None)

    def update_mrp_from_new(self):
        '''Update when "Create new MRP" is selected'''
        self.mrp_from_exist.value = False
        self.export_button.disabled = True
        set_widget_visibility(self.hide_list, 'hidden')

    def update_mrp_select(self, change):
        '''Update when MRP selection is changed'''
        if self.mrp_exist_select.options:
            self.cur_mrp = self.mrp_exist_select.value
            self.update_contour_dropdown(change=None)
            self.update_diag(change=None)
        
    def dist_slider_prev(self, change):
        '''Move dist_slider to previous selection'''
        idx = self.dist_slider.options.index(self.dist_slider.value)
        idx = max([0, idx - 1])
        self.dist_slider.value = self.dist_slider.options[idx]

    def dist_slider_next(self, change):
        '''Move dist_slider to next selection'''
        idx = self.dist_slider.options.index(self.dist_slider.value)
        idx = min([len(self.dist_slider.options) - 1, idx + 1])
        self.dist_slider.value = self.dist_slider.options[idx]
    

class SaveFileButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog.
    Modified from https://codereview.stackexchange.com/questions/162920/file-selection-button-for-jupyter-notebook
    """

    def __init__(self, file_type, default_type, **kwargs):
        super(SaveFileButton, self).__init__(**kwargs)
        self.file_type = file_type
        self.default_type = default_type
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
        b.file_name = filedialog.asksaveasfilename(
            filetypes=self.file_type, defaultextension=self.default_type)


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


def button_visual(button_widget, description=None):
    original_description = button_widget.description
    button_widget.style.button_color = 'lightgreen'
    button_widget.icon = 'check'
    if description:
        button_widget.description = description
    time.sleep(1)
    button_widget.style.button_color = None
    button_widget.icon = ''
    if description:
        button_widget.description = original_description
    
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
    '''Hide all related widgets for fitting config'''
    for widget in widget_list:
        setattr(widget.layout, 'visibility', visibility)
