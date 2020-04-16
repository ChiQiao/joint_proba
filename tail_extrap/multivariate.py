import json
import os
import warnings
from functools import partial

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize

from tail_extrap.univariate import Univariate

class Multivariate:
    ''' Tail extrapolation of multi-variable as a paired time series, resulting
            an Environemntal Contour associated with a Mean Return Period (MRP)
            using Inverse First Order Reliability Method (IFORM) with Rosenblatt
            transformation.
        The joint probability density f(X, Y) is expressed as f(X)*f(Y|X), where
            f(X) and f(Y|X) estimated at various X are estimated using the 
            Univariate class for accurate tail extrapolation. f(Y|X) is then 
            approximated using a prescribed distribution to reduce the number of
            parameters. Parameters of f(Y|X) are then fit as functions of X to 
            calculate a smooth contour
        
        Methods include fit, predict, and plot_diagnosis.
        Parameters
        ----------
        data: pandas series with datetime index. If constructing conditional 
            distribution, NaN should be retained for total year to be correctly
            inferred
        sample_coor: numpy array. Sample coordinate for outputs reference. 
            Inferred from data if not provided
    '''
    def __init__(self, df, col_x=0, col_y=1, condY_x=None, dist_cands=None):
        self.x_data = df.iloc[:, col_x]
        self.y_data = df.iloc[:, col_y]
        self.x_name = df.columns[col_x]
        self.y_name = df.columns[col_y]
        if condY_x is None:
            self.condY_x = np.linspace(
                self.x_data.min(), self.x_data.max(), 10)
        else:
            self.condY_x = condY_x

        # Distributions for re-parameterization and their configs
        if dist_cands is None:
            dist_cands = np.array([
                'burr12', 'expon', 'fatiguelife', 'gamma', 'genextreme',
                'genpareto', 'gumbel_r', 'invgauss', 'logistic',
                'lognorm', 'nakagami', 'norm', 'rayleigh', 'weibull_min',
            ])
        dirname = os.path.dirname(__file__)
        config_path = os.path.join(dirname, 'dist_config.json')
        with open(config_path, 'r') as f:
            dist_config = json.load(f)
        idx_valid = np.array([dist in dist_config for dist in dist_cands])
        # Delete candidates that has no config
        if not all(idx_valid):
            warnings.warn(f'Distribution {dist_cands[~idx_valid]} is not '
                          'included in dist_repara.json and will be ignored')
            dist_cands = dist_cands[idx_valid]
        self.dist_cands = dist_cands
        self.dist_config = dist_config

    def fit(self, plot_diagnosis=True, verbose=True):
        '''Fit results that are independent from MRP 

        Variables added:
        ----------------
            self.x_dist, y_dist, condY_dist: Instances of Univariate class for 
                marginal X & Y, and conditional Y
            self.condYs_bulk: list of instances of _CondY class 
                Candidates of condY fitting results using the bulk of the data
        '''
        # Fit marginal X
        if verbose:
            print('Fitting marginal X')
        self._fit_marginalX()

        # Fit marginal Y
        if verbose:
            print('Fitting marginal Y')
        self._fit_marginalY()

        # Fit discrete conditional Y
        if verbose:
            print('Fitting discrete conditional Y')
        self._fit_condY_disc()

        # Median of condY
        self._get_condY_median()

        # Fit the parameters of conditional Y using bulk of the data
        if verbose:
            print('Fitting continuous conditional Y using bulk')
        df = self._get_condY_para_bulk()
        self.condY_cont_dists_bulk = self._fit_condY_cont(df)

    def predict(self, mrp, range_ratio=10):
        '''Re-parameterize tail based on MRP and construct environmental contour 
        
        Parameters:
        -----------
            MRP: numpy array. Target MRP
        '''
        ct = {'mrp': mrp}

        # MRP of independent X & Y 
        ct['x_mrp'] = self.x_dist.predict(mrp=mrp)
        ct['y_mrp'] = self.y_dist.predict(mrp=mrp)

        # Jagged contour
        ct['jagged'] = self._get_jaggaed_contour(mrp)

        # Smooth contour (lower part)
        ct['lower'], ct['df_lower'] = self._smooth_contour_lower(ct)

        # Smooth contour (upper part)
        ct['upper'], ct['df_upper'] = self._smooth_contour_upper(
            ct, range_ratio=range_ratio)

        # Combine contour
        ct['final_x'], ct['final_y'] = self._smooth_contour_combine(ct)

        return ct

    def plot_diagnosis(self):
        def plot_dist_diagnosis():
            if ' ' in dropdown_dist.value:  # contains list index
                attr_name, idx = dropdown_dist.value.split(sep=' ')
                dist = getattr(self, attr_name)[int(idx)]
            else:
                dist = getattr(self, dropdown_dist.value)
            display(dist.diag_fig)

        def update_dist_plot(change):
            dist_display.clear_output(wait=True)
            with dist_display:
                plot_dist_diagnosis()
                plt.show()

        # Tab 1: Univirate fitting
        dropdown_options = [('Marginal X', 'x_dist'), ('Marginal Y', 'y_dist')] +\
            [('Conditional Y at X={:.1f}'.format(condY_x), 'condY_disc_dists {}'.format(idx))
             for idx, condY_x in enumerate(self.condY_x)]
        dropdown_dist = widgets.Dropdown(
            options=dropdown_options, description='Item')
        dropdown_dist.observe(update_dist_plot, names="value")
        dist_display = widgets.Output()
        with dist_display:
            plot_dist_diagnosis()
            plt.show()
        tab1 = widgets.VBox(children=[dropdown_dist, dist_display])

        # Tab 2: Multivirate fitting
        tab2 = widgets.VBox(children=[])

        tab = widgets.Tab(children=[tab1, tab2])
        tab.set_title(0, 'Univariate fitting')
        tab.set_title(1, 'Multivariate fitting')
        return widgets.VBox(children=[tab])
        
    @staticmethod
    def plot_repara_result(df, condY_x, dist_name):
        fit_coor = df['fit_coor'][dist_name]
        mrp_true = df['mrp_true'][dist_name]
        mrp_pred = df['mrp_pred'][dist_name]
        para = df['param'][dist_name]

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        for y, true, pred, x in zip(fit_coor, mrp_true, mrp_pred, condY_x):
            h = plt.plot(true, y, '-', label=x)
            plt.plot(pred, y, '--', color=h[0].get_color())
        plt.xscale('log')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(condY_x, para)
        plt.grid(True)
        plt.show()
    
    def _fit_marginalX(self, **kwargs):
        ''' Fit marginal distribution for x using Univariate object
            Variables added:
            ----------------
                self.x_dist: Univariate object
        '''
        x_dist = Univariate(
            self.x_data,
            sample_coor=np.linspace(0, 2*self.x_data.max(), 1000))
        x_dist.fit(verbose=False, **kwargs)
        self.x_dist = x_dist

    def _fit_marginalY(self, **kwargs):
        ''' Fit marginal distribution for y using Univariate object
            Variables added:
            ----------------
                self.y_dist: Univariate object
        '''
        y_dist = Univariate(
            self.y_data,
            sample_coor=np.linspace(0, 2*self.y_data.max(), 1000))
        y_dist.fit(verbose=False, **kwargs)
        self.y_dist = y_dist

    def _fit_condY_disc(self, **kwargs):
        ''' Fit conditional distributions using Univariate object
            Note:
            -----
                x coordinate is determined by self.condY_x
            Variables added:
            ----------------
                self.condY_dist_dists: list of Univariate objects
        '''
        condY_disc_dists = []
        condY_dx = np.diff(self.condY_x).mean()
        for cur_x in self.condY_x:
            condY_data = self.y_data.copy()
            condY_data[
                (self.x_data < cur_x - condY_dx) |
                (self.x_data > cur_x + condY_dx)
            ] = np.nan
            cur_dist = Univariate(
                condY_data,
                sample_coor=np.linspace(0, 2*self.x_data.max(), 1000)
            )
            cur_dist.fit(verbose=False, **kwargs)
            condY_disc_dists.append(cur_dist)
        self.condY_disc_dists = condY_disc_dists

    def _fit_condY_cont(self, df):
        ''' Fit condY parameters as functions of x for some dist. candidates
            Parameters:
            -----------
                df: DataFrame with index as dist. names and column 'param' as a 
                    numpy array (dist. parameters at different x, where x is 
                    determined by self.condY_x)
                dist_conf: dict of dist. fitting configuration
            Returns:
            ----------------
                condY_cont_dists: list of _CondY objects
        '''
        condY_cont_dists = []
        for idx in range(len(df)):
            params = df['param'][idx]
            idx_valid = ~np.isnan(params).any(axis=1)
            condY = _CondY(
                dist_name=df.index[idx],
                x=self.condY_x[idx_valid],
                params=params[idx_valid],
                dist_config=self.dist_config[df.index[idx]]
            )
            condY.fit()
            condY_cont_dists.append(condY)
        return condY_cont_dists

    def _get_condY_median(self):
        ''' Fitting the median of condY for self.x_dist.sample_coor '''
        def fitting_func(x, a, b, c):
            return a * x**b + c

        median_emp = [np.median(condY_data.data) 
            for condY_data in self.condY_disc_dists]
        popt, _ = curve_fit(
            fitting_func, self.condY_x, median_emp, method='trf', 
            bounds=([0, 0, 0], np.inf), max_nfev=1e4
        )
        median_pred = fitting_func(self.x_dist.sample_coor, *popt)
        self.median_emp = median_emp
        self.median_pred = median_pred
        
    def _get_condY_F(self, mrp, x):
        ''' Return F of condY given MRP
            Parameters:
            -----------
                mrp: float
                x: array-like, output coordinates for y_F
            Returns:
            --------
                y_F: numpy array with the same size as x
        '''
        std_norm = stats.norm()
        beta = std_norm.ppf(1 - 1/self.x_dist.c_rate/mrp)
        x_F = interp1d(self.x_dist.sample_coor, self.x_dist.sample_F)(x)
        x_beta = std_norm.ppf(x_F)
        y_beta_square = beta ** 2 - x_beta ** 2
        y_beta_square[y_beta_square < 0] = np.nan
        y_beta = np.sqrt(y_beta_square)
        with np.errstate(invalid='ignore'):
            y_F = std_norm.cdf(y_beta)
        return y_F

    def _get_condY_para_bulk(self):
        ''' Fit dist parameters for self.condY_disc_dists using the bulk
            Returns:
            --------
                df: Dataframe of different distributions (as index) with
                    averaged chi_square, averaged r2, and parameters for the 
                    corresponding x.
            Notes:
            ------
                Only distributions with chi_square lower than 2 * optimal result 
                    are returned
        '''
        df_temp = pd.DataFrame()
        for condY_dist in self.condY_disc_dists:
            data = condY_dist.data
            df_cur = Univariate.best_fit(
                data, dist_names=self.dist_cands, dist_config=self.dist_config)
            df_temp = pd.concat(
                [df_temp, df_cur.set_index('Distribution')], axis=1)

        # Arrange condY fitting results
        # chi_square, r2: mean
        # param: stack for different condY_x
        df = pd.concat(
            [
                df_temp['chi_square'].mean(axis=1, skipna=False),
                df_temp['r2'].mean(axis=1, skipna=False),
                df_temp['param'].apply(np.vstack, axis=1),
            ], axis=1
        ).rename(
            columns={0: 'chi_square', 1: 'r2', 2: 'param'}
        ).sort_values(by='chi_square')

        # Select candidates based on chi_square
        df = df[df['chi_square'] < df['chi_square'][0] * 2]
        return df

    def _get_condY_para_tail(self, mrp, range_ratio):
        ''' Fit dist parameters for self.condY_disc_dists using tail re-para
            Parameters:
            -----------
                mrp: number
                range_ratio: the conditional dist. between MRP/range_ratio and
                    MRP*range_ratio will be used for re-para
            Returns:
            --------
                df: Dataframe of different distributions (as index) with
                    'param' for the dist parameters, 'fit_coor', 'mrp_true', and 
                    'mrp_pred' for diagnosis
        '''
        def get_condY_F_range(mv, mrp, range_ratio):
            ''' Determine range of re-parameterization '''
            std_norm = stats.norm()
            # Three circles for MRP, MRP/range_ratio, MRP*range_ratio
            beta = std_norm.ppf(1 - 1/mv.x_dist.c_rate/mrp)
            beta_lb = std_norm.ppf(1 - 1/mv.x_dist.c_rate/(mrp/range_ratio))
            beta_ub = std_norm.ppf(1 - 1/mv.x_dist.c_rate/(mrp*range_ratio))
            # x coordinates of the MRP circle (x_beta)
            x_F = interp1d(mv.x_dist.sample_coor,
                           mv.x_dist.sample_F)(mv.condY_x)
            x_beta = std_norm.ppf(x_F)
            # y coordinates of the ellipse connecting circle 1&2, 1&3
            y_beta_lb = beta_lb * np.sqrt(1 - x_beta**2 / beta**2)
            y_beta_ub = beta_ub * np.sqrt(1 - x_beta**2 / beta**2)
            # CDF of these y coordinates
            condY_F_lb = std_norm.cdf(y_beta_lb)
            condY_F_ub = std_norm.cdf(y_beta_ub)
            return condY_F_lb, condY_F_ub

        def get_para_bounds(dist_config):
            ''' Process parameter boundaries for a given distribution config 
                Convert the 'fit_kwargs' and 'para_lowerbound' in dist_config
                into the form of ((lb1, ub1), (lb2, ub2), ...) for optimization
            '''
            lbs = [val if not isinstance(val, str) else None
                   for val in dist_config['para_lowerbound']]
            ubs = [None] * len(lbs)
            if 'fscale' in dist_config['fit_kwargs']:
                ubs[-1] = lbs[-1] = dist_config['fit_kwargs']['fscale']
            if 'floc' in dist_config['fit_kwargs']:
                ubs[-2] = lbs[-2] = dist_config['fit_kwargs']['floc']
            bnds = tuple((lb, ub) for lb, ub in zip(lbs, ubs))
            return bnds

        def dist_to_mrp(param, dist_name, fit_coor, rate):
            ''' Predict MRP for fit_coor using dist_name with param '''
            dist = stats._distn_infrastructure.rv_frozen(
                getattr(stats, dist_name),
                *param[:-2], loc=param[-2], scale=param[-1]
            )
            F = dist.cdf(fit_coor)
            mrp_pred = 1 / rate / (1-F)
            return mrp_pred

        def mrp_msle(param, dist_name, fit_coor, rate, mrp_true):
            ''' Loss function for optimization '''
            mrp_pred = dist_to_mrp(param, dist_name, fit_coor, rate)
            msle = np.nanmean((np.log(mrp_pred) - np.log(mrp_true)) ** 2)
            return msle

        condY_F_lb, condY_F_ub = get_condY_F_range(self, mrp, range_ratio)
        df = pd.DataFrame()
        for dist_name in self.dist_cands:
            # Skip some distributions due to slow fitting
            # if dist_name in ['genextreme', 'genpareto']:
            #     continue
            
            # print(dist_name)
            param, diag_fit_coor, diag_mrp_true, diag_mrp_pred = [], [], [], []
            param_prev = None
            bnds = get_para_bounds(self.dist_config[dist_name])
            for uv, F_lb, F_ub in zip(self.condY_disc_dists, condY_F_lb, condY_F_ub):
                mrp_true = np.copy(uv.sample_mrp)
                fit_coor = np.copy(uv.sample_coor)
                y_lb, y_ub = interp1d(
                    uv.sample_F, uv.sample_coor)([F_lb, F_ub])
                idx_nan = (uv.sample_F < F_lb) | (uv.sample_F > F_ub)
                mrp_true[idx_nan] = np.nan
                fit_coor[idx_nan] = np.nan
                if param_prev is None:
                    x0 = getattr(stats, dist_name).fit(
                        uv.data, **self.dist_config[dist_name]['fit_kwargs'])
                else:
                    x0 = param_prev
                res = minimize(
                    mrp_msle, x0,
                    args=(dist_name, fit_coor, uv.c_rate, mrp_true),
                    method='SLSQP',
                    bounds=bnds, tol=1e-4, options={'maxiter': 500}
                )
                mrp_pred = dist_to_mrp(
                    res['x'], dist_name, fit_coor, uv.c_rate)
                if res['success']:
                    param_prev = np.copy(res['x'])
                else:
                    res['x'].fill(np.nan)
                    mrp_pred.fill(np.nan)
                # print(f"{dist_name}: {res['message']}")
                param.append(res['x'])
                # Record diagnostic information
                diag_fit_coor.append(fit_coor)
                diag_mrp_true.append(mrp_true)
                diag_mrp_pred.append(mrp_pred)
            df = df.append(
                pd.DataFrame.from_dict({
                    'dist_name': dist_name,
                    'param': [np.vstack(param)],
                    'fit_coor': [np.vstack(diag_fit_coor)],
                    'mrp_true': [np.vstack(diag_mrp_true)],
                    'mrp_pred': [np.vstack(diag_mrp_pred)],
                }), ignore_index=True
            )
        df = df.set_index('dist_name')
        # Delete dist that gives all nan in param
        df = df[df['param'].apply(lambda x: ~np.all(np.isnan(x)))]
        return df

    def _get_jaggaed_contour(self, mrp):
        ''' Calculate a jagged contour from self.condY_disc_dists
            Parameters:
            -----------
                mrp: MRP of the contour
            Returns:
            --------
                res: dict including x, y_bot, and y_top
            Note:
            -----
                x coordinate is determined by self.condY_x
        '''
        condY_F = self._get_condY_F(mrp, self.condY_x)
        contour = np.array(
            [interp1d(
                dist.sample_F, dist.sample_coor, bounds_error=False
            )([1-F, F])
            for dist, F in zip(self.condY_disc_dists, condY_F)])
        res = {
            'x': self.condY_x,
            'y_bot': contour[:, 0],
            'y_top': contour[:, 1],
        }
        return res

    def _get_smooth_contour(self, condY, mrp):
        ''' Calculate a smooth contour from a _CondY object
            Parameters:
            -----------
                condY: _CondY object
                mrp: MRP of the contour
            Returns:
            --------
                res: dict including dist_name, x, y_bot, and y_top
            Note:
            -----
                x coordinate is determined by self.x_dist.sample_coor
        '''
        condY_F = self._get_condY_F(mrp, self.x_dist.sample_coor)
        contour = np.array([
            condY.predict(x).ppf([1-F, F]) for x, F
            in zip(self.x_dist.sample_coor, condY_F)
        ])
        res = {
            'dist_name': condY.dist_name,
            'x': self.x_dist.sample_coor,
            'y_bot': contour[:, 0],
            'y_top': contour[:, 1],
        }
        return res

    def _smooth_contour_upper(self, ct, range_ratio=10):
        ''' Get upper part of the smooth contour using re-para method '''
        # Fit condY para using re-para method
        df_diag = self._get_condY_para_tail(ct['mrp'], range_ratio=range_ratio)

        # Fit condY para as functions of x
        condY_cont_dists_tail = self._fit_condY_cont(df_diag)
        
        # Construct contours
        contours_tail = {}
        for condY in condY_cont_dists_tail:
            res = self._get_smooth_contour(condY, ct['mrp'])
            dist_name = res.pop('dist_name')
            contours_tail[dist_name] = res
        df_diag = pd.concat([
            df_diag, 
            pd.DataFrame.from_dict(contours_tail, orient='index')],
            axis=1,
        )
        df_diag.drop('x', axis=1, inplace=True)

        # Evaluate contour using the jagged contour and y_mrp
        err_emp, err_apex = [], []
        ct_emp = ct['jagged']['y_top']
        for dist_name, ct_tail in contours_tail.items():
            ct_pred = interp1d(
                ct_tail['x'], ct_tail['y_top']
            )(self.condY_x) 
            mse = np.mean((ct_pred - ct_emp) ** 2)
            df_diag.loc[dist_name, 'err_emp'] = np.sqrt(mse)
            df_diag.loc[dist_name, 'err_apex'] = np.abs(
                np.nanmax(ct_tail['y_top']) - ct['y_mrp'])
        df_diag['err'] = df_diag.apply(
            lambda x: 0.25*x['err_emp'] + 0.75*x['err_apex'], axis=1)

        # Select the best contour
        df_diag = df_diag.sort_values('err')
        contour_upper = df_diag['y_top'][0]
        return contour_upper, df_diag

    def _smooth_contour_lower(self, ct):
        ''' Get lower part of the smooth contour using MLE fitting 
            Returns:
            --------
                diag_df: Dataframe recording results of the candidates 
                        Columns: 'y_bot', 'err'
        '''
        # Get contour candidates from self.condY_cont_dists_bulk
        contours_bulk = {}
        for condY in self.condY_cont_dists_bulk:
            res = self._get_smooth_contour(condY, ct['mrp'])
            dist_name = res.pop('dist_name')
            contours_bulk[dist_name] = res
        # Restore contour candidates in df_diag
        df_diag = pd.DataFrame.from_dict(contours_bulk, orient='index')
        df_diag.drop('x', axis=1, inplace=True)
            
        # Select the best contour based on mse using contour_jag
        err_emp = []
        ct_emp = ct['jagged']['y_bot']
        for dist_name, ct_bulk in contours_bulk.items():
            ct_pred = interp1d(
                ct_bulk['x'], ct_bulk['y_bot']
            )(self.condY_x) 
            mse = np.mean((ct_pred - ct_emp) ** 2)
            df_diag.loc[dist_name, 'err'] = np.sqrt(mse)
        df_diag = df_diag.sort_values('err')
        contour_lower = df_diag['y_bot'][0]
        return contour_lower, df_diag

    def _smooth_contour_combine(self, ct):
        x_start = max([
            self.condY_x.max(), 
            self.x_dist.sample_coor[np.nanargmax(ct['upper'])]
        ])
        idx_adj = (self.x_dist.sample_coor >= x_start) & \
            (self.x_dist.sample_coor <= ct['x_mrp'])
        x_adj = self.x_dist.sample_coor[idx_adj]

        # Upper part
        upper_adj = ct['upper'][idx_adj]
        upper_max = upper_adj[0]
        upper_temp = (upper_max - upper_adj)
        ratio = (upper_max - self.median_pred[idx_adj][-1]) / upper_temp[-1]
        upper_temp *= ratio
        # In case the median gives higher result than the apex
        upper_temp[upper_temp < 0] = 0 
        upper_temp = upper_max - upper_temp
        upper_new = np.copy(ct['upper'])
        upper_new[idx_adj] = upper_temp

        # Lower part
        lower_adj = ct['lower'][idx_adj]
        lower_min = lower_adj[0]
        lower_temp = (lower_adj - lower_min)
        ratio = (upper_temp[-1] - lower_min) / lower_temp[-1]
        lower_temp *= ratio
        lower_temp += lower_min
        lower_new = np.copy(ct['lower'])
        lower_new[idx_adj] = lower_temp

        # Combined contour
        idx_ct = ~np.isnan(upper_new)
        final_x = np.hstack(
            [np.flip(self.x_dist.sample_coor[idx_ct]),
            self.x_dist.sample_coor[idx_ct]])
        final_y = np.hstack(
            [np.flip(upper_new[idx_ct]),
            lower_new[idx_ct]])
        return final_x, final_y


class _CondY:
    ''' Conditional distribution f(Y|X)
        Parameters
        ----------
            dist_name: str, name of the distribution used for fitting
            x: numpy array, x coordinates for condY
            params: numpy ndarray, each column is the value of a parameter at x
            dist_config: dict with key 'para_lowerbound', fitting config
    '''

    def __init__(self, dist_name, x, params, dist_config):
        self.dist_name = dist_name
        params_name = getattr(stats, dist_name).shapes
        if params_name is None:
            self.params_name = ['loc', 'scale']
        else:
            self.params_name = params_name.split(', ') + ['loc', 'scale']
        self.x = x
        self.params_raw = params
        self._params_lb = dist_config['para_lowerbound']

    def __str__(self):
        return f'Conditional distribution fitting using {self.dist_name}'

    def fit(self):
        ''' Fit each condY parameter as a function of x
            Note:
            -----
                If a parameter is fixed (e.g., floc=0), a function returning that 
                    value is used
                If a parameter is varying, fitting expression is determined by
                    fitting_func
            Variables added:
            ----------------
                self._coef_func: list with the same length of condY parameters
                    loc and scale are included
                    each list element is a function of x
        '''
        def fitting_func(x, a, b, c):
            return a * x**b + c

        def constant_func(x, c):
            return c

        coef_lb = {
            '-inf': -np.inf * np.array([1, 1, 1]),
            0: [0, -np.inf, 0]
        }

        coef_func = []
        for param_raw, param_lb in zip(self.params_raw.T, self._params_lb):
            unique_values = np.unique(param_raw)
            if len(unique_values) == 1:  # Fixed parameter
                coef_func.append(partial(constant_func, c=unique_values[0]))
            else:
                popt, _ = curve_fit(
                    fitting_func, self.x, param_raw, method='trf',
                    bounds=(coef_lb[param_lb], np.inf), max_nfev=1e4
                )
                coef_func.append(
                    partial(fitting_func, a=popt[0], b=popt[1], c=popt[2]))
        self._coef_func = coef_func

    def predict(self, x):
        ''' Return a scipy distribution object '''
        param = [func(x) for func in self._coef_func]
        dist = stats._distn_infrastructure.rv_frozen(
            getattr(stats, self.dist_name),
            *param[:-2], loc=param[-2], scale=param[-1]
        )
        return dist

    def plot_diagnosis(self, x_sample=None):
        ''' Plot condY parameter fitting result
            Parameters:
            -----------
                x_sample: array-like. If not provided, CondY.x is used with upper bound 
                    extended to 1.5 times
        '''
        if x_sample is None:
            x_sample = np.linspace(self.x.min(), 1.5 * self.x.max(), 200)
        plt.figure()
        for idx in range(self.params_raw.shape[1]):
            h = plt.plot(self.x, self.params_raw[:, idx], 'x')
            plt.plot(x_sample, [self._coef_func[idx](x) for x in x_sample],
                     '-', color=h[0].get_color(), label=self.params_name[idx])
        plt.xlabel('x')
        plt.title(self.dist_name)
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    import pickle
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, '../datasets/D.pkl')
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    mv = Multivariate(df)
