import calendar
import copy
import ipywidgets as widgets
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from functools import partial
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class Multivariate:
    def __init__(self, df, col_x=0, col_y=1, condY_x=None, dist_cands=None):
        self.x_data = df.iloc[:, col_x]
        self.y_data = df.iloc[:, col_y]
        if condY_x is None:
            self.condY_x = np.linspace(
                self.x_data.min(), self.x_data.max(), 10)
        else:
            self.condY_x = condY_x

        if dist_cands is None:
            dist_cands = np.array([
                'burr12', 'expon', 'fatiguelife', 'gamma', 'genextreme',
                'genpareto', 'gumbel_r', 'invgauss', 'logistic',
                'lognorm', 'nakagami', 'norm', 'rayleigh', 'weibull_min',
            ])
        with open('../config/dist_repara.json', 'r') as f:
            dist_config = json.load(f)
        idx_valid = np.array([dist in dist_config for dist in dist_cands])
        # Delete candidates that has no config
        if not all(idx_valid):
            warnings.warn(f'Distribution {dist_cands[~idx_valid]} is not '
                          'included in dist_repara.json and will be ignored')
            dist_cands = dist_cands[idx_valid]
        self.dist_cands = dist_cands
        self.dist_config = dist_config

    def fit(self, dist_cands=None, plot_diagnosis=True, verbose=True):
        ''' Fit results that are independent from MRP 
            Variables added:
            ----------------
                self.x_dist, y_dist, condY_dist: Univariate objects for marginal X & Y, 
                    and conditional Y
                self.condYs_bulk: list of _CondY object as candidates of condY fitting
                    results using the bulk of the data
        '''
        # Distributions for re-parameterization and their configs
        if dist_cands is None:
            dist_cands = np.array([
                'burr12', 'expon', 'fatiguelife', 'gamma', 'genextreme',
                'genpareto', 'gumbel_r', 'invgauss', 'logistic',
                'lognorm', 'nakagami', 'norm', 'rayleigh', 'weibull_min',
            ])
        with open('../config/dist_repara.json', 'r') as f:
            dist_config = json.load(f)
        idx_valid = np.array([dist in dist_config for dist in dist_cands])
        # Delete candidates that has no config
        if not all(idx_valid):
            warnings.warn(f'Distribution {dist_cands[~idx_valid]} is not '
                          'included in dist_repara.json and will be ignored')
            dist_cands = dist_cands[idx_valid]
        self.dist_cands = dist_cands

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
        ''' Re-parameterize tail based on MRP and construct environmental contour 
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
    
    def _fit_marginalX(self):
        ''' Fit marginal distribution for x using Univariate object
            Variables added:
            ----------------
                self.x_dist: Univariate object
        '''
        x_dist = Univariate(
            self.x_data,
            sample_coor=np.linspace(0, 2*self.x_data.max(), 1000))
        x_dist.fit(
            maxima_extract='Annual', maxima_fit='GumbelChart',
            method_bulk='Empirical', outlier_detect=False, verbose=False)
        self.x_dist = x_dist

    def _fit_marginalY(self):
        ''' Fit marginal distribution for y using Univariate object
            Variables added:
            ----------------
                self.y_dist: Univariate object
        '''
        y_dist = Univariate(
            self.y_data,
            sample_coor=np.linspace(0, 2*self.y_data.max(), 1000))
        y_dist.fit(
            maxima_extract='Annual', maxima_fit='GumbelChart',
            method_bulk='Empirical', outlier_detect=False, verbose=False)
        self.y_dist = y_dist

    def _fit_condY_disc(self):
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
            cur_dist.fit(
                maxima_extract='Annual', maxima_fit='GumbelChart',
                method_bulk='Empirical', outlier_detect=False, verbose=False)
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

class Univariate:
    ''' Univariate random variable object with accurate tail extrapolation 
            The right tail is extrapolated, assuming a right-skewed distribution
        Parameters
        ----------
            data: pandas series or numpy array. If constructing conditional distribution, 
                NaN should be retained for total_year to be correctly inferred
            year: numpy array with the same size as data. Ignored when data is pandas series
            total_year: float. Ignored when data is pandas series
            sample_coor: numpy array. Sample coordinate as a reference for outputs. 
                Inferred from data if not provided

        Examples
        --------
            urv = joint_prob.TailExtrap(series)
            urv.fit()
            urv.plot_diagnosis()
    '''

    def __init__(self, data, year=None, total_year=None, sample_coor=None):
        def infer_total_year(series):
            ''' Infer total years from a pandas series 
            based on starting and ending days in that year '''
            first_year_ratio = 1 - \
                (series.index[0].dayofyear - 1) / \
                days_in_year(series.index[0].year)
            last_year_ratio = series.index[-1].dayofyear / \
                days_in_year(series.index[-1].year)
            total_year = series.index[-1].year - series.index[0].year - 1 +\
                first_year_ratio + last_year_ratio
            return total_year, first_year_ratio, last_year_ratio

        def days_in_year(year):
            ''' Total days in a year '''
            return 366 if calendar.isleap(year) else 365

        if isinstance(data, pd.Series):
            self.total_year, first_year_ratio, last_year_ratio = infer_total_year(
                data)
            idx_valid = ~np.isnan(data.values)
            self.data = data.values[idx_valid]
            self.year = np.array(data.index.year)[idx_valid]
        elif isinstance(data, np.ndarray):
            if year is None or total_year is None:
                raise AttributeError(
                    'year, total_year need to be provided when data is a numpy array')
            if len(data) != len(year):
                raise AttributeError(
                    'year should be of the same length as data')
            idx_valid = ~np.isnan(data)
            self.data = data[idx_valid]
            self.year = year[idx_valid]
            self.total_year = total_year
        else:
            raise TypeError(
                'data should be either pandas series or numpy array')

        if np.any(np.diff(self.year) < 0):
            raise AttributeError('year should be in ascending order')

        if sample_coor is None:
            # Default value: starts from 0.5 * minima, ends at 1.5 * maxima, 1000 samples
            self.sample_coor = np.linspace(
                self.data.min() * 0.5, self.data.max() * 1.5, 1000)
        else:
            self.sample_coor = sample_coor

        # Discard first or end year if missing too much
        ratio_threshold = 0.7
        if first_year_ratio < ratio_threshold:
            idx_valid = self.year != self.year[0]
            self.data = self.data[idx_valid]
            self.year = self.year[idx_valid]
            self.total_year -= 1
        if last_year_ratio < ratio_threshold:
            idx_valid = self.year != self.year[-1]
            self.data = self.data[idx_valid]
            self.year = self.year[idx_valid]
            self.total_year -= 1

        self.notebook_backend = matplotlib.get_backend(
        ) in ['module://ipykernel.pylab.backend_inline']

    def fit(self, maxima_extract='Annual', maxima_fit='GumbelChart', method_bulk='Empirical',
            outlier_detect=False, verbose=True):
        ''' Fit a univirate distribution using the bulk and tail of the data respectively
            The bulk fitting is controled by method_bulk
            The tail fitting is converted from the result of the maxima subset using Extreme Value Theory 
            The conversion is achieved through fitting MRP ratio
            Parameters:
            -----------
                method_maxima: Currently only 'Annual' is supported, which is to use annual maxima
                method_tail: Currently only 'GumbelChart' is supported, which is to fit the annual maxima 
                    using Gumbel chart
                method_bulk: Either 'Empirical' or 'BestFit'
                    'Empirical' is to use the empirical CDF
                    'BestFit' is to choose the best fitting distribution from a set of candidates using chi-square
                outlier_detect: bool. Whether to assume outliers when fitting annual maxima using Gumbel chart
                verbose: bool. Whether to print progress
        '''
        total_steps = 3
        fig = plt.figure(figsize=(16, 8), tight_layout=True)

        # Fit bulk
        if verbose:
            print(f'Step 1/{total_steps}: Fitting bulk of the data')
        if method_bulk == 'Empirical':
            self._bulk_empirical_fit()
        elif method_bulk == 'BestFit':
            self._bulk_best_fit(verbose)
        else:
            raise AttributeError(
                'Unsupported bulk fitting method, check method_bulk')

        # Fit right tail
        if verbose:
            print(f'Step 2/{total_steps}: Fitting right tail')
        tail_right = _TailExtrapolation(self, fig_handle=fig)
        tail_right.fit(maxima_extract=maxima_extract,
                       maxima_fit=maxima_fit, outlier_detect=outlier_detect)

        # Fit left tail
        if verbose:
            print(f'Step 3/{total_steps}: Fitting left tail')
        tail_left = _TailExtrapolation(self, left_tail=True, fig_handle=fig)
        with np.errstate(over='ignore'):
            tail_left.fit(maxima_extract=maxima_extract,
                          maxima_fit=maxima_fit, outlier_detect=outlier_detect)

        # Arrange diagnostic plot
        fig.axes[0].change_geometry(2, 4, 1)
        fig.axes[2].change_geometry(2, 4, 2)
        fig.axes[3].change_geometry(2, 4, 5)
        fig.axes[5].change_geometry(2, 4, 6)
        fig.axes[4].remove()
        fig.axes[1].remove()

        # Combine tail and bulk
        sample_F = np.copy(self.bulk_F)
        idx_right_tail = self.sample_coor >= tail_right.threshold
        sample_F[idx_right_tail] = tail_right.tail_F[idx_right_tail]
        idx_left_tail = self.sample_coor <= -tail_left.threshold
        sample_F[idx_left_tail] = 1 - tail_left.tail_F[idx_left_tail]
        self.sample_F = sample_F
        for attr in ['c_rate', 'm_rate']:
            setattr(self, attr, getattr(tail_right, attr))
        with np.errstate(divide='ignore'):
            self.sample_mrp = 1 / self.c_rate / (1 - sample_F)

        # Diagnositc plot
        plt.subplot(1, 4, (3, 4))
        mrp_emp = 1 / self.c_rate / (
            1 - Univariate._plotting_position(self.data, method='unbiased'))
        plt.plot(mrp_emp, np.sort(self.data), '.', color=[0.6, 0.6, 0.6],
                 markersize=8, label='Empirical')
        plt.xscale('log')
        idx_tail = (self.sample_coor >= tail_right.threshold) | (
            self.sample_coor <= -tail_left.threshold)
        sample_mrp_tail = np.copy(self.sample_mrp)
        sample_mrp_tail[~idx_tail] = np.nan
        plt.plot(sample_mrp_tail, self.sample_coor, 'b-', label='Tail fit')
        xlm = list(plt.xlim())
        xlm[1] = mrp_emp[-1] * 10 # Limit MRP to be 10 * data period
        ylm = list(plt.ylim())
        with np.errstate(invalid='ignore'):
            ylm[1] = self.sample_coor[sample_mrp_tail < xlm[1]][-1] # Corresponding y
        plt.plot(1 / self.c_rate / (1 - self.bulk_F), self.sample_coor, '--',
                 color=[0, 0.5, 0])
        plt.plot(1 / self.c_rate / (1 - self.bulk_F[~idx_tail]), self.sample_coor[~idx_tail], '-',
                 color=[0, 0.5, 0], label='Bulk fit')
        plt.plot(xlm, tail_right.threshold * np.array([1, 1]), 'k--')
        plt.plot(xlm, -tail_left.threshold * np.array([1, 1]), 'k--')
        plt.xlim(xlm)
        plt.ylim(ylm)
        plt.xlabel('Return period (year)')
        plt.ylabel('X')
        plt.title('Fitting result')
        plt.grid(True, which='both')
        plt.legend(loc='upper left')
        self.diag_fig = fig
        if self.notebook_backend:
            plt.close(fig)

    def predict(self, mrp=None, val=None):
        ''' Predict value given an MRP, or MRP given a value
            Parameters:
            -----------
                MRP: float, mean return period for prediction
                val: float, value of the variable
            Returns:
            --------
                Value corresponding to MRP (if MRP is input), or MRP corresponding to val
            Note:
            -----
                Only one of MRP and val should be provided
        '''
        if mrp is not None and val is None:  # MRP provided
            idx = np.isnan(self.sample_mrp)
            return interp1d(self.sample_mrp[~idx], self.sample_coor[~idx])(mrp)
        elif mrp is None and val is not None:  # val provided
            return interp1d(self.sample_coor, self.sample_mrp)(val)
        else:
            raise AttributeError('Only one of MRP and val should be provided')

    def plot_diagnosis(self):
        ''' Display diagnostic plot
        '''
        if hasattr(self, 'diag_fig'):
            if self.notebook_backend:
                display(self.diag_fig)
            else:
                plt.show()
        else:
            raise AttributeError(
                'No diagnostic plot found. Call fit method first.')

    def _bulk_empirical_fit(self):
        ''' Fit bulk using empirical CDF
            Variables added:
            ----------------
                self.bulk_F: CDF corresponding to self.sample_coor
        '''
        x = np.sort(self.data)
        F_emp = Univariate._plotting_position(x, method='unbiased')
        if self.sample_coor[0] < x[0]:
            x = np.concatenate(([0], x))
            F_emp = np.concatenate(([0], F_emp))
        self.bulk_F = interp1d(
            x, F_emp, bounds_error=False)(self.sample_coor)

    def _bulk_best_fit(self, verbose):
        ''' Fit bulk using optimal distribution (see best_fit for more information)
            Variables added:
            ----------------
                self.bulk_F: CDF corresponding to self.sample_coor
        '''
        ss = StandardScaler().fit(self.data.reshape(-1, 1))
        data_std = ss.transform(self.data.reshape(-1, 1)).flatten()
        fit_df = Univariate.best_fit(data_std)
        self.bulk_F = getattr(stats, fit_df.Distribution[0]).cdf(
            ss.transform(self.sample_coor.reshape(-1, 1)),
            *fit_df.param[0][:-2],
            loc=fit_df.param[0][-2],
            scale=fit_df.param[0][-1]).flatten()
        if verbose:
            print(f'          Best fit distribution: {fit_df.Distribution[0]}')

    @staticmethod
    def best_fit(data, dist_names=None, qq_plot=False, dist_config=None):
        ''' Search for best distribution fitting based on chi-squar test
            List for scipy distribution: 
                https://docs.scipy.org/doc/scipy/reference/stats.html
            Regarding chi-squre test:
                https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
        '''
        warnings.filterwarnings("ignore")
        if dist_names is None:
            dist_names = [
                'expon', 'gumbel_l', 'gumbel_r', 'logistic', 'norm', 'rayleigh',  # 2 para
                'exponnorm', 'fatiguelife', 'gamma', 'genextreme', 'genlogistic',  # 3 para
                'invgamma', 'invgauss', 'lognorm', 'nakagami',  # 3 para
                'genpareto', 't', 'weibull_max', 'weibull_min',  # 3 para
                'exponweib', 'beta', 'burr',  # 4 para
            ]  # 20 common distributions

        # Prepare observation data for the chi-square test
        number_of_bins = 20
        observed_values, bin_edges = np.histogram(data, bins=number_of_bins)

        # Prepare quantile for the customized method to calculate R2 from Q-Q plot
        val_true = np.linspace(data.min(), data.max(), 200)[
            1:-1]  # Skip end to avoid inf
        quantile = [(data <= val).sum() / len(data) for val in val_true]

        chi_square, r2, params = [], [], []
        for dist_name in dist_names:
            # Fit distribution
            dist = getattr(stats, dist_name)
            if dist_config is None:
                kwargs = {}
            else:
                kwargs = dist_config[dist_name]['fit_kwargs']
            param = dist.fit(data, **kwargs)
            params.append(param)

            # calculate chi-squared
            cdf = dist.cdf(
                bin_edges, *param[:-2], loc=param[-2], scale=param[-1])
            expected_values = len(data) * np.diff(cdf)
            chi_square.append(
                stats.chisquare(observed_values, expected_values,
                                ddof=len(param)-2)[0]
            )  # loc and scale not included

            # Customized way to quickly calculate R2 from Q-Q plot
            val_pred = dist.ppf(
                quantile, *param[:-2], loc=param[-2], scale=param[-1])
            r2.append(r2_score(val_true, val_pred))

        # Collate results and sort by goodness of fit (best at top)
        fit_df = pd.DataFrame({
            'Distribution': dist_names,
            'chi_square': chi_square,
            'r2': r2,
            'param': params,
        })
        fit_df = fit_df.sort_values(
            ['r2'], ascending=False).reset_index(drop=True)
        # fit_df = fit_df.sort_values(['chi_square'], ascending=True).reset_index(drop=True)

        if qq_plot:
            plt.figure(figsize=(16, 8))
            for idx in range(min(7, len(fit_df))):
                plt.subplot(2, 4, idx+1)
                stats.probplot(
                    data, sparams=fit_df.param[idx],
                    dist=fit_df.Distribution[idx], plot=plt)
                plt.title(fit_df.Distribution[idx])
                plt.tight_layout()
            plt.show()
        return fit_df

    @staticmethod
    def _plotting_position(data, method='unbiased'):
        ''' Plotting position of data (with NaN discarded) '''
        assert not np.any(np.isnan(data)), 'data should not include any NaN'
        n = len(data)
        if method == 'unbiased':
            return ((np.arange(n) + 1) - 0.44) / (n + 0.12)
        elif method == 'simple':
            return (np.arange(n) + 1) / (n + 1)
        else:
            raise AttributeError(
                'Unsupported calculation method for plotting position')


class _TailExtrapolation:
    ''' Extrapolation the right tail of a distribution
        Parameters
        ----------
            data: pandas series or numpy array. If constructing conditional distribution, 
                NaN should be retained for total_year to be correctly inferred
            year: numpy array with the same size as data. Ignored when data is pandas series
            total_year: float. Ignored when data is pandas series
            sample_coor: numpy array. Sample coordinate as a reference for outputs. 
                Inferred from data if not provided
    '''

    def __init__(self, univariate_obj, left_tail=False, fig_handle=None):
        for attr in ['data', 'year', 'total_year', 'sample_coor', 'bulk_F']:
            setattr(self, attr, getattr(univariate_obj, attr))
        if left_tail:
            self.data = -self.data
            self.sample_coor = -self.sample_coor
            self.bulk_F = 1 - self.bulk_F
            self.label = 'left'
        else:
            self.label = 'right'
        if fig_handle is None:
            self.diag_fig = plt.figure(figsize=(8, 3), tight_layout=True)
        else:
            self.diag_fig = fig_handle

    def fit(self, maxima_extract='Annual', maxima_fit='GumbelChart', outlier_detect=False):
        # Extract maxima (sorted, no NaN)
        if maxima_extract == 'Annual':
            self._extract_annual_maxima()
        else:
            raise AttributeError(
                'Unsupported maxima extraction method, check method_maxima')

        if maxima_fit == 'GumbelChart':
            self._fit_gumbel_chart(outlier_detect, plot_diagnosis=True)
        else:
            raise AttributeError(
                'Unsupported tail fitting method, check method_tail.')

        # Fitting tail
        self._maxima_to_continuous(plot_diagnosis=True)

    def _extract_annual_maxima(self) -> None:
        ''' Extract annual maxima 
            Variables added:
            ----------------
                self.maxima: numpy array in ascending order
        '''
        unique_year = np.unique(self.year)
        result = []
        for year in unique_year:
            result.append(max(self.data[self.year == year]))
        self.maxima = np.sort(result)

    def _fit_gumbel_chart(self, outlier_detect: bool, plot_diagnosis: bool):
        ''' Fit a Gumbel distribution fit via Gumbel chart 
            Variables added:
            ----------------
                self.maxima_inlier_mask: Mask indicating inliers
                self.maxima_dist: Probability distribution for the maxima
                self.threshold: Threshold of X between bulk and tail, minimum value is 
                    constrained to be no lower than 5 percentile of F_maxima
            Parameters:
            -----------
                outlier_detect: Whether to assume the existance of outliers. 
                    Use OLS when False
                plot_diagnosis: Whether to generate diagnostic plot
        '''
        def _gumbel_y(F):
            ''' Calculate y coordinates on the Gumbel chart from CDF '''
            return -np.log(-np.log(F))

        x = self.maxima
        F = Univariate._plotting_position(x, method='unbiased')
        y = _gumbel_y(F)
        if outlier_detect:
            # ToDo: Test different model on condY for dataset ABC
            mdl = linear_model.RANSACRegressor(
                random_state=1).fit(x.reshape(-1, 1), y)
            self.maxima_inlier_mask = mdl.inlier_mask_
            mdl = mdl.estimator_

#             mdl = linear_model.HuberRegressor(
#                 epsilon=1.35).fit(x.reshape(-1, 1), y)
#             self.maxima_inlier_mask = np.array(
#                 [True] * len(self.maxima))  # Create mask manually
        else:
            mdl = linear_model.LinearRegression().fit(x.reshape(-1, 1), y)
            self.maxima_inlier_mask = np.array(
                [True] * len(self.maxima))  # Create mask manually
        k, b = mdl.coef_[0], mdl.intercept_

        if plot_diagnosis:
            ax = self.diag_fig.add_subplot(1, 3, 1, label=self.label)
            ax.plot(x[self.maxima_inlier_mask], y[self.maxima_inlier_mask],
                    'b.', markersize=10, label='Maxima(inliers)')
            ax.plot(x[~self.maxima_inlier_mask], y[~self.maxima_inlier_mask],
                    'r.', markersize=10, label='Maxima(outliers)')
            xlm, ylm = ax.get_xlim(), ax.get_ylim()
            ax.plot(self.sample_coor, mdl.predict(self.sample_coor.reshape(-1, 1)),
                    'r--', label='Linear fitting')
            ax.set_xlim(xlm)
            ax.set_ylim(ylm)
            ax.set_xlabel('Maxima data')
            ax.set_ylabel('$-ln(-ln(F))$')
            ax.set_title(f'Gumbel chart ({self.label} tail)')
            ax.grid(True)
            ax.legend(loc='best')

        self.maxima_dist = stats.gumbel_r(loc=-b/k, scale=1/k)
        self.maxima_inlier_mask[self.maxima <
                                self.maxima_dist.ppf(0.05)] = False
        self.threshold = self.maxima[self.maxima_inlier_mask].min()

    def _maxima_to_continuous(self, plot_diagnosis: bool):
        # Calculate empirical MRP for continuous and maxima datasets
        c_data = np.sort(self.data)
        m_data = self.maxima
        c_rate = len(c_data) / self.total_year
        m_rate = len(m_data) / self.total_year
        c_mrp_emp = 1 / c_rate / \
            (1 - Univariate._plotting_position(c_data, method='unbiased'))
        m_mrp_emp = 1 / m_rate / \
            (1 - Univariate._plotting_position(m_data, method='unbiased'))

        # Calculate empirical MRP ratio
        mrp_ratio_emp = m_mrp_emp / interp1d(c_data, c_mrp_emp)(m_data)

        # Calculate the corresponding t coordinates for the empirical MRP ratio
        t_emp = -np.log(self.maxima_dist.cdf(m_data))

        # Target MRP ratio at self.threshold
        t_threshold = -np.log(self.maxima_dist.cdf(self.threshold))
        c_mrp_threshold = 1 / c_rate / \
            (1 - interp1d(self.sample_coor, self.bulk_F)(self.threshold))
        m_mrp_threshold = 1 / m_rate / \
            (1 - self.maxima_dist.cdf(self.threshold))
        mrp_ratio_threshold = m_mrp_threshold / c_mrp_threshold

        # Prepare fitting data
        # Maximum data yields incorrect MRP ratio (always 1)
        self.maxima_inlier_mask[-1] = False
        # Exclude data below threshold
        self.maxima_inlier_mask[self.maxima < self.threshold] = False
        t_emp = t_emp[self.maxima_inlier_mask]
        mrp_ratio_emp = mrp_ratio_emp[self.maxima_inlier_mask]
        t_emp = np.concatenate((t_emp, [t_threshold]))  # Append threshold
        mrp_ratio_emp = np.concatenate((mrp_ratio_emp, [mrp_ratio_threshold]))
        sigma = np.ones(t_emp.shape)
        sigma[-1] = 1 / len(sigma)  # Set the threshold point for more weight

        # Fitting MRP ratio ~ t
        def func(t, a, b, c):
            return (a * t + c) ** b
        popt, _ = curve_fit(func, t_emp, mrp_ratio_emp, bounds=(
            [0, 0, 1], np.inf), sigma=sigma, max_nfev=1e4)

        # Convert tail MRP
        m_sample_F = self.maxima_dist.cdf(self.sample_coor)
        m_sample_F[self.sample_coor < self.threshold] = np.nan
        with np.errstate(divide='ignore'):
            m_sample_mrp = 1 / m_rate / (1 - m_sample_F)
        c_sample_mrp = m_sample_mrp / func(-np.log(m_sample_F), *popt)
        c_sample_F = 1 - 1 / c_rate / c_sample_mrp

        # Record results
        self.m_rate = m_rate
        self.c_rate = c_rate
        self.tail_F = c_sample_F

        if plot_diagnosis:
            # MRP ratio fitting
            ax = self.diag_fig.add_subplot(1, 3, 2, label=self.label)
            sample_t = np.linspace(0, 3.5, 100)
            ax.plot(t_emp[:-1], mrp_ratio_emp[:-1], 'k.',
                    markersize=8, label='Empirical')
            ax.plot(t_threshold, mrp_ratio_threshold, 'rx',
                    markersize=10, label='Connecting point')
            ax.plot(sample_t, func(sample_t, *popt), 'r-', label='Fit')
            ax.set_xlim([0, 3.5])
            ax.set_xlabel('t(X)')
            ax.set_ylabel('MRP ratio')
            ax.set_title(f'MRP ratio ({self.label} tail)')
            ax.grid(True)
            ax.legend(loc='lower right')

            # Maxima to continuous conversion
            ax = self.diag_fig.add_subplot(1, 3, 3, label=self.label)
            ax.plot(m_mrp_emp, m_data, '.', color=[
                    1, 0.4, 0.4], markersize=8, label='Maxima')
            ax.plot(c_mrp_emp[c_data >= m_data.min()],
                    c_data[c_data >= m_data.min()], '.', color=[0.4, 0.4, 1],
                    markersize=8, label='Continuous')
            ax.set_xscale('log')
            xlm = ax.get_xlim()
            ylm = ax.get_ylim()
            ax.plot(m_sample_mrp, self.sample_coor, 'r-', label='Maxima fit')
            ax.plot(c_sample_mrp, self.sample_coor,
                    'b-', label='Continuous fit')
            ax.plot(xlm, self.threshold * np.array([1, 1]), 'k--')
            ax.set_xlim(xlm)
            ax.set_ylim([m_data.min(), ylm[1]])
            ax.set_xlabel('Return period (year)')
            ax.set_ylabel('X')
            ax.set_title(f'Tail extrap. ({self.label} tail)')
            ax.grid(True, which='both')
            ax.legend(loc='upper left')
            # if self.notebook_backend:
            #     plt.close()


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
    with open('../datasets/D.pkl', 'rb') as f:
        df = pickle.load(f)
    # df = pd.read_csv('../datasets/D.txt', sep=';', index_col=0, parse_dates=True)

    # test = Univariate(df.iloc[:, 0])
    # test.fit()
    # test.plot_diagnosis()

    # x_dist = Univariate(df.iloc[:, 0])
    # x_dist.fit()
    # x_dist.plot_diagnosis()
    # print(dir(te))

    # Test for Multivariate class
    test = Multivariate(df, condY_x=np.arange(1, 22))
    test.fit()
