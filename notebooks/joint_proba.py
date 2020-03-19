import calendar
import copy
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from functools import partial
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class Multivariate:
    def __init__(self, df, col_x=0, col_y=1, condY_x=None):
        self.x = df.iloc[:, col_x]
        self.y = df.iloc[:, col_y]
        if condY_x is None:
            self.condY_x = np.linspace(x.min(), x.max(), 10)
        else:
            self.condY_x = condY_x
        self.condY_dx = np.diff(condY_x).mean()
        

    def fit(self, plot_diagnosis=True, verbose=True):
        ''' Fit results that are independent from MRP 
            Variables added:
            ----------------
                self.x_pd, y_pd, condY_pd: Univariate objects for marginal X & Y, 
                    and conditional Y
                self.condYs_bulk: list of _CondY object as candidates of condY fitting
                    results using the bulk of the data
        '''

        def get_condY_para_bulk():
            ''' Fit distribution for condY at various x 
                Returns:
                --------
                    df: Dataframe of different distributions with averaged chi_square, 
                        averaged r2, and parameters for the corresponding x.
                Notes:
                ------
                    Only distributions with chi_square lower than 2 * optimal result 
                        are returned
            '''
            df_temp = pd.DataFrame()
            for condY_pd in self.condY_pd:
                data = condY_pd.data
                df_cur = Univariate.best_fit(
                    data, dist_names=dist_names, dist_config=dist_config)
                df_temp = pd.concat([df_temp, df_cur.set_index('Distribution')], axis=1)

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
                columns={0:'chi_square', 1:'r2', 2:'param'}
            ).sort_values(by='chi_square')

            # Select candidates based on chi_square
            df = df[df['chi_square'] < df['chi_square'][0] * 2]
            return df

        total_steps = 5

        # Distributions for re-parameterization and their configs
        dist_names = np.array([
            'burr12', 'expon', 'fatiguelife', 'gamma', 'genextreme', 
            'genpareto', 'gumbel_r', 'invgauss', 'logistic', 
            'lognorm', 'nakagami', 'norm', 'rayleigh', 'weibull_min',
        ]) # Candidates for re-para
        with open('../config/dist_repara.json', 'r') as f: 
            dist_config = json.load(f) 
        idx_valid = np.array([dist in dist_config for dist in dist_names])
        # Delete candidates that has no config
        if not all(idx_valid):
            warnings.warn(f'Distribution {dist_names[~idx_valid]} is not included in '
                        'dist_repara.json and will be ignored')
            dist_names = dist_names[idx_valid]
        
        # Fit marginal X
        if verbose:
            print(f'Step 1/{total_steps}: Fitting marginal X')
        x_pd = Univariate(self.x, sample_coor=np.linspace(0, 2*self.x.max(), 1000))
        x_pd.fit(maxima_extract='Annual', maxima_fit='GumbelChart', method_bulk='Empirical', 
                 outlier_detect=False, verbose=False)
        self.x_pd = x_pd
        
        # Fit marginal Y
        if verbose:
            print(f'Step 2/{total_steps}: Fitting marginal Y')
        y_pd = Univariate(self.y, sample_coor=np.linspace(0, 2*self.y.max(), 1000))
        y_pd.fit(maxima_extract='Annual', maxima_fit='GumbelChart', method_bulk='Empirical', 
                 outlier_detect=False, verbose=False)
        self.y_pd = y_pd
        
        # Fit conditional Y
        if verbose:
            print(f'Step 3/{total_steps}: Fitting individual conditional Y')
        condY_pd = []
        for cur_x in self.condY_x:
            condY_data = self.y.copy()
            condY_data[(self.x < cur_x - self.condY_dx) | (self.x > cur_x + self.condY_dx)] = np.nan
            cur_pd = Univariate(condY_data, sample_coor=np.linspace(0, 2*self.x.max(), 1000))
            cur_pd.fit(maxima_extract='Annual', maxima_fit='GumbelChart', method_bulk='Empirical', 
                     outlier_detect=False, verbose=False)
            condY_pd.append(cur_pd)
        self.condY_pd = condY_pd
        
        # Fit the parameters of conditional Y using bulk of the data
        if verbose:
            print(f'Step 4/{total_steps}: Finding distributions to fit conditional Y')
        df = get_condY_para_bulk()
        condYs = []
        for idx in range(len(df)):
            condY = _CondY(
                dist_name=df.index[idx], x=self.condY_x, 
                params=df['param'][idx], dist_config=dist_config[df.index[idx]])
            condY.fit()
            condYs.append(condY)
        self.condYs_bulk = condYs
        
    def predict(self, MRPs):
        ''' Re-parameterize tail based on MRP and construct environmental contour 
            Parameters:
            -----------
                MRP: numpy array. Target MRP
        '''
        
        def get_condY_F(x_pd, beta, x):
            ''' Return F of condY given beta '''
            x_F = interp1d(x_pd.sample_coor, x_pd.sample_F)(x)
            x_beta = std_norm.ppf(x_F)
            y_beta_square = beta ** 2 - x_beta ** 2
            y_beta_square[y_beta_square < 0] = np.nan
            y_beta = np.sqrt(y_beta_square)
            y_F = std_norm.cdf(y_beta)
            return y_beta
        
        std_norm = stats.norm()

        for MRP in MRPs:
            beta = std_norm.ppf(1 - 1/self.x_pd.c_rate/MRP)
            
            # MRP of independent Y for validation
            y_mrp = self.y_pd.predict(MRP=MRP)

            # Jagged contour
            condY_F = get_condY_F(self.x_pd, beta, self.condY_x)

            # Determine range of re-parameterization

            # Upper contour

            # Lower contour

            # Combine result
    
    
        
    def plot_diagnosis(self):
        def plot_pd_diagnosis():
            if ' ' in dropdown_pd.value: # contains list index
                attr_name, idx = dropdown_pd.value.split(sep=' ')
                pd = getattr(self, attr_name)[int(idx)]
            else:
                pd = getattr(self, dropdown_pd.value)
            display(pd.diag_fig)

        def update_pd_plot(change):
            pd_display.clear_output(wait=True)
            with pd_display:
                plot_pd_diagnosis()
                plt.show()

        # Tab 1: Univirate fitting
        dropdown_options = [('Marginal X', 'x_pd'), ('Marginal Y', 'y_pd')] +\
            [('Conditional Y at X={:.1f}'.format(condY_x), 'condY_pd {}'.format(idx)) 
             for idx, condY_x in enumerate(self.condY_x)]
        dropdown_pd = widgets.Dropdown(options=dropdown_options, description='Item')
        dropdown_pd.observe(update_pd_plot, names="value")
        pd_display = widgets.Output()
        with pd_display:
            plot_pd_diagnosis()
            plt.show()
        tab1 = widgets.VBox(children=[dropdown_pd, pd_display])

        # Tab 2: Multivirate fitting
        tab2 = widgets.VBox(children=[])

        tab = widgets.Tab(children=[tab1, tab2])
        tab.set_title(0, 'Univariate fitting')
        tab.set_title(1, 'Multivariate fitting')
        return widgets.VBox(children=[tab])


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
        if isinstance(data, pd.Series):
            self.total_year, first_year_ratio, last_year_ratio = Univariate._infer_total_year(data)
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
            raise TypeError('data should be either pandas series or numpy array')
            
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

        self.notebook_backend = matplotlib.get_backend() in ['module://ipykernel.pylab.backend_inline']

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
            raise AttributeError('Unsupported bulk fitting method, check method_bulk')
        
        # Fit right tail
        if verbose:
            print(f'Step 2/{total_steps}: Fitting right tail')
        tail_right = _TailExtrapolation(self, fig_handle=fig)
        tail_right.fit(maxima_extract=maxima_extract, maxima_fit=maxima_fit, outlier_detect=outlier_detect)
        
        # Fit left tail
        if verbose:
            print(f'Step 3/{total_steps}: Fitting left tail')
        tail_left = _TailExtrapolation(self, left_tail=True, fig_handle=fig)
        with np.errstate(over='ignore'):
            tail_left.fit(maxima_extract=maxima_extract, maxima_fit=maxima_fit, outlier_detect=outlier_detect)

        # Arrange diagnostic plot
        fig.axes[0].change_geometry(2,4,1)
        fig.axes[2].change_geometry(2,4,2)
        fig.axes[3].change_geometry(2,4,5)
        fig.axes[5].change_geometry(2,4,6)
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
        plt.subplot(1,4,(3,4))
        plt.plot(1 / self.c_rate / (1 - Univariate._plotting_position(self.data, method='unbiased')), 
                    np.sort(self.data), '.', color=[0.6, 0.6, 0.6], 
                    markersize=8, label='Empirical')
        plt.xscale('log')
        idx_tail = (self.sample_coor >= tail_right.threshold) | (self.sample_coor <= -tail_left.threshold)
        sample_mrp_tail = np.copy(self.sample_mrp)
        sample_mrp_tail[~idx_tail] = np.nan
        plt.plot(sample_mrp_tail, self.sample_coor, 'b-', label='Tail fit')
        xlm = plt.xlim()
        ylm = plt.ylim()
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
        plt.grid(True)
        plt.legend(loc='upper left')
        self.diag_fig = fig
        if self.notebook_backend:
            plt.close(fig)

    def predict(self, MRP=None, val=None):
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
        if MRP is not None and val is None: # MRP provided
            idx = np.isnan(self.sample_mrp)
            return interp1d(self.sample_mrp[~idx], self.sample_coor[~idx])(MRP)
        elif MRP is None and val is not None: # val provided
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
            raise AttributeError('No diagnostic plot found. Call fit method first.')

    
    
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
    def best_fit(data, dist_names=None, qq_plot=False, dist_config={}):
        ''' Search for best distribution fitting based on chi-squar test
            List for scipy distribution: 
                https://docs.scipy.org/doc/scipy/reference/stats.html
            Regarding chi-squre test:
                https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
        '''
        warnings.filterwarnings("ignore")
        if dist_names is None:
            dist_names= [
                'expon', 'gumbel_l', 'gumbel_r', 'logistic', 'norm', 'rayleigh', # 2 para
                'exponnorm', 'fatiguelife', 'gamma', 'genextreme', 'genlogistic', # 3 para
                'invgamma', 'invgauss', 'lognorm', 'nakagami', # 3 para
                'genpareto', 't', 'weibull_max', 'weibull_min', # 3 para
                'exponweib', 'beta', 'burr', # 4 para
            ] # 20 common distributions

        # Prepare observation data for the chi-square test
        number_of_bins = 20
        observed_values, bin_edges = np.histogram(data, bins=number_of_bins)

        # Prepare quantile for the customized method to calculate R2 from Q-Q plot
        val_true = np.linspace(data.min(), data.max(), 200)[1:-1] # Skip end to avoid inf
        quantile = [(data <= val).sum() / len(data) for val in val_true]    

        chi_square, r2, params = [], [], []
        for dist_name in dist_names:
            # Fit distribution
            dist = getattr(stats, dist_name)
            if dist_config:
                kwargs = dist_config[dist_name]['fit_kwargs']
            else:
                kwargs = {}
            param = dist.fit(data, **kwargs)
            params.append(param)
            
            # calculate chi-squared
            cdf = dist.cdf(bin_edges, *param[:-2], loc=param[-2], scale=param[-1])
            expected_values = len(data) * np.diff(cdf)
            chi_square.append(
                stats.chisquare(observed_values, expected_values, ddof=len(param)-2)[0]
            ) # loc and scale not included

            # Customized way to quickly calculate R2 from Q-Q plot
            val_pred = dist.ppf(quantile, *param[:-2], loc=param[-2], scale=param[-1])
            r2.append(r2_score(val_true, val_pred))

        # Collate results and sort by goodness of fit (best at top)
        fit_df = pd.DataFrame({
            'Distribution': dist_names,
            'chi_square': chi_square,
            'r2': r2,
            'param': params,
        })
        fit_df = fit_df.sort_values(['r2'], ascending=False).reset_index(drop=True)
        # fit_df = fit_df.sort_values(['chi_square'], ascending=True).reset_index(drop=True)

        if qq_plot:
            plt.figure(figsize=(16, 8))
            for idx in range(min(7, len(fit_df))):
                plt.subplot(2,4,idx+1)
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

    @staticmethod
    def _infer_total_year(series):
        ''' Infer total years from a pandas series 
        based on starting and ending days in that year '''
        first_year_ratio = 1 - \
            (series.index[0].dayofyear - 1) / \
            Univariate._days_in_year(series.index[0].year)
        last_year_ratio = series.index[-1].dayofyear / \
            Univariate._days_in_year(series.index[-1].year)
        total_year = series.index[-1].year - series.index[0].year - 1 +\
            first_year_ratio + last_year_ratio
        return total_year, first_year_ratio, last_year_ratio

    @staticmethod
    def _days_in_year(year):
        ''' Total days in a year '''
        return 366 if calendar.isleap(year) else 365


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
            raise AttributeError('Unsupported maxima extraction method, check method_maxima')
        
        if maxima_fit == 'GumbelChart':
            self._fit_gumbel_chart(outlier_detect, plot_diagnosis=True)
        else:
            raise AttributeError('Unsupported tail fitting method, check method_tail.')
        
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
                self.maxima_pd: Probability distribution for the maxima
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
            mdl = linear_model.RANSACRegressor(random_state=1).fit(x.reshape(-1, 1), y)
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
            ax = self.diag_fig.add_subplot(1,3,1, label=self.label)
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
            
        self.maxima_pd = stats.gumbel_r(loc=-b/k, scale=1/k)
        self.maxima_inlier_mask[self.maxima < self.maxima_pd.ppf(0.05)] = False
        self.threshold = self.maxima[self.maxima_inlier_mask].min()

    def _maxima_to_continuous(self, plot_diagnosis: bool):
        # Calculate empirical MRP for continuous and maxima datasets
        c_data = np.sort(self.data)
        m_data = self.maxima
        c_rate = len(c_data) / self.total_year
        m_rate = len(m_data) / self.total_year
        c_mrp_emp = 1 / c_rate / (1 - Univariate._plotting_position(c_data, method='unbiased'))
        m_mrp_emp = 1 / m_rate / (1 - Univariate._plotting_position(m_data, method='unbiased'))

        # Calculate empirical MRP ratio
        mrp_ratio_emp = m_mrp_emp / interp1d(c_data, c_mrp_emp)(m_data)

        # Calculate the corresponding t coordinates for the empirical MRP ratio
        t_emp = -np.log(self.maxima_pd.cdf(m_data))

        # Target MRP ratio at self.threshold
        t_threshold = -np.log(self.maxima_pd.cdf(self.threshold))
        c_mrp_threshold = 1 / c_rate / (1 - interp1d(self.sample_coor, self.bulk_F)(self.threshold))
        m_mrp_threshold = 1 / m_rate / (1 - self.maxima_pd.cdf(self.threshold))
        mrp_ratio_threshold = m_mrp_threshold / c_mrp_threshold

        # Prepare fitting data
        self.maxima_inlier_mask[-1] = False # Maximum data yields incorrect MRP ratio (always 1)
        self.maxima_inlier_mask[self.maxima < self.threshold] = False # Exclude data below threshold
        t_emp = t_emp[self.maxima_inlier_mask]
        mrp_ratio_emp = mrp_ratio_emp[self.maxima_inlier_mask]
        t_emp = np.concatenate((t_emp, [t_threshold])) # Append threshold
        mrp_ratio_emp = np.concatenate((mrp_ratio_emp, [mrp_ratio_threshold]))
        sigma = np.ones(t_emp.shape)
        sigma[-1] = 1 / len(sigma) # Set the threshold point for more weight

        # Fitting MRP ratio ~ t 
        def func(t, a, b, c):
            return (a * t + c) ** b
        popt, _ = curve_fit(func, t_emp, mrp_ratio_emp, bounds=([0, 0, 1], np.inf), sigma=sigma, max_nfev=1e4)

        # Convert tail MRP
        m_sample_F = self.maxima_pd.cdf(self.sample_coor)
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
            ax = self.diag_fig.add_subplot(1,3,2, label=self.label)
            sample_t = np.linspace(0, 3.5, 100)
            ax.plot(t_emp[:-1], mrp_ratio_emp[:-1], 'k.', markersize=8, label='Empirical')
            ax.plot(t_threshold, mrp_ratio_threshold, 'rx', markersize=10, label='Connecting point')
            ax.plot(sample_t, func(sample_t, *popt), 'r-', label='Fit')
            ax.set_xlim([0, 3.5])
            ax.set_xlabel('t(X)')
            ax.set_ylabel('MRP ratio')
            ax.set_title(f'MRP ratio ({self.label} tail)')
            ax.grid(True)
            ax.legend(loc='lower right')
            
            # Maxima to continuous conversion
            ax = self.diag_fig.add_subplot(1,3,3, label=self.label)
            ax.plot(m_mrp_emp, m_data, '.', color=[1, 0.4, 0.4], markersize=8, label='Maxima')
            ax.plot(c_mrp_emp[c_data >= m_data.min()], 
                     c_data[c_data >= m_data.min()], '.', color=[0.4, 0.4, 1], 
                     markersize=8, label='Continuous')
            ax.set_xscale('log')
            xlm = ax.get_xlim()
            ylm = ax.get_ylim()
            ax.plot(m_sample_mrp, self.sample_coor, 'r-', label='Maxima fit')
            ax.plot(c_sample_mrp, self.sample_coor, 'b-', label='Continuous fit')
            ax.plot(xlm, self.threshold * np.array([1, 1]), 'k--')
            ax.set_xlim(xlm)
            ax.set_ylim([m_data.min(), ylm[1]])
            ax.set_xlabel('Return period (year)')
            ax.set_ylabel('X')
            ax.set_title(f'Tail extrap. ({self.label} tail)')
            ax.grid(True)
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
            if len(unique_values) == 1: # Fixed parameter
                coef_func.append(partial(constant_func, c=unique_values[0]))
            else:
                popt, _ = curve_fit(
                    fitting_func, self.x, param_raw, method='trf', 
                    bounds=(coef_lb[param_lb], np.inf), max_nfev=1e4
                )
                coef_func.append(partial(fitting_func, a=popt[0], b=popt[1], c=popt[2]))
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

    x_pd = Univariate(df.iloc[:, 0])
    x_pd.fit()
    x_pd.plot_diagnosis()
    # print(dir(te))