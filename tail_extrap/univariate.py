import calendar
import warnings

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from tail_extrap import util


class Univariate:
    ''' Tail extrapolation of uni-variable as a time series
        Environmental variables (e.g., wind, wave) with short intervals are not
            dependent, as such Extreme Value Theory cannot be used for 
            distribution tail extrapolation. 
        This Univariate class applies Extreme Value Theory to the independent
            subset, and then convert the extrapolation result to the original
            dataset using MRP ratio curve. 
        Methods include fit, predict, and plot_diagnosis. During the fitting, 
            the distribution is estimated separately for the left tail, bulk, 
            and the right tail, and then combined together. Left and right tails 
            are extrapolated through the _TailExtrapolation class.
        Parameters
        ----------
            data: pandas series with datetime index. If constructing conditional 
                distribution, NaN should be retained for total year to be 
                correctly inferred
            sample_coor: numpy array. Sample coordinate for outputs reference. 
                Inferred from data if not provided
    '''

    def __init__(self, data, sample_coor=None):
        ''' Variables added:
                data: with NaN removed
                time: datetimes corresponding to data
                total_year: float, total years covered by data
                sample_coor: numpy array, coordinates of discretized variable
                notebook_backend: bool, whether this is a notebook environment
        '''
        def days_in_year(year):
            ''' Total days in a year '''
            return 366 if calendar.isleap(year) else 365

        # Sanity check
        if not isinstance(data, pd.Series):
            raise TypeError('data should be a pandas series')
        if not isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise TypeError('data index should be datetimes')
        ratio_threshold = 0.7
        data = data.sort_index()
        first_year_ratio = 1 - \
            (data.index[0].dayofyear - 1) / \
            days_in_year(data.index[0].year)
        last_year_ratio = data.index[-1].dayofyear / \
            days_in_year(data.index[-1].year)
        if min([first_year_ratio, last_year_ratio]) < ratio_threshold:
            warnings.warn('Missing to much data in the first or last year '
                'might affect accuracy')

        self.total_year = data.index[-1].year - data.index[0].year - 1 +\
            first_year_ratio + last_year_ratio
        idx_valid = ~np.isnan(data.values)
        self.data = data.values[idx_valid]
        self.time = data.index[idx_valid]
        if sample_coor is None:
            self.sample_coor = np.linspace(
                self.data.min() * 0.5, self.data.max() * 1.5, 1000)
        else:
            self.sample_coor = sample_coor
        self.notebook_backend = matplotlib.get_backend() \
            in ['module://ipykernel.pylab.backend_inline']

    def fit(self, maxima_extract='Annual', maxima_fit='GumbelChart', 
            bulk_fit='Empirical', outlier_detect=None, verbose=False):
        '''Fit a univirate distribution for the bulk and tail of the data
        Parameters:
        -----------
            maxima_extract: one of ['Annual']
                How to extract independent maxima subset.
                Annual: Annual maxima
            maxima_fit: one of ['GumbelChart']
                How to fit a distribution to the extracted maxima. 
            bulk_fit: one of ['Empirical', 'BestFit']
                How to fit a distribution to the bulk of the data.
                Empirical: Use the empirical CDF
                BestFit: Choose the best fitting distribution 
            outlier_detect: one of [None, 'RANSAC', 'Huber']
                Whether to assume outliers when fitting maxima data
                None: Assume no outliers
                RANSAC: Use RANSACRegressor to filter outliers
                Huber: Use HuberRegressor to filter outliers
            verbose: bool. Whether to print progress
        '''
        total_steps = 3
        self.diag_fig = plt.figure(figsize=(16, 8), tight_layout=True)

        # Fit bulk
        if verbose:
            print(f'Step 1/{total_steps}: Fitting bulk of the data')
        if bulk_fit == 'Empirical':
            self._bulk_empirical_fit()
        elif bulk_fit == 'BestFit':
            self._bulk_best_fit(verbose)
        else:
            raise AttributeError(
                'Unsupported bulk fitting method, check method_bulk')

        # Fit right tail
        if verbose:
            print(f'Step 2/{total_steps}: Fitting right tail')
        tail_right = _TailExtrapolation(self, fig_handle=self.diag_fig)
        tail_right.fit(maxima_extract=maxima_extract,
                       maxima_fit=maxima_fit, outlier_detect=outlier_detect)

        # Fit left tail
        if verbose:
            print(f'Step 3/{total_steps}: Fitting left tail')
        tail_left = _TailExtrapolation(
            self, left_tail=True, fig_handle=self.diag_fig)
        with np.errstate(over='ignore'):
            tail_left.fit(maxima_extract=maxima_extract,
                          maxima_fit=maxima_fit, outlier_detect=outlier_detect)

        # Arrange diagnostic plot
        self.diag_fig.axes[0].change_geometry(2, 4, 1)
        self.diag_fig.axes[2].change_geometry(2, 4, 2)
        self.diag_fig.axes[3].change_geometry(2, 4, 5)
        self.diag_fig.axes[5].change_geometry(2, 4, 6)
        self.diag_fig.axes[4].remove()
        self.diag_fig.axes[1].remove()

        # Combine tail and bulk
        self._combine_bulk_tail(tail_right, tail_left)

        if self.notebook_backend:
            plt.close(self.diag_fig)

    def predict(self, mrp=None, val=None):
        ''' Predict value given an MRP, or MRP given a value
            Parameters:
            -----------
                MRP: float, mean return period for prediction
                val: float, value of the variable
            Returns:
            --------
                Value corresponding to MRP (if MRP is input), or MRP 
                    corresponding to val
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
        F_emp = util.plotting_position(x, method='unbiased')
        if self.sample_coor[0] < x[0]:
            x = np.concatenate(([0], x))
            F_emp = np.concatenate(([0], F_emp))
        self.bulk_F = interp1d(
            x, F_emp, bounds_error=False)(self.sample_coor)

    def _bulk_best_fit(self, verbose):
        ''' Fit bulk using optimal distribution 
                See best_fit for more information
            Variables added:
            ----------------
                self.bulk_F: CDF corresponding to self.sample_coor
        '''
        ss = StandardScaler().fit(self.data.reshape(-1, 1))
        data_std = ss.transform(self.data.reshape(-1, 1)).flatten()
        dist, dist_name, _ = util.best_fit(data_std)
        self.bulk_F = dist.cdf(
            ss.transform(self.sample_coor.reshape(-1, 1))).flatten()
        if verbose:
            print(f'          Best fit distribution: {dist_name}')

    def _combine_bulk_tail(self, tail_right, tail_left):
        ''' Combine bulk, tail_right, and tail_left 
            Variables added:
            ----------------
                sample_F: CDF for self.sample_coor
                sample_mrp: MRP for self.sample_coor
        '''
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
        ax = self.diag_fig.add_subplot(1, 4, (3, 4))
        mrp_emp = 1 / self.c_rate / (
            1 - util.plotting_position(self.data, method='unbiased'))
        ax.plot(mrp_emp, np.sort(self.data), '.', color=[0.6, 0.6, 0.6],
                 markersize=8, label='Empirical')
        ax.set_xscale('log')
        idx_tail = (self.sample_coor >= tail_right.threshold) | (
            self.sample_coor <= -tail_left.threshold)
        sample_mrp_tail = np.copy(self.sample_mrp)
        sample_mrp_tail[~idx_tail] = np.nan
        ax.plot(sample_mrp_tail, self.sample_coor, 'b-', label='Tail fit')
        xlm = list(ax.get_xlim())
        xlm[1] = mrp_emp[-1] * 10 # Limit MRP to be 10 * data period
        ylm = list(ax.get_ylim())
        with np.errstate(invalid='ignore'):
            ylm[1] = self.sample_coor[sample_mrp_tail < xlm[1]][-1] # Corresponding y
        ax.plot(1 / self.c_rate / (1 - self.bulk_F), self.sample_coor, '--',
                 color=[0, 0.5, 0])
        ax.plot(
            1 / self.c_rate / (1 - self.bulk_F[~idx_tail]), 
            self.sample_coor[~idx_tail], '-',
            color=[0, 0.5, 0], label='Bulk fit')
        ax.plot(xlm, tail_right.threshold * np.array([1, 1]), 'k--')
        ax.plot(xlm, -tail_left.threshold * np.array([1, 1]), 'k--')
        ax.set_xlim(xlm)
        ax.set_ylim(ylm)
        ax.set_xlabel('Return period (year)')
        ax.set_ylabel('X')
        ax.set_title('Fitting result')
        ax.grid(True, which='both')
        ax.legend(loc='upper left')

class _TailExtrapolation:
    ''' Extrapolation the tail of a distribution.
        Parameters
        ----------
            univariate_obj: Instance of Univariate class
            left_tail: bool. Whether it's extrapolating the left tail
            fig_handle: Figure handler for diagnosis plot
    '''

    def __init__(self, univariate_obj, left_tail=False, fig_handle=None):
        for attr in ['data', 'time', 'total_year', 'sample_coor', 'bulk_F']:
            setattr(self, attr, getattr(univariate_obj, attr))
        if left_tail:
            # Reverse data so that it becomes a maxima extrapolation
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

    def fit(self, maxima_extract='Annual', maxima_fit='GumbelChart', 
            outlier_detect=None):
        '''Fit EVT tail, MRP ratio, then convert the EVT tail to raw data
        
        Parameters
        ----------
        maxima_extract : str, optional
            Method to extract independent maxima subset, by default 'Annual'
        maxima_fit : str, optional
            Method to fit maxima subset, by default 'GumbelChart'
        outlier_detect : str, optional
            Whether to assume outliers in the maxima subset, by default 'None'
            Options are 'None', 'RANSAC', and 'Huber'
        '''
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

    def _extract_annual_maxima(self):
        '''Extract annual maxima 

        Variables added
        ---------------
            self.maxima : numpy array in ascending order
        '''
        year = np.array(self.time.year)
        unique_year = np.unique(year)
        result = [max(self.data[year == cur_year]) for cur_year in unique_year]
        self.maxima = np.sort(result)

    def _fit_gumbel_chart(self, outlier_detect, plot_diagnosis):
        '''Fit a Gumbel distribution fit via Gumbel chart 

        Parameters
        ----------
            outlier_detect : bool
                Whether to assume outliers. Use OLS when False.
            plot_diagnosis: bool
                Whether to generate diagnostic plot.

        Variables added
        ---------------
            self.maxima_inlier_mask: Mask indicating inliers
            self.maxima_dist: Probability distribution for the maxima
            self.threshold: Threshold of X between bulk and tail, minimum 
                is constrained to be no lower than 5 percentile of F_maxima
        '''
        def _gumbel_y(F):
            ''' Calculate y coordinates on the Gumbel chart from CDF '''
            return -np.log(-np.log(F))

        x = self.maxima
        F = util.plotting_position(x, method='unbiased')
        y = _gumbel_y(F)
        if outlier_detect is None:
            mdl = linear_model.LinearRegression().fit(x.reshape(-1, 1), y)
            self.maxima_inlier_mask = np.array(
                [True] * len(self.maxima))  # Create mask manually
        elif outlier_detect == 'RANSAC':
            mdl = linear_model.RANSACRegressor(
                random_state=1).fit(x.reshape(-1, 1), y)
            self.maxima_inlier_mask = mdl.inlier_mask_
            mdl = mdl.estimator_
        elif outlier_detect == 'Huber':
            mdl = linear_model.HuberRegressor(
                epsilon=1.35).fit(x.reshape(-1, 1), y)
            self.maxima_inlier_mask = np.array(
                [True] * len(self.maxima))  # Create mask manually
        else:
            raise ValueError('Unrecognized outlier_detect keyword')
        k, b = mdl.coef_[0], mdl.intercept_

        if plot_diagnosis:
            ax = self.diag_fig.add_subplot(1, 3, 1, label=self.label)
            ax.plot(x[self.maxima_inlier_mask], y[self.maxima_inlier_mask],
                    'b.', markersize=10, label='Maxima(inliers)')
            ax.plot(x[~self.maxima_inlier_mask], y[~self.maxima_inlier_mask],
                    'r.', markersize=10, label='Maxima(outliers)')
            xlm, ylm = ax.get_xlim(), ax.get_ylim()
            ax.plot(
                self.sample_coor, mdl.predict(self.sample_coor.reshape(-1, 1)),
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
        '''Convert the EVT tail to the continuous dataset
        
        Parameters
        ----------
        plot_diagnosis : bool
            Whether to generate diagnostic plot
        
        Variables added
        ---------------
        self.m_rate : float
            Annual occurrence rate of the maxima data
        self.c_rate : float
            Annual occurrence rate of the continuous data
        self.tail_F : numpy array
            CDF of the tail of the continuous dataset. The part below
            self.threshold is set to be np.nan
        '''
        # Calculate empirical MRP for continuous and maxima datasets
        c_data = np.sort(self.data)
        m_data = self.maxima
        c_rate = len(c_data) / self.total_year
        m_rate = len(m_data) / self.total_year
        c_mrp_emp = 1 / c_rate / \
            (1 - util.plotting_position(c_data, method='unbiased'))
        m_mrp_emp = 1 / m_rate / \
            (1 - util.plotting_position(m_data, method='unbiased'))

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


if __name__ == '__main__':
    import os
    import pickle
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, '../datasets/D.pkl')
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    data = df.iloc[:, 0]
    urv = Univariate(data, sample_coor=np.linspace(0, 2*data.max(), 1000))
    urv.fit(bulk_fit='BestFit')
    urv.plot_diagnosis()
