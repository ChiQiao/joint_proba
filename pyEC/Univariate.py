import calendar
import copy
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


class Univariate:
    ''' Univariate random variable object with accurate tail extrapolation 
            The right tail is extrapolated, assuming a right-skewed distribution
        Parameters
        ----------
            data: pandas series with datetime index. If constructing conditional 
                distribution, NaN should be retained for total_year to be 
                correctly inferred
            sample_coor: numpy array. Sample coordinate for outputs reference. 
                Inferred from data if not provided
    '''

    def __init__(self, data, sample_coor=None):
        def days_in_year(year):
            ''' Total days in a year '''
            return 366 if calendar.isleap(year) else 365

        if not isinstance(data, pd.Series):
            raise TypeError('data should be a pandas series')

        first_year_ratio = 1 - \
            (data.index[0].dayofyear - 1) / \
            days_in_year(data.index[0].year)
        last_year_ratio = data.index[-1].dayofyear / \
            days_in_year(data.index[-1].year)
        total_year = data.index[-1].year - data.index[0].year - 1 +\
            first_year_ratio + last_year_ratio

        self.total_year = total_year
        idx_valid = ~np.isnan(data.values)
        self.data = data.values[idx_valid]
        self.year = np.array(data.index.year)[idx_valid]

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
