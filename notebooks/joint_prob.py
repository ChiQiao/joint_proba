import calendar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
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

    def fit(self, method_maxima='Annual', method_tail='GumbelChart', method_bulk='Empirical',
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
        fig = plt.figure(figsize=(10, 8))
            
        # Fitting bulk
        if verbose:
            print(f'Step 1/{total_steps}: Fitting bulk of the data')
        if method_bulk == 'Empirical':
            self._bulk_empirical_fit()
        elif method_bulk == 'BestFit':
            self._bulk_best_fit(verbose)
        else:
            raise AttributeError('Unsupported bulk fitting method, check method_bulk')
        
        # Extract maxima (sorted, no NaN)
        if verbose:
            print(f'Step 2/{total_steps}: Extracting and fitting maxima')
        if method_maxima == 'Annual':
            self._extract_annual_maxima()
        else:
            raise AttributeError('Unsupported maxima extraction method, check method_maxima')
        
        if method_tail == 'GumbelChart':
            self._fit_gumbel_chart(outlier_detect, plot_diagnosis=True)
        else:
            raise AttributeError('Unsupported tail fitting method, check method_tail.')
        
        # Fitting tail
        if verbose:
            print(f'Step 3/{total_steps}: Fitting tail of the data')
        self._maxima_to_continuous(plot_diagnosis=True)
        
        # Combine tail and bulk
        sample_F = np.copy(self.bulk_F)
        idx_tail = self.sample_coor >= self.threshold
        sample_F[idx_tail] = self.tail_F[idx_tail]
        self.sample_F = sample_F
        with np.errstate(divide='ignore'):
            self.sample_mrp = 1 / self.c_rate / (1 - sample_F)
        
        # Diagnositc plot
        plt.subplot(2,2,4)
        plt.plot(1 / self.c_rate / (1 - Univariate._plotting_position(self.data, method='unbiased')), 
                    np.sort(self.data), '.', color=[0.6, 0.6, 0.6], 
                    markersize=8, label='Empirical')
        plt.xscale('log')
        plt.plot(self.sample_mrp[idx_tail], self.sample_coor[idx_tail], 'b-', label='Tail fit')
        xlm = plt.xlim()
        ylm = plt.ylim()
        plt.plot(1 / self.c_rate / (1 - self.bulk_F[~idx_tail]), self.sample_coor[~idx_tail], '-',
                color=[0, 0.5, 0], label='Bulk fit')
        plt.plot(1 / self.c_rate / (1 - self.bulk_F[idx_tail]), self.sample_coor[idx_tail], '--',
                color=[0, 0.5, 0])
        plt.plot(xlm, self.threshold * np.array([1, 1]), 'k--')
        plt.xlim(xlm)
        plt.ylim(ylm)
        plt.xlabel('Return period (year)')
        plt.ylabel('X')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.tight_layout()
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
            plt.subplot(2,2,2)
            sample_t = np.linspace(0, 3.5, 100)
            plt.plot(t_emp[:-1], mrp_ratio_emp[:-1], 'k.', markersize=8, label='Empirical')
            plt.plot(t_threshold, mrp_ratio_threshold, 'rx', markersize=10, label='Connecting point')
            plt.plot(sample_t, func(sample_t, *popt), 'r-', label='Fit')
            plt.xlim([0, 3.5])
            plt.xlabel('t(X)')
            plt.ylabel('MRP ratio')
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.tight_layout()
            
            # Maxima to continuous conversion
            plt.subplot(2,2,3)
            plt.plot(m_mrp_emp, m_data, '.', color=[1, 0.4, 0.4], markersize=8, label='Maxima')
            plt.plot(c_mrp_emp[c_data >= m_data.min()], 
                     c_data[c_data >= m_data.min()], '.', color=[0.4, 0.4, 1], 
                     markersize=8, label='Continuous')
            plt.xscale('log')
            xlm = plt.xlim()
            ylm = plt.ylim()
            plt.plot(m_sample_mrp, self.sample_coor, 'r-', label='Maxima fit')
            plt.plot(c_sample_mrp, self.sample_coor, 'b-', label='Continuous fit')
            plt.plot(xlm, self.threshold * np.array([1, 1]), 'k--')
            plt.xlim(xlm)
            plt.ylim([m_data.min(), ylm[1]])
            plt.xlabel('Return period (year)')
            plt.ylabel('X')
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.tight_layout()
    
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
        ''' Fit bulk using optimal distribution (see _fit_best for more information)
            Variables added:
            ----------------
                self.bulk_F: CDF corresponding to self.sample_coor
        '''
        fit_df, ss = Univariate._fit_best(self.data)
        self.bulk_F = getattr(stats, fit_df.Distribution[0]).cdf(
            ss.transform(self.sample_coor.reshape(-1, 1)), 
            *fit_df.param[0][:-2], 
            loc=fit_df.param[0][-2], 
            scale=fit_df.param[0][-1]).flatten()
        if verbose:
            print(f'          Best fit distribution: {fit_df.Distribution[0]}')

    @staticmethod
    def _fit_best(data, dist_names=None, qq_plot=False):
        ''' Search for best distribution fitting based on chi-squar test
            List for scipy distribution: 
                https://docs.scipy.org/doc/scipy/reference/stats.html 
            Regarding chi-squre test:
                https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
        '''
        warnings.filterwarnings("ignore")
        if dist_names is None:
            dist_names = [
                'beta', 'burr', 'expon', 'exponweib', 
                'genextreme', 'gamma', 'gumbel_r', 'gumbel_l', 
                'logistic', 'lognorm', 'nakagami', 'norm',
                'rayleigh', 't', 'weibull_min', 'weibull_max'
            ]

        # Standardize data
        data = data.reshape(-1, 1)
        ss = StandardScaler().fit(data)
        data_std = ss.transform(data)

        # Prepare observation data for the chi-square test
        number_of_bins = 20
        observed_values, bin_edges = np.histogram(data_std, bins=number_of_bins)

        chi_square, p_values, params = [], [], []
        for dist_name in dist_names:
            # Fit distribution
            dist = getattr(stats, dist_name)
            param = dist.fit(data_std)
            params.append(param)

            # Obtain the KS test P statistic
            p_values.append(stats.kstest(data_std.T, dist_name, args=param)[1])    

            # calculate chi-squared
            cdf = dist.cdf(bin_edges, *param[:-2], loc=param[-2], scale=param[-1])
            expected_values = len(data_std) * np.diff(cdf)
            chi_square.append(
                stats.chisquare(observed_values, expected_values, ddof=len(param)-2)[0]
            ) # loc and scale not included

        # Collate results and sort by goodness of fit (best at top)
        fit_df = pd.DataFrame({
            'Distribution': dist_names,
            'chi_square': chi_square,
            'p_value': np.round(p_values, 5),
            'param': params,
        })
        fit_df = fit_df.sort_values(['chi_square']).reset_index(drop=True)

        if qq_plot:
            plt.figure(figsize=(16, 8))
            for idx in range(min(7, len(fit_df))):
                plt.subplot(2,4,idx+1)
                stats.probplot(
                    data_std.flatten(), sparams=fit_df.param[idx], 
                    dist=fit_df.Distribution[idx], plot=plt)
                plt.title(fit_df.Distribution[idx])
                plt.tight_layout()
            plt.show()
        return fit_df, ss

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
        x = self.maxima
        F = Univariate._plotting_position(x, method='unbiased')
        y = Univariate._gumbel_y(F)
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
            plt.subplot(2,2,1)
            plt.plot(x[self.maxima_inlier_mask], y[self.maxima_inlier_mask],
                     'b.', markersize=10, label='Maxima(inliers)')
            plt.plot(x[~self.maxima_inlier_mask], y[~self.maxima_inlier_mask],
                     'r.', markersize=10, label='Maxima(outliers)')
            xlm, ylm = plt.xlim(), plt.ylim()
            plt.plot(self.sample_coor, mdl.predict(self.sample_coor.reshape(-1, 1)),
                     'r--', label='Linear fitting')
            plt.xlim(xlm)
            plt.ylim(ylm)
            plt.xlabel('Maxima data')
            plt.ylabel('$-ln(-ln(F))$')
            plt.title('Gumbel chart')
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            
        self.maxima_pd = stats.gumbel_r(loc=-b/k, scale=1/k)
        self.maxima_inlier_mask[self.maxima < self.maxima_pd.ppf(0.05)] = False
        self.threshold = self.maxima[self.maxima_inlier_mask].min()

    @staticmethod
    def _gumbel_y(F):
        ''' Calculate y coordinates on the Gumbel chart from CDF '''
        return -np.log(-np.log(F))

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


class TailExtrapolation:
    def __init__(self):
        pass


if __name__ == '__main__':
    import pickle
    import time
    with open('../datasets/D.pkl', 'rb') as f:
        df = pickle.load(f)
    # df = pd.read_csv('../datasets/D.txt', sep=';', index_col=0, parse_dates=True)
    test = Univariate(df.iloc[:, 0])
    test.fit()
    print('Now plot')
    time.sleep(3)
    test.plot_diagnosis()