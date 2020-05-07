import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from sklearn.metrics import r2_score

def best_fit(data, dist_names=None, criteria='chi-square', qq_plot=False, dist_config=None):
    ''' Search for best distribution fitting
        Parameters:
        -----------
            data: array-like raw data for distribution fitting
            dist_names: list of distribution candidate names in scipy format. 
                If dist_names=None, a default list of ~20 candidates is used.
            criteria: 'chi-square' or 'r2', for best distribution selection
            qq_plot: bool. whether to display qq plot
            dist_config: dict of configuration, see config/dist_repara.json
                as an example. Key 'fit_kwargs' will be used to constrain 
                distirbution fitting.
        Returns:
        --------
            dist: scipy.stats distribution with the best fit
            fit_df: DataFrame including columns Distribution', 'chi-square', 
                'r2', and 'param'. Sorted by 'r2' in descending order. 
        Note:
        -----
            r2 (R squared) of the qq plot is implemented in a customized way to
                improve efficiency. Quantiles are evaluated at ~200 points 
                evenly distributed from min to max, rather than at each data
                point. As such, bulk and tail have the same weight in evaluating
                goodness of fit.
        Reference:
        ----------
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
        ]  # common distributions

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
        )  # loc and scale not included in ddof

        # Customized way to quickly calculate R2 from Q-Q plot
        val_pred = dist.ppf(
            quantile, *param[:-2], loc=param[-2], scale=param[-1])
        r2.append(r2_score(val_true, val_pred))

    # Collate results and sort by goodness of fit (best at top)
    fit_df = pd.DataFrame({
        'Distribution': dist_names,
        'chi-square': chi_square,
        'r2': r2,
        'param': params,
    })
    asc_order = {'chi-square': True, 'r2': False}
    fit_df = fit_df.sort_values(
        criteria, ascending=asc_order[criteria]).reset_index(drop=True)

    # Create distribution object for the best fitting
    param = fit_df['param'][0]
    dist_name = fit_df['Distribution'][0]
    dist = stats._distn_infrastructure.rv_frozen(
        getattr(stats, dist_name),
        *param[:-2], loc=param[-2], scale=param[-1]
    )

    if qq_plot:
        plt.figure(figsize=(16, 8))
        for idx in range(min(8, len(fit_df))):
            plt.subplot(2, 4, idx+1)
            stats.probplot(
                data, sparams=fit_df.param[idx],
                dist=fit_df.Distribution[idx], plot=plt)
            plt.title(fit_df.Distribution[idx])
            plt.grid(True)
            plt.tight_layout()
        plt.show()

    return dist, dist_name, fit_df

def plotting_position(data, method='unbiased'):
        ''' Plotting position (empirical CDF) of data 
            Parameters:
            -----------
                data: array-like
                method:
                    'unbiased': Unbiased estimation used for Gumbel chart
                    'simple': order / (N + 1), open range of (0, 1)
        '''
        assert not np.any(np.isnan(data)), 'data should not include any NaN'
        n = len(data)
        if method == 'unbiased':
            return ((np.arange(n) + 1) - 0.44) / (n + 0.12)
        elif method == 'simple':
            return (np.arange(n) + 1) / (n + 1)
        else:
            raise AttributeError(
                'Unsupported calculation method for plotting position')


if __name__ == '__main__':
    from scipy.stats import weibull_min
    r = weibull_min.rvs(2, size=1000)
    dist, dist_name, fit_df = best_fit(r, criteria='r2', qq_plot=True)