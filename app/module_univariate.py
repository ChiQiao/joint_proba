

def init_univariate(item):
    # Options on the sidebar
    st.sidebar.header('')
    st.sidebar.subheader('Options')
    maxima_extract_opt = st.sidebar.selectbox(
        'Maxima extraction', ['Annual maxima'])
    maxima_fit_opt = st.sidebar.selectbox(
        'Maxima fitting', ['Gumbel chart'])
    bulk_fit_opt = st.sidebar.selectbox(
        'Bulk fitting', ['Empirical', 'Parametric'])
    outlier_opt = st.sidebar.selectbox(
        'Outlier detection', ['None', 'RANSAC regression', 'Huber regression'])

    # Option parser
    maxima_extract_dict = {
        'Annual maxima': 'Annual',
    }
    maxima_fit_dict = {
        'Gumbel chart': 'GumbelChart',
    }
    bulk_fit_dict = {
        'Empirical': 'Empirical',
        'Parametric': 'BestFit',
    }
    outlier_dict = {
        'None': None,
        'RANSAC regression': 'RANSAC', 
        'Huber regression': 'Huber',
    }

    # Load multivariate instance
    with open('temp.pkl', 'rb') as f:
        mv = pickle.load(f)
    
    # Fit univariate distribution
    if item == 'x':
        mv._fit_marginalX(
            maxima_extract=maxima_extract_dict[maxima_extract_opt], 
            maxima_fit=maxima_fit_dict[maxima_fit_opt], 
            bulk_fit=bulk_fit_dict[bulk_fit_opt], 
            outlier_detect=outlier_dict[outlier_opt],
        )
        st.pyplot(fig=mv.x_dist.diag_fig)
        
    elif item == 'y':
        mv._fit_marginalY(
            maxima_extract=maxima_extract_dict[maxima_extract_opt], 
            maxima_fit=maxima_fit_dict[maxima_fit_opt], 
            bulk_fit=bulk_fit_dict[bulk_fit_opt], 
            outlier_detect=outlier_dict[outlier_opt],
        )

    elif item == 'condY':
        mv._fit_condY_disc(
            maxima_extract=maxima_extract_dict[maxima_extract_opt], 
            maxima_fit=maxima_fit_dict[maxima_fit_opt], 
            bulk_fit=bulk_fit_dict[bulk_fit_opt], 
            outlier_detect=outlier_dict[outlier_opt],
        )