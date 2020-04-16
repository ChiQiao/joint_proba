import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
import streamlit as st

from tail_extrap import multivariate

@st.cache
def initialize(file_path):
    df = pd.read_csv(file_path, sep=';', index_col=0, parse_dates=True)
    mv = multivariate.Multivariate(df)
    return df, mv

def condY_to_str(condY_x: list) -> str:
    '''Convert a list into the format of "start : interval : end" for display
    '''
    return (f'{condY_x[0]:.1f} : '
        f'{condY_x[1] - condY_x[0]:.1f} : '
        f'{condY_x[-1]:.1f}')

def str_to_condY(s: str) -> list:
    '''Convert condY_x expression from text to list
        s has the format of "start : interval : end"
    '''
    condY_x = list(map(float, s.split(':')))
    condY_x = np.arange(condY_x[0], condY_x[2] + 1e-5, condY_x[1])
        # add a small value of 1e-5 so that end is included
    return condY_x

# This module is not meant to be standalone, as session_state is used from the
#   main script. However, for the sake of testing and linting, session_state is
#   initilized again.
if 'ss' not in globals():
    import session_state
    ss = session_state.get(mv=None)

place_holder_1 = st.empty()
place_holder_1.header('Upload data')
place_holder_2 = st.empty()
file_path = place_holder_2.file_uploader(
    'Format: txt or csv, with first column as time index')

# For develop
file_path = r'C:\Users\joey3\OneDrive\CS_DS\tail_extrap\datasets\D.pkl'
with open(file_path, 'rb') as f:
    df = pickle.load(f)
mv_init = multivariate.Multivariate(df)

if file_path is not None:

    # Load data

    # df, mv_init = initialize(file_path)
    if ss.mv is None:
        ss.mv = mv_init
    # session_state is initialized at the first time data is uploaded, so the
    # file uploader is disabled afterward
    place_holder_1.header('Data summary')
    place_holder_2.markdown('''
        Below shows the first and last two rows. \n
        To use another file, please refresh page.
        ''')
    st.dataframe(data=df.iloc[[1, 2, -2, -1], :])

    # Display options on sidebar

    col_names = list(df.columns)
    st.sidebar.header('')
    st.sidebar.subheader('Options')

    x_name = st.sidebar.selectbox(
        'Independent variable',
        options=col_names,
        index=col_names.index(ss.mv.x_name))
    col_x = col_names.index(x_name)
    
    y_name = st.sidebar.selectbox(
        'Dependent variable',
        options=col_names,
        index=col_names.index(ss.mv.y_name))
    col_y = col_names.index(y_name)
    
    condY_x_text = st.sidebar.text_input(
        'Discrete values of x to evaluate f(Y|X)',
        condY_to_str(ss.mv.condY_x))
    condY_x = str_to_condY(condY_x_text)
    st.sidebar.markdown(
        '''&emsp;<span style="font-size:0.8em; color:grey;">
        Format: "start : interval : end"</span>''',
        unsafe_allow_html=True)

    # Plot data

    plt.plot(df.iloc[:, col_x], df.iloc[:, col_y], 'b.', label='Raw data')
    ylm = plt.ylim()
    plt.plot(
        np.vstack([condY_x, condY_x]), 
        np.matlib.repmat(np.array(ylm).reshape(2, 1), 1, len(condY_x)),
        '--', color=[1, 0.5, 0.5], label='CondY_X')
    plt.ylim(ylm)
    plt.xlabel(col_names[col_x])
    plt.ylabel(col_names[col_y])
    plt.grid(True)
    plt.legend(labels=['Raw data', 'X to evaluate f(Y|X)'], loc='upper left')
    st.pyplot()

    # Save options

    save = st.sidebar.button('Save')
    if save:
        ss.mv = multivariate.Multivariate(
            df, col_x=col_x, col_y=col_y, condY_x=condY_x)
        place_holder_3 = st.sidebar.empty()
        place_holder_3.success('Options saved')
        time.sleep(1)
        place_holder_3.empty()
          


