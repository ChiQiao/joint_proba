import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
import streamlit as st

from tail_extrap import multivariate

print('************ module_load called ******************')

@st.cache
def initialize(file_path, col_x):
    print('initialize called')
    df = pd.read_csv(file_path, sep=';', index_col=0, parse_dates=True)
    condY_x_init = list(np.linspace(
        df.iloc[:, col_x].min(), df.iloc[:, col_x].max(), 10))
    return df, condY_x_init

@st.cache
def init_condY(df, col_x):
    # Initialize condY in session_state

    print('init_condY called')
    condY_x = list(np.linspace(
        df.iloc[:, col_x].min(), df.iloc[:, col_x].max(), 10))
    return condY_x

def condY_to_str(condY_x: list) -> str:
    if condY_x:
        return (f'{condY_x[0]:.1f} : '
            f'{condY_x[1] - condY_x[0]:.1f} : '
            f'{condY_x[-1]:.1f}')
    else:
        return 'empty'

def str_to_condY(s: str) -> list:
    '''Convert condY_x expression from text to list
        s has the format of "start : interval : end"
    '''
    condY_x = list(map(float, s.split(':')))
    condY_x = np.arange(condY_x[0], condY_x[2] + 1e-5, condY_x[1])
        # add a small value of 1e-5 so that end is included
    return condY_x

place_holder_1 = st.empty()
place_holder_1.header('Upload data')
place_holder_2 = st.empty()
file_path = place_holder_2.file_uploader(
    'Format: txt or csv, with first column as time index')

# file_path = r'C:\Users\joey3\OneDrive\CS_DS\tail_extrap\datasets\D.pkl'
# with open(file_path, 'rb') as f:
#     df = pickle.load(f)

if file_path is not None:

    # Load data

    df, condY_x_init = initialize(file_path, 0)
    if not ss.condY_x:
        ss.condY_x = condY_x_init
        print('ss.condY_x updated')
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
        index=ss.col_x)
    col_x = col_names.index(x_name)

    y_name = st.sidebar.selectbox(
        'Dependent variable',
        options=col_names,
        index=ss.col_y)
    col_y = col_names.index(y_name)
    
    condY_x = init_condY(df, col_x)
    condY_x_text = st.sidebar.text_input(
        'Discrete values of x to evaluate f(Y|X)',
        condY_to_str(ss.condY_x))
    condY_x = str_to_condY(condY_x_text)
    st.sidebar.markdown(
        '''&emsp;<span style="font-size:0.8em; color:grey;">
        Format: "start : interval : end"</span>''',
        unsafe_allow_html=True)
    print([ss.col_x, ss.col_y, condY_to_str(ss.condY_x), type(ss.mv)])

    

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

    save = st.sidebar.button('Save')
    if save:
        ss.col_x = col_x
        ss.col_y = col_y
        ss.condY_x = condY_x
        ss.mv = multivariate.Multivariate(
            df, col_x=col_x, col_y=col_y, condY_x=condY_x)
        print('Saved!')
        print([ss.col_x, ss.col_y, condY_to_str(ss.condY_x), type(ss.mv)])