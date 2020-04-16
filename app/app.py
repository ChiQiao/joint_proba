import importlib.util
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
import streamlit as st

from tail_extrap import multivariate
import session_state

ss = session_state.get(
    mv=None,
    # col_x=0,
    # col_y=1,
    # condY_x=[],
)

module = {
    'Import data': 'module_load.py',
    'Fit marginal X': 'module_marginal_X.py',
    'Fit marginal Y': 'module_marginal_Y.py',
    'Fit conditional Y': 'module_conditional_Y.py',
    'Contour': 'init_multivariate()',
}
st.sidebar.subheader('Workflow')
selection = st.sidebar.selectbox(
    '', options=['Select one'] + list(module.keys())
)
# selection = st.sidebar.selectbox(
#     '', options=list(module.keys()), key='workflow'
# )


if selection == 'Select one':
    # TODO: Intro page
    st.title('Tail Extrapolation')
else:
    # Create module from filepath and put in sys.modules, so Streamlit knows
    # to watch it for changes. (For develop use)
    fake_module_count = 0
    def load_module(filepath):
        global fake_module_count
        modulename = "_dont_care_%s" % fake_module_count
        spec = importlib.util.spec_from_file_location(modulename, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[modulename] = module
        fake_module_count += 1

    # Run the selected file.
    with open(module[selection]) as f:
        load_module(module[selection])
        filebody = f.read()
    exec(filebody)
