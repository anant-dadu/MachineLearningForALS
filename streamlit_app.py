import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
import plotly.express as px
import plotly
import copy
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
import warnings
warnings.filterwarnings("ignore")
import logging


logger = logging.getLogger()
logger.disabled = True

from MachineLearningStreamlitBase.multiapp import MultiApp
from MachineLearningStreamlitBase.apps import streamlit_prediction_component_multiclass, streamlit_shapley_component

# add any app you like in apps directory
from apps import topological_space, select

app = MultiApp()
max_width = 4000
padding_top = 10
padding_right = 10
padding_left = 10
padding_bottom = 10
COLOR = 'black'
BACKGROUND_COLOR = 'white'
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

import copy

import psutil
# gives a single float value
mem = psutil.virtual_memory()
cols = st.beta_columns(4)
cols[0].write("Available Memory:", mem.available/1e9, "GB")
cols[1].write( 'Available memory fraction', psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
cols[2].write("Total CPUS:", psutil.cpu_count())
cols[3].write("Cores utilization", [(i*100)/psutil.cpu_count() for i in psutil.getloadavg()])
cols[0].write ('Fraction of RAM usage:', psutil.virtual_memory().percent)
cols[1].write("Load Average", psutil.getloadavg())
cols[2].write("CPU percent usage", psutil.cpu_percent(interval=None))
# gives an object with many fields
# you can convert that object to a dictionary
cols[3].write("All Memory USE", dict(psutil.virtual_memory()._asdict()))
# st.write(dict(psutil.virtual_memory()._asdict()))

# you can have the percentage of used RAM
# you can calculate percentage of available memory

##TODO: UPDATE TITLE
st.write('# Machine Learning for ALS') 
app.add_app("Select", select.app)
app.add_app("Scientific background", streamlit_shapley_component.app)
app.add_app("Predict Patient ALS Subtype", streamlit_prediction_component_multiclass.app)
##TODO: Add any apps you like
app.add_app("Explore the ALS subtype topological space", topological_space.app)
app.run()

