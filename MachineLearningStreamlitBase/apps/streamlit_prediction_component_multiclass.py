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
import joblib
import xgboost as xgb

def app():
    st.title("Model Perturbation Analysis")
    with open('saved_models/trainXGB_class_map.pkl', 'rb') as f:
        class_names = list(pickle.load(f))
    
    M_dict = {}
    for classname in class_names:
        M_dict[classname] = joblib.load( 'saved_models/trainXGB_gpu_{}.model'.format(classname) )
    
    with open('saved_models/trainXGB_gpu_{}.data'.format(class_names[0]), 'rb') as f:
        train = pickle.load(f)
    with open('saved_models/trainXGB_categorical_map.pkl', 'rb') as f:
        col_dict_map = pickle.load(f)

    X = train[1]['X_valid'].copy() 
    ids = list(train[3]['ID_test'])
    X.index = ids
    labels_pred =  list(train[3]['y_pred_test']) 
    labels_actual = list(train[3]['y_test']) 
    select_patient = st.selectbox("Select the patient", list(X.index), index=0)
    select_patient_index = ids.index(select_patient) 
    categorical_columns = []
    numerical_columns = []
    X_new = X.fillna('X')
    for col in X_new.columns:
        # if len(X_new[col].value_counts()) <= 10:
        if col_dict_map.get(col, None) is not None:
            categorical_columns.append(col)
        else:
            numerical_columns.append(col) 
    
    st.write(categorical_columns, numerical_columns) 
    from collections import defaultdict
    
    
    new_feature_input = defaultdict(list) 
    for key, val in col_dict_map.items():
        rval = {j:i for i,j in val.items()}
        X_new[key] = X_new[key].map(lambda x: rval.get(x, x))
    
    st.subheader('Select feature values to see what-if analysis')
    st.write('--'*10)
    
    col1, col2, col3, col4 = st.beta_columns(4)
    for i in range(0, len(categorical_columns), 4):
        with col1:
            if (i+0) >= len(categorical_columns):
                continue
            c1 = categorical_columns[i+0] 
            idx = list(X_new[c1].unique()).index(X_new.loc[select_patient, c1]) 
            f1 = st.selectbox("Select feature for {}".format(c1), list(X_new[c1].unique()), index=idx)
            new_feature_input[c1].append(col_dict_map[c1].get(f1, np.nan))
        with col2:
            if (i+1) >= len(categorical_columns):
                continue
            c2 = categorical_columns[i+1] 
            idx = list(X_new[c2].unique()).index(X_new.loc[select_patient, c2]) 
            f2 = st.selectbox("Select feature for {}".format(c2), list(X_new[c2].unique()), index=idx)
            new_feature_input[c2].append(col_dict_map[c2].get(f2, np.nan))
        with col3:
            if (i+2) >= len(categorical_columns):
                continue
            c3 = categorical_columns[i+2] 
            idx = list(X_new[c3].unique()).index(X_new.loc[select_patient, c3]) 
            f3 = st.selectbox("Select feature for {}".format(c3), list(X_new[c3].unique()), index=idx)
            new_feature_input[c3].append(col_dict_map[c3].get(f3, np.nan))
        with col4:
            if (i+3) >= len(categorical_columns):
                continue
            c4 = categorical_columns[i+3] 
            idx = list(X_new[c4].unique()).index(X_new.loc[select_patient, c4]) 
            f4 = st.selectbox("Select feature for {}".format(c4), list(X_new[c4].unique()), index=idx)
            new_feature_input[c4].append(col_dict_map[c4].get(f4, np.nan))
    
    for col in numerical_columns:
        X_new[col] = X_new[col].map(lambda x: float(x) if not x=='X' else np.nan)
    for i in range(0, len(numerical_columns), 4):
        with col1:
            if (i+0) >= len(numerical_columns):
                continue
            c1 = numerical_columns[i+0] 
            idx = X_new.loc[select_patient, c1]
            f1 = st.number_input("Select feature for {}".format(c1), min_value=X_new[c1].min(),  max_value=X_new[c1].max(), value=idx)
            new_feature_input[c1].append(f1)
        with col2:
            if (i+1) >= len(numerical_columns):
                continue
            c2 = numerical_columns[i+1] 
            idx = X_new.loc[select_patient, c2]
            f2 = st.number_input("Select feature for {}".format(c2), min_value=X_new[c2].min(),  max_value=X_new[c2].max(), value=idx)
            new_feature_input[c2].append(f2)
        with col3:
            if (i+2) >= len(numerical_columns):
                continue
            c3 = numerical_columns[i+2] 
            idx = X_new.loc[select_patient, c3]
            f3 = st.number_input("Select feature for {}".format(c3), min_value=X_new[c3].min(),  max_value=X_new[c3].max(), value=idx)
            new_feature_input[c3].append(f3)
        with col4:
            if (i+3) >= len(numerical_columns):
                continue
            c4 = numerical_columns[i+3] 
            idx = X_new.loc[select_patient, c4]
            f4 = st.number_input("Select feature for {}".format(c4), min_value=X_new[c4].min(),  max_value=X_new[c4].max(), value=idx)
            new_feature_input[c4].append(f4)
    
    st.write('--'*10)
    
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "goldenred", "magenta", "yellow", "pink", "grey"]
    for e, classname in enumerate(class_names):
        color_discrete_map[classname] = color_discrete_map_list[e] 
    
    
    col01, col02 = st.beta_columns(2)
    with col01:
        st.subheader('Prediction on actual feature values')
        st.code(X_new.loc[select_patient, :].fillna('X')) 
        predicted_prob = defaultdict(list)
        for key, val in M_dict.items():
            predicted_prob['predicted_probability'].append(val.predict(xgb.DMatrix(X.loc[select_patient, :].values.reshape(1, -1), feature_names=X.columns))[0])
            predicted_prob['classname'].append(key)
        fig = px.pie(pd.DataFrame(predicted_prob), values='predicted_probability', names='classname', color='classname', color_discrete_map=color_discrete_map)
        fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=-0.05
            ))
        st.plotly_chart(fig)
    with col02:
        dfl = pd.DataFrame(new_feature_input)
        ndfl = dfl.copy()
        for key, val in col_dict_map.items():
            rval = {j:i for i,j in val.items()}
            ndfl[key] = ndfl[key].map(lambda x: rval.get(x, x))
        st.subheader('Prediction on selected feature values')
        st.code(ndfl.iloc[0].fillna('X'))
        dfl = dfl[X.columns].replace('X', np.nan)
        predicted_prob = defaultdict(list)
        for key, val in M_dict.items():
            predicted_prob['predicted_probability'].append(val.predict(xgb.DMatrix(dfl.iloc[0, :].values.reshape(1, -1), feature_names=dfl.columns))[0])
            predicted_prob['classname'].append(key)
        fig = px.pie(pd.DataFrame(predicted_prob), values='predicted_probability', names='classname', color='classname', color_discrete_map=color_discrete_map)
        fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.05
            ))
        st.plotly_chart(fig)