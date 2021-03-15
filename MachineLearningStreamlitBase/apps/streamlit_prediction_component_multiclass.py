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
    
    st.markdown("<font color='RED'>Discliamer: This predictive tool is only for research purposes</font>", unsafe_allow_html=True)
    st.write("## Model Perturbation Analysis")
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
    # select_patient = st.selectbox("Select the patient", list(X.index), index=0)
    
    categorical_columns = []
    numerical_columns = []
    X_new = X.fillna('X')
    for col in X_new.columns:
        # if len(X_new[col].value_counts()) <= 10:
        if col_dict_map.get(col, None) is not None:
            categorical_columns.append(col)
        else:
            numerical_columns.append(col) 
    
    st.write('### Please enter the following {} factors to perform prediction'.format(len(categorical_columns + numerical_columns)))
    # st.write("***Categorical Columns:***", categorical_columns) 
    # st.write("***Numerical Columns:***", numerical_columns) 
    from collections import defaultdict
    if st.button("Random Patient"):
        import random
        select_patient = random.choice(list(X.index))
    else:
        select_patient = list(X.index)[0]

    select_patient_index = ids.index(select_patient) 
    new_feature_input = defaultdict(list) 
    for key, val in col_dict_map.items():
        rval = {j:i for i,j in val.items()}
        X_new[key] = X_new[key].map(lambda x: rval.get(x, x))
    
    st.write('--'*10)
    st.write('##### Note: X denoted NA values')
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
        st.write('### Prediction on actual feature values')
        st.code(X_new.loc[select_patient, :].fillna('X')) 
        predicted_prob = defaultdict(list)
        predicted_class = -1
        max_val = -1
        for key, val in M_dict.items():
            predicted_prob['predicted_probability'].append(val.predict(xgb.DMatrix(X.loc[select_patient, :].values.reshape(1, -1), feature_names=X.columns))[0])
            predicted_prob['classname'].append(key)
            if predicted_prob['predicted_probability'][-1] > max_val:
                predicted_class = key
                max_val = predicted_prob['predicted_probability'][-1] 

        fig = px.bar(pd.DataFrame(predicted_prob), y='predicted_probability', x=sorted(list(predicted_prob['classname'])))
        st.plotly_chart(fig)
        st.write('#### Trajectory for Predicted Class')
        with open('saved_models/trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
            new_train = pickle.load(f)
        exval = new_train[2]['explainer_train'] 
        explainer_train = shap.TreeExplainer(M_dict[predicted_class])
        t1 = pd.DataFrame(X.loc[select_patient, :]).T
        t2 = pd.DataFrame(X_new.loc[select_patient, :].fillna('X')).T
        shap_values_train = explainer_train.shap_values(t1)
        shap.force_plot(exval, shap_values_train, t1, show=False, matplotlib=True)
        st.pyplot()
        fig, ax = plt.subplots()
        _ = shap.decision_plot(exval, shap_values_train, t2, link='logit', return_objects=True, new_base_value=0, highlight=0)
        st.pyplot(fig)
    with col02:
        dfl = pd.DataFrame(new_feature_input)
        ndfl = dfl.copy()
        for key, val in col_dict_map.items():
            rval = {j:i for i,j in val.items()}
            ndfl[key] = ndfl[key].map(lambda x: rval.get(x, x))
        st.write('### Prediction on selected feature values')
        st.code(ndfl.iloc[0].fillna('X'))
        dfl = dfl[X.columns].replace('X', np.nan)
        predicted_prob = defaultdict(list)
        predicted_class = -1
        max_val = -1
        for key, val in M_dict.items():
            predicted_prob['predicted_probability'].append(val.predict(xgb.DMatrix(dfl.iloc[0, :].values.reshape(1, -1), feature_names=dfl.columns))[0])
            predicted_prob['classname'].append(key)
            if predicted_prob['predicted_probability'][-1] > max_val:
                predicted_class = key
                max_val = predicted_prob['predicted_probability'][-1] 

        fig = px.bar(pd.DataFrame(predicted_prob), y='predicted_probability', x=sorted(list(predicted_prob['classname'])))
        st.plotly_chart(fig)
        st.write('#### Trajectory for Predicted Class')
        with open('saved_models/trainXGB_gpu_{}.data'.format(predicted_class), 'rb') as f:
            new_train = pickle.load(f)
        exval = new_train[2]['explainer_train'] 
        explainer_train = shap.TreeExplainer(M_dict[predicted_class])
        t1 = dfl.copy() 
        shap_values_train = explainer_train.shap_values(t1)
        shap.force_plot(exval, shap_values_train, t1, show=False, matplotlib=True)
        st.pyplot()
        fig, ax = plt.subplots()
        _ = shap.decision_plot(exval, shap_values_train, ndfl.fillna('X'), link='logit', return_objects=True, new_base_value=0, highlight=0)
        st.pyplot(fig)
    
    # st.write('### Force Plots')
    # patient_name = st.selectbox('Select patient id', options=list(patient_index))
    # sample_id = patient_index.index(patient_name)
    # col8, col9 = st.beta_columns(2)
    # with col8:
    #     st.info('Actual Label: ***{}***'.format('PD' if labels_actual[sample_id]==1 else 'HC'))
    #     st.info('Predicted PD class Probability: ***{}***'.format(round(float(labels_pred[sample_id]), 2)))
    # with col9:
    #     shap.force_plot(exval, shap_values[sample_id,:], X.iloc[sample_id,:], show=False, matplotlib=True)
    #     st.pyplot()
    
    # col10, col11 = st.beta_columns(2)
    # with col10:
    #     fig, ax = plt.subplots()
    #     shap.decision_plot(exval, shap_values[sample_id], X.iloc[sample_id], link='logit', highlight=0, new_base_value=0)
    #     st.pyplot()



# fig = px.pie(pd.DataFrame(predicted_prob), values='predicted_probability', names='classname', color='classname', color_discrete_map=color_discrete_map)
        # fig.update_layout(legend=dict(
        #         yanchor="top",
        #         y=0.99,
        #         xanchor="right",
        #         x=1.05
        #     ))
