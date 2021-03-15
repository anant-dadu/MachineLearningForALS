import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    umap_org_full = pd.read_csv('saved_models/ALS_NN_umap_org.csv', sep=',')
    umap_rep_full = pd.read_csv('saved_models/ALS_NN_umap_rep.csv', sep=',')
    colorable_columns = ['clinicaltype_at_oneyear', 'clinicaltype_at_onset', 'initial_dx_was_PLS',
                        'mutationPresent', 'cognitiveStatus1', 'cognitiveImpairmentPresent', 'PEGinserted', 'BIPAP']
    colorable_columns = list(set(colorable_columns).intersection(set(list(umap_rep_full.columns))))
    select_color = st.selectbox('Select a color to visualize', colorable_columns, index=0)
    umap_org = umap_org_full[[select_color] + ['UMAP_3_1', 'UMAP_3_2', 'UMAP_3_3']].dropna()
    umap_rep = umap_rep_full[[select_color] + ['UMAP_3_1', 'UMAP_3_2', 'UMAP_3_3']].dropna()
    st.write("Total Points in Original Data:", len(umap_org))
    st.write("Total Points in Replication Data:", len(umap_rep))
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "magenta", "yellow", "pink", "grey", "black", "brown", "purple"]
    for e, classname in enumerate(sorted( list(set(umap_org[select_color]).union(set(umap_rep[select_color]))) ) ) :
        color_discrete_map[classname] = color_discrete_map_list[e] 
    
    st.write("## 3D-UMAP Plots for ALS subtypes")
    st.write('### Training Cohort')
    fig = px.scatter_3d(umap_org, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color, color_discrete_map=color_discrete_map)
    st.plotly_chart(fig, use_container_width=True)
    st.write('### Replication Cohort')
    fig = px.scatter_3d(umap_rep, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color, color_discrete_map=color_discrete_map)
    st.plotly_chart(fig, use_container_width=True)
    