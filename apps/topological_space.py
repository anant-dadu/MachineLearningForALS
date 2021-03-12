import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    umap_org = pd.read_csv('saved_models/df_umap_org.csv', sep=',')
    umap_rep = pd.read_csv('saved_models/df_umap_rep.csv', sep=',')
    colorable_columns = ['clinicaltype_at_oneyear']
    select_color = st.selectbox('Select a color to visualize', colorable_columns, index=0)
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "magenta", "yellow", "pink", "grey"]
    for e, classname in enumerate(list(set(umap_rep[select_color]))):
        color_discrete_map[classname] = color_discrete_map_list[e] 
    st.subheader("3D-UMAP Plots for ALS subtypes")
    st.write('#### Training Cohort')
    fig = px.scatter_3d(umap_org, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color, color_discrete_map=color_discrete_map)
    st.plotly_chart(fig, use_container_width=True)
    st.write('#### Replication Cohort')
    fig = px.scatter_3d(umap_rep, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color, color_discrete_map=color_discrete_map)
    st.plotly_chart(fig, use_container_width=True)
    