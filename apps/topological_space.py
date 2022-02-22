import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.write("## Topological Space for ALS Subtypes using Semi-supervised Approach")
    original_data = pd.read_csv("saved_models/ALSregistry.AdrianoChio.wrangled.nodates.freeze5.csv")
    umap_org_full = pd.read_csv('saved_models/ALS_NN_umap_org.csv', sep=',')
    umap_org_full = pd.merge(original_data, umap_org_full, left_on='number', right_on='number')
    replication_data = pd.read_csv("saved_models/ALSregistry.JessicaMandrioli.wrangled.nodates.freeze5.csv")
    umap_rep_full = pd.read_csv('saved_models/ALS_NN_umap_rep.csv', sep=',')
    umap_rep_full = pd.merge(original_data, umap_rep_full, left_on='number', right_on='number')
    colorable_columns_maps ={
        'clinicaltype_at_oneyear_y': "ALS clinical subtype at 1 year", 
        'elEscorialAtDx': "El Escorial category at diagnosis",
        'c9orf72_status_x': "C9orf72 Status",
        'firstALSFRS_total': "ALSFRS Score", 
        'familyHistoryOfALS': "Family History of ALS", 
    }
    colorable_columns = list(colorable_columns_maps) 
    
    colorable_columns = list(set(colorable_columns).intersection(set(list(umap_rep_full.columns))))
    st.write("### Select a factor to color according to the factor")
    select_color = st.selectbox('', [colorable_columns_maps[i] for i in colorable_columns], index=0)
    umap_org_full = umap_org_full.rename(columns=colorable_columns_maps) 
    umap_rep_full = umap_rep_full.rename(columns=colorable_columns_maps) 
    umap_org = umap_org_full[[select_color] + ['UMAP_3_1', 'UMAP_3_2', 'UMAP_3_3']].dropna()
    umap_rep = umap_rep_full[[select_color] + ['UMAP_3_1', 'UMAP_3_2', 'UMAP_3_3']].dropna()
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "magenta", "yellow", "pink", "grey", "black", "brown", "purple"]
    for e, classname in enumerate(sorted( list(set(umap_org[select_color]).union(set(umap_rep[select_color]))) ) ) :
        color_discrete_map[classname] = color_discrete_map_list[e%10] 

    
    if len(color_discrete_map) < 10:
        col1, col2 = st.columns(2)
        with col1:
            st.write('### Discovery Cohort')
            fig = px.scatter_3d(umap_org, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color, color_discrete_map=color_discrete_map, opacity=1, size=[4]*len(umap_org), height=600, width=600)
            fig.layout.update(showlegend=False)
            # fig.update_layout(legend=dict(
            #     orientation="h",
            #     yanchor="bottom",
            #     y=1.02,
            #     xanchor="right",
            #     x=1
            # ))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write('### Replication Cohort')
            fig = px.scatter_3d(umap_rep, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color, color_discrete_map=color_discrete_map,  opacity=1, size=[4]*len(umap_rep), height=600, width=600)
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
                ))
            st.plotly_chart(fig, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write('### Discovery Cohort')
            fig = px.scatter_3d(umap_org, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color,  opacity=1, size=[4]*len(umap_org), height=600, width=600)
            fig.layout.update(showlegend=False)
            # fig.update_layout(legend=dict(
            #     orientation="h",
            #     yanchor="bottom",
            #     y=1.02,
            #     xanchor="right",
            #     x=1
            # ))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write('### Replication Cohort')
            fig = px.scatter_3d(umap_rep, x='UMAP_3_1', y='UMAP_3_2', z='UMAP_3_3', color=select_color,  opacity=1, size=[4]*len(umap_rep), height=600, width=600)
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            st.plotly_chart(fig, use_container_width=True)
    