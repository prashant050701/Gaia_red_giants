import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title('HR Diagram of Exoplanet Host Stars')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    stars_df = pd.read_csv(uploaded_file)
    stars_df['source_id'] = stars_df['source_id'].astype(str)
    stars_df['log_L/Lsolar'] = np.log10(stars_df['luminosity'])
    st.sidebar.header('Filter options for HR Diagram')
    min_teff = st.sidebar.slider('Minimum Teff (K)', min_value=float(stars_df['teff'].min()),
                                 max_value=float(stars_df['teff'].max()), value=float(stars_df['teff'].min()))
    max_teff = st.sidebar.slider('Maximum Teff (K)', min_value=float(stars_df['teff'].min()),
                                 max_value=float(stars_df['teff'].max()), value=float(stars_df['teff'].max()))
    max_logg = st.sidebar.slider('Maximum log(g)', min_value=float(stars_df['logg'].min()),
                                 max_value=float(stars_df['logg'].max()), value=float(stars_df['logg'].max()))
    filtered_stars = stars_df[
        (stars_df['logg'] <= max_logg) & (stars_df['teff'] >= min_teff) & (stars_df['teff'] <= max_teff)]

    fig = px.scatter(filtered_stars, x='teff', y='log_L/Lsolar', color_continuous_scale='Viridis',
                     hover_data=['source_id', 'ra', 'dec', 'parallax', 'teff', 'luminosity', 'logg', 'metallicity'],
                     title='Interactive HR Diagram')
    fig.update_layout(xaxis_title='Effective Temperature (K)', yaxis_title='Log(L/Lsolar)', xaxis_autorange='reversed')
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, use_container_width=True)


    st.sidebar.header("Data Table Filter")
    min_teff_for_table = st.sidebar.slider("Min Teff for Table", min_value=float(stars_df["teff"].min()),
                                           max_value=float(stars_df["teff"].max()), value=float(stars_df["teff"].min()),
                                           key='min_teff_table')
    max_teff_for_table = st.sidebar.slider("Max Teff for Table", min_value=float(stars_df["teff"].min()),
                                           max_value=float(stars_df["teff"].max()), value=float(stars_df["teff"].max()),
                                           key='max_teff_table')
    min_log_Lsolar_for_table = st.sidebar.slider("Min Log(L/Lsolar) for Table",
                                                 min_value=float(stars_df["log_L/Lsolar"].min()),
                                                 max_value=float(stars_df["log_L/Lsolar"].max()),
                                                 value=float(stars_df["log_L/Lsolar"].min()),
                                                 key='min_log_Lsolar_table')
    max_log_Lsolar_for_table = st.sidebar.slider("Max Log(L/Lsolar) for Table",
                                                 min_value=float(stars_df["log_L/Lsolar"].min()),
                                                 max_value=float(stars_df["log_L/Lsolar"].max()),
                                                 value=float(stars_df["log_L/Lsolar"].max()),
                                                 key='max_log_Lsolar_table')

    table_filtered_stars = stars_df[(stars_df["teff"] >= min_teff_for_table) &
                                    (stars_df["teff"] <= max_teff_for_table) &
                                    (stars_df["log_L/Lsolar"] >= min_log_Lsolar_for_table) &
                                    (stars_df["log_L/Lsolar"] <= max_log_Lsolar_for_table)]


    st.sidebar.header("Select Columns to Display")
    available_columns = stars_df.columns.tolist()
    selected_columns = []
    for col in available_columns:
        if st.sidebar.checkbox(col, True):  
            selected_columns.append(col)

    if selected_columns:
        st.subheader('Data of Selected Stars')
        st.dataframe(table_filtered_stars[selected_columns])

        st.subheader('Histogram of Selected Column')
        column_for_histogram = st.selectbox("Select a column for the histogram:", options=selected_columns)


        if column_for_histogram:
            fig = px.histogram(table_filtered_stars, x=column_for_histogram,
                                title=f'Histogram of {column_for_histogram}')

            fig.update_traces(marker=dict(line=dict(width=1, color='black')))
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("No columns selected. Please select columns to display data.")
else:
    st.write("Please upload a file to view the HR diagram and filter stars.")
