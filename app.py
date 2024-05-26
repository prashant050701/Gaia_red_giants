import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

lick_gk = pd.read_csv("lick_GK_survey_with_gaia_id.csv")
express = pd.read_csv("express_post_MS_with_gaia_id.csv")
eapsnet1 = pd.read_csv("EAPSNet1_stellar_params_with_gaia_id.csv")
ppps = pd.read_csv("PPPS_star_with_gaia_id.csv")
eapsnet3 = pd.read_csv("EAPSNet3_stellar_params_with_gaia_id.csv")
eapsnet2 = pd.read_csv("EAPSNet2_stellar_params_with_gaia_id.csv")
coralie = pd.read_csv("coralie_star_with_gaia_id.csv")
ptps = pd.read_csv("ptps_with_gaia_id.csv")

gaia_data = {
    "Lick GK": pd.read_csv("lick_gk-result.csv"),
    "Express": pd.read_csv("express-result.csv"),
    "EAPSNet 1": pd.read_csv("eapsnet1-result.csv"),
    "PPPS": pd.read_csv("PPPS-result.csv"),
    "EAPSNet 3": pd.read_csv("eapsnet3-result.csv"),
    "EAPSNet 2": pd.read_csv("eapsnet2-result.csv"),
    "Coralie": pd.read_csv("coralie-result.csv"),
    "PTPS": pd.read_csv("ptps-result.csv"),
}

surveys = {
    "Lick GK": {"data": lick_gk, "Teff": "Teff", "log_L": "L*", "logg": "logg", "log_conversion": True},
    "Express": {"data": express, "Teff": "Teff", "log_L": "log_L", "logg": "logg", "log_conversion": False},
    "EAPSNet 1": {"data": eapsnet1, "Teff": "Teff", "log_L": "log_L", "logg": "log_g", "log_conversion": False},
    "PPPS": {"data": ppps, "Teff": "T_eff", "log_L": "log_L", "logg": "log g", "log_conversion": False},
    "EAPSNet 3": {"data": eapsnet3, "Teff": "Teff_BV", "log_L": "log_L", "logg": "logg_BV", "log_conversion": False},
    "EAPSNet 2": {"data": eapsnet2, "Teff": "T_eff", "log_L": "log_L", "logg": "log_g", "log_conversion": False},
    "Coralie": {"data": coralie, "Teff": "Teff", "log_L": "Lum", "logg": "logg", "log_conversion": True},
    "PTPS": {"data": ptps, "Teff": "Teff", "log_L": "logL", "logg": "logg", "log_conversion": False},
}

def plot_hr_diagram(data, teff_col, log_l_col, logg_col, title, log_conversion):
    fig, ax = plt.subplots(figsize=(8, 6))
    luminosity = np.log10(data[log_l_col]) if log_conversion else data[log_l_col]
    sc = ax.scatter(data[teff_col], luminosity, c=data[logg_col], cmap='viridis', alpha=0.5, s=10, edgecolor='k')
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("log(L/Lsun)" if log_conversion else "L/Lsun")
    ax.set_title(title)
    plt.colorbar(sc, label="logg")
    plt.gca().invert_xaxis()
    st.pyplot(fig)

def plot_distribution(data, columns, title):
    for column in columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data[column], kde=False, ax=ax)
        ax.set_title(f"{title}: {column}")
        st.pyplot(fig)

def plot_scatter(data_x, x_param, data_y, y_param, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data_x[x_param], data_y[y_param], alpha=0.5, s=10, edgecolor='k')
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(title)
    st.pyplot(fig)

def plot_combined_histogram(data_x, x_param, data_y, y_param, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data_x[x_param].dropna(), color='blue', kde=False, label='Original', ax=ax)
    sns.histplot(data_y[y_param].dropna(), color='orange', kde=False, label='Gaia', ax=ax)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

st.title("Planetary Survey Data Analysis")

st.header("Section 1: HR Diagram")
survey1 = st.selectbox("Select Survey", list(surveys.keys()), key="survey1")

plot_original = st.checkbox("Plot HR Diagram from Original Survey")
plot_gaia = st.checkbox("Plot HR Diagram from Gaia")

if plot_original or plot_gaia:
    col1, col2 = st.columns(2)
    if plot_original:
        with col1:
            data_info = surveys[survey1]
            plot_hr_diagram(data_info["data"], data_info["Teff"], data_info["log_L"], data_info["logg"],
                            f"HR Diagram - {survey1} (Original Survey)", data_info["log_conversion"])
    if plot_gaia:
        with col2:
            gaia = gaia_data[survey1]
            plot_hr_diagram(gaia, "effective_temperature", "luminosity", "surface_gravity",
                            f"HR Diagram - {survey1} (Gaia Data)", True)

st.header("Section 2: Distribution Plots")
survey2 = st.selectbox("Select Survey", list(surveys.keys()), key="survey2")
data_source2 = st.radio("Select Data Source", ["Original", "Gaia"], key="data_source2")
columns2 = st.multiselect("Select Parameters to Plot",
                          surveys[survey2]["data"].columns if data_source2 == "Original" else gaia_data[survey2].columns)
if st.button("Plot Distributions"):
    data = surveys[survey2]["data"] if data_source2 == "Original" else gaia_data[survey2]
    plot_distribution(data, columns2, f"Distributions - {survey2} ({data_source2} Data)")

st.header("Section 3: Scatter Plots")
survey3 = st.selectbox("Select Survey", list(surveys.keys()), key="survey3")
x_param3 = st.selectbox("Select X Parameter from Original Survey", surveys[survey3]["data"].columns, key="x_param3")
y_param3 = st.selectbox("Select Y Parameter from Gaia Data", gaia_data[survey3].columns, key="y_param3")
if st.button("Plot Scatter Plot"):
    data_x = surveys[survey3]["data"]
    data_y = gaia_data[survey3]

    merged_data = pd.merge(data_x[['source_id', x_param3]], data_y[['source_id', y_param3]], on='source_id')

    plot_scatter(merged_data, x_param3, merged_data, y_param3, f"Scatter Plot - {survey3} (Original vs Gaia)")

st.header("Section 4: Combined Histogram")
survey4 = st.selectbox("Select Survey", list(surveys.keys()), key="survey4")
x_param4 = st.selectbox("Select Parameter from Original Survey", surveys[survey4]["data"].columns, key="x_param4")
y_param4 = st.selectbox("Select Parameter from Gaia Data", gaia_data[survey4].columns, key="y_param4")

if st.button("Plot Combined Histogram"):
    data_x = surveys[survey4]["data"]
    data_y = gaia_data[survey4]

    valid_x_count = data_x[x_param4].dropna().shape[0]
    valid_y_count = data_y[y_param4].dropna().shape[0]

    st.write(f"Total valid entries for {x_param4} from Original: {valid_x_count}")
    st.write(f"Total valid entries for {y_param4} from Gaia: {valid_y_count}")

    plot_combined_histogram(data_x, x_param4, data_y, y_param4, f"Combined Histogram - {survey4} (Original vs Gaia)")
