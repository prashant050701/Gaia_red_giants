import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ks_2samp, anderson, mannwhitneyu
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

lick_gk = pd.read_csv("database/lick_GK_survey_with_gaia_id.csv")
express = pd.read_csv("database/express_post_MS_with_gaia_id.csv")
eapsnet1 = pd.read_csv("database/EAPSNet1_stellar_params_with_gaia_id.csv")
ppps = pd.read_csv("database/PPPS_star_with_gaia_id.csv")
eapsnet3 = pd.read_csv("database/EAPSNet3_stellar_params_with_gaia_id.csv")
eapsnet2 = pd.read_csv("database/EAPSNet2_stellar_params_with_gaia_id.csv")
coralie = pd.read_csv("database/coralie_star_with_gaia_id.csv")
ptps = pd.read_csv("database/ptps_with_gaia_id.csv")

gaia_data = {
    "Lick GK": pd.read_csv("database/lick_gk-result.csv"),
    "Express": pd.read_csv("database/express-result.csv"),
    "EAPSNet 1": pd.read_csv("database/eapsnet1-result.csv"),
    "PPPS": pd.read_csv("database/PPPS-result.csv"),
    "EAPSNet 3": pd.read_csv("database/eapsnet3-result.csv"),
    "EAPSNet 2": pd.read_csv("database/eapsnet2-result.csv"),
    "Coralie": pd.read_csv("database/coralie-result.csv"),
    "PTPS": pd.read_csv("database/ptps-result.csv"),
}

tess_data = {
    "Lick GK": pd.read_csv("database/TESS/lick_gk_unique_tic.csv"),
    "Express": pd.read_csv("database/TESS/express_tic.csv"),
    "EAPSNet 1": pd.read_csv("database/TESS/eapsnet1_tic.csv"),
    "PPPS": pd.read_csv("database/TESS/ppps_tic.csv"),
    "EAPSNet 3": pd.read_csv("database/TESS/eapsnet3_tic.csv"),
    "EAPSNet 2": pd.read_csv("database/TESS/eapsnet2_tic.csv"),
    "Coralie": pd.read_csv("database/TESS/coralie_tic.csv"),
    "PTPS": pd.read_csv("database/TESS/ptps_tic.csv"),
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

def plot_hr_diagram(data, teff_col, log_l_col, logg_col, title, log_conversion, use_cmap=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    luminosity = np.log10(data[log_l_col]) if log_conversion else data[log_l_col]
    if use_cmap:
        sc = ax.scatter(data[teff_col], luminosity, c=data[logg_col], cmap='viridis', alpha=0.5, s=10, edgecolor='k')
        plt.colorbar(sc, label="logg")
        ax.set_xlabel("Teff (K)")
        ax.set_ylabel("log(L/Lsun)")
        ax.set_title(title)
        plt.gca().invert_xaxis()
        st.pyplot(fig)
    else:
        ax.scatter(data[teff_col], luminosity, alpha=0.5, s=10, edgecolor='k')
        ax.set_xlabel("Teff (K)")
        ax.set_ylabel("GaiaMag")
        ax.set_title(title)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
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

def plot_combined_histogram(data_x, x_param, data_y, y_param, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data_x[x_param].dropna(), color='blue', kde=False, label=x_label, ax=ax)
    sns.histplot(data_y[y_param].dropna(), color='orange', kde=False, label=y_label, ax=ax)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)
    
def handle_scatter_plot(survey_x, x_data_source, x_param, survey_y, y_data_source, y_param):
    data_x = surveys[survey_x]["data"] if x_data_source == "Original" else gaia_data[survey_x] if x_data_source == "Gaia" else tess_data[survey_x]
    data_y = surveys[survey_y]["data"] if y_data_source == "Original" else gaia_data[survey_y] if y_data_source == "Gaia" else tess_data[survey_y]

    if survey_x == survey_y and x_data_source == y_data_source and x_param == y_param:
        plot_scatter(data_x, x_param, data_x, x_param, f"Scatter Plot - {survey_x} (Same Parameter)")
    else:
        suffixes = ('_x', '_y') if x_param == y_param else ('', '')
        merged_data = pd.merge(data_x[['source_id', x_param]], data_y[['source_id', y_param]], on='source_id', suffixes=suffixes)
        x_col = f"{x_param}{suffixes[0]}"
        y_col = f"{y_param}{suffixes[1]}"
        plot_scatter(merged_data, x_col, merged_data, y_col, f"Scatter Plot - {survey_x} vs {survey_y} ({x_data_source} vs {y_data_source})")

def perform_statistical_tests(data_x, x_param, data_y, y_param, test_type, auto=True, range_x=None, range_y=None):
    try:
        if not (is_numeric(data_x[x_param]) and is_numeric(data_y[y_param])):
            return None, "Selected parameters must be numeric for statistical tests."

        if auto:
            common_min = max(data_x[x_param].min(), data_y[y_param].min())
            common_max = min(data_x[x_param].max(), data_y[y_param].max())
            mask_x = (data_x[x_param] >= common_min) & (data_x[x_param] <= common_max)
            mask_y = (data_y[y_param] >= common_min) & (data_y[y_param] <= common_max)
        else:
            mask_x = (data_x[x_param] >= range_x[0]) & (data_x[x_param] <= range_x[1])
            mask_y = (data_y[y_param] >= range_y[0]) & (data_y[y_param] <= range_y[1])

        filtered_x = data_x[x_param][mask_x]
        filtered_y = data_y[y_param][mask_y]

        if filtered_x.empty or filtered_y.empty:
            return None, "No data available in the selected range for one or both parameters."

        if test_type == "KS":
            stat, pvalue = ks_2samp(filtered_x, filtered_y)
        elif test_type == "MWU":
            stat, pvalue = mannwhitneyu(filtered_x, filtered_y, alternative='two-sided')
        else:
            return None, "Invalid test type specified."

        return stat, pvalue

    except ValueError as e:
        return None, f"Error performing {test_type} test: {str(e)}"

def is_numeric(series):
    return series.dtype.kind in 'biufc'


st.title("Planetary Survey Data Analysis")

st.header("Section 1: HR Diagram")
survey1 = st.selectbox("Select Survey", list(surveys.keys()), key="survey1")

plot_original = st.checkbox("Plot HR Diagram from Original Survey")
plot_gaia = st.checkbox("Plot HR Diagram from Gaia")
plot_tess = st.checkbox("Plot HR Diagram from TESS")

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

if plot_tess:
    st.header("TESS HR Diagram")
    tess = tess_data[survey1]
    plot_hr_diagram(tess, "Teff", "GAIAmag", "logg", f"HR Diagram - {survey1} (TESS Data)", False, use_cmap=False)

st.header("Section 2: Distribution Plots")
survey2 = st.selectbox("Select Survey", list(surveys.keys()), key="survey2")
data_source2 = st.radio("Select Data Source", ["Original", "Gaia", "TESS"], key="data_source2")
columns2 = st.multiselect("Select Parameters to Plot",
                          surveys[survey2]["data"].columns if data_source2 == "Original" else gaia_data[survey2].columns if data_source2 == "Gaia" else tess_data[survey2].columns)
if st.button("Plot Distributions"):
    data = surveys[survey2]["data"] if data_source2 == "Original" else gaia_data[survey2] if data_source2 == "Gaia" else tess_data[survey2]
    plot_distribution(data, columns2, f"Distributions - {survey2} ({data_source2} Data)")

st.header("Section 3: Scatter Plots")
st.subheader("X-Axis Configuration")
survey_x = st.selectbox("Select Survey for X Parameter", list(surveys.keys()), key="survey_x")
x_data_source = st.radio("Select Data Source for X Parameter", ["Original", "Gaia", "TESS"], key="x_data_source")
x_param = st.selectbox("Select X Parameter",
                       surveys[survey_x]["data"].columns if x_data_source == "Original" else
                       gaia_data[survey_x].columns if x_data_source == "Gaia" else
                       tess_data[survey_x].columns, key="x_param")
st.subheader("Y-Axis Configuration")
survey_y = st.selectbox("Select Survey for Y Parameter", list(surveys.keys()), key="survey_y")
y_data_source = st.radio("Select Data Source for Y Parameter", ["Original", "Gaia", "TESS"], key="y_data_source")
y_param = st.selectbox("Select Y Parameter",
                       surveys[survey_y]["data"].columns if y_data_source == "Original" else
                       gaia_data[survey_y].columns if y_data_source == "Gaia" else
                       tess_data[survey_y].columns, key="y_param")

if st.button("Plot Scatter Plot"):
    handle_scatter_plot(survey_x, x_data_source, x_param, survey_y, y_data_source, y_param)

st.header("Section 4: Combined Histogram")
survey_x4 = st.selectbox("Select Survey for First Data Source", list(surveys.keys()), key="survey_x4")

x_data_source4 = st.radio("Select First Data Source", ["Original", "Gaia", "TESS"], key="x_data_source4")
x_param4 = st.selectbox("Select Parameter",
                        surveys[survey_x4]["data"].columns if x_data_source4 == "Original" else
                        gaia_data[survey_x4].columns if x_data_source4 == "Gaia" else
                        tess_data[survey_x4].columns, key="x_param4")

survey_y4 = st.selectbox("Select Survey for Second Data Source", list(surveys.keys()), key="survey_y4")
y_data_source4 = st.radio("Select Second Data Source", ["Original", "Gaia", "TESS"], key="y_data_source4")
y_param4 = st.selectbox("Select Parameter",
                        surveys[survey_y4]["data"].columns if y_data_source4 == "Original" else
                        gaia_data[survey_y4].columns if y_data_source4 == "Gaia" else
                        tess_data[survey_y4].columns, key="y_param4")

if st.button("Plot Combined Histogram"):
    data_x = surveys[survey_x4]["data"] if x_data_source4 == "Original" else gaia_data[survey_x4] if x_data_source4 == "Gaia" else tess_data[survey_x4]
    data_y = surveys[survey_y4]["data"] if y_data_source4 == "Original" else gaia_data[survey_y4] if y_data_source4 == "Gaia" else tess_data[survey_y4]

    valid_x_count = data_x[x_param4].dropna().shape[0]
    valid_y_count = data_y[y_param4].dropna().shape[0]

    st.write(f"Total valid entries for {x_param4} from {x_data_source4}: {valid_x_count}")
    st.write(f"Total valid entries for {y_param4} from {y_data_source4}: {valid_y_count}")

    x_label = survey_x4
    y_label = survey_y4

    plot_combined_histogram(data_x, x_param4, data_y, y_param4, f"Combined Histogram - {survey_x4} vs {survey_y4} ({x_data_source4} vs {y_data_source4})", x_label, y_label)

st.header("Section 5: Statistical Tests")

survey_x5 = st.selectbox("Select Survey for First Dataset", list(surveys.keys()), key="survey_x5")
data_source_x5 = st.radio("Select Data Source for First Dataset", ["Original", "Gaia", "TESS"], key="data_source_x5")
param_x5 = st.selectbox(
    "Select Parameter for First Dataset",
    surveys[survey_x5]["data"].columns if data_source_x5 == "Original" else
    gaia_data[survey_x5].columns if data_source_x5 == "Gaia" else tess_data[survey_x5].columns, key=f"param_x5_{survey_x5}_{data_source_x5}"
)

survey_y5 = st.selectbox("Select Survey for Second Dataset", list(surveys.keys()), key="survey_y5")
data_source_y5 = st.radio("Select Data Source for Second Dataset", ["Original", "Gaia", "TESS"], key="data_source_y5")
param_y5 = st.selectbox(
    "Select Parameter for Second Dataset",
    surveys[survey_y5]["data"].columns if data_source_y5 == "Original" else
    gaia_data[survey_y5].columns if data_source_y5 == "Gaia" else tess_data[survey_y5].columns, key=f"param_y5_{survey_y5}_{data_source_y5}"
)

data_x = surveys[survey_x5]["data"] if data_source_x5 == "Original" else gaia_data[survey_x5] if data_source_x5 == "Gaia" else tess_data[survey_x5]
data_y = surveys[survey_y5]["data"] if data_source_y5 == "Original" else gaia_data[survey_y5] if data_source_y5 == "Gaia" else tess_data[survey_y5]

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data_x[param_x5], nbinsx=50, name=f"{survey_x5} {data_source_x5}",
    marker=dict(line=dict(color='black', width=1))
))
fig.add_trace(go.Histogram(
    x=data_y[param_y5], nbinsx=50, name=f"{survey_y5} {data_source_y5}",
    marker=dict(line=dict(color='black', width=1))
))

fig.update_layout(barmode='overlay', title_text='Interactive Distribution Comparison')
fig.update_traces(opacity=0.6)
st.plotly_chart(fig)

test_type = st.radio("Select Test Type", ["KS", "MWU"], key="test_type")

manual_selection = st.checkbox("Manual Range Selection")

if manual_selection:
    if is_numeric(data_x[param_x5]) and is_numeric(data_y[param_y5]):
        min_x, max_x = float(data_x[param_x5].min()), float(data_x[param_x5].max())
        min_y, max_y = float(data_y[param_y5].min()), float(data_y[param_y5].max())
        range_x = st.slider("Select Range for First Dataset", min_x, max_x, (min_x, max_x))
        range_y = st.slider("Select Range for Second Dataset", min_y, max_y, (min_y, max_y))

        if st.button(f"Perform {test_type} Test on Selected Ranges"):
            stat, message = perform_statistical_tests(data_x, param_x5, data_y, param_y5, test_type, auto=False, range_x=range_x, range_y=range_y)
            if stat is not None:
                st.write(f"{test_type} Statistic: {stat:.4f}, P-value: {message:.4f}")
            else:
                st.error("No data available in the selected range for one or both parameters.")
    else:
        st.error("Selected parameters must be numeric to perform the selected statistical test and select ranges.")

if st.button(f"Perform {test_type} Test on Overlapping Ranges (Auto Mode)"):
    stat, message = perform_statistical_tests(data_x, param_x5, data_y, param_y5, test_type)
    if stat is not None:
        st.write(f"{test_type} Statistic: {stat:.4f}, P-value: {message:.4f}")
    else:
        st.error(message)
