import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
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
keck = pd.read_csv("database/keck_hires_with_gaia_id.csv")

golden_giant_ptps = pd.read_csv("database/golden_sample/golden_giant_ptps-result.csv")

all_data = pd.read_csv("database/all_planetary_survey_original.csv")

gaia_data = {
    "Lick GK": pd.read_csv("database/lick_gk-result.csv"),
    "Express": pd.read_csv("database/express-result.csv"),
    "EAPSNet 1": pd.read_csv("database/eapsnet1-result.csv"),
    "PPPS": pd.read_csv("database/PPPS-result.csv"),
    "EAPSNet 3": pd.read_csv("database/eapsnet3-result.csv"),
    "EAPSNet 2": pd.read_csv("database/eapsnet2-result.csv"),
    "Coralie": pd.read_csv("database/coralie-result.csv"),
    "PTPS": pd.read_csv("database/ptps-result.csv"),
    "Keck HIRES": pd.read_csv("database/keck_hires-result.csv")
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
    "Keck HIRES": pd.read_csv("database/TESS/keck_hires_tic.csv")
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
    "Keck HIRES": {"data": keck, "Teff": "Teff", "log_L": "log_L", "logg": "log(g)", "log_conversion": False}
}

def plot_hr_diagram(data, teff_col, log_l_col, logg_col, title, log_conversion, use_cmap=True):
    luminosity = np.log10(data[log_l_col]) if log_conversion else data[log_l_col]
    if use_cmap:
        fig = px.scatter(data, x=teff_col, y=luminosity, color=logg_col, color_continuous_scale='Viridis', labels={"color": "logg"}, title=title)
    else:
        fig = px.scatter(data, x=teff_col, y=luminosity, title=title)

    fig.update_xaxes(title="Teff (K)", autorange="reversed")
    
    if 'TESS' in title:
        fig.update_yaxes(title="V_mag")
    else:
        fig.update_yaxes(title="log(L/Lsun)")
    
    st.plotly_chart(fig, use_container_width=True)

        
def plot_distribution(data, columns, title):
    for column in columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data[column], kde=False, ax=ax)
        ax.set_title(f"{title}: {column}")
        
        try:
            valid_data = data[column].dropna()
            if is_numeric(valid_data):
                min_val = valid_data.min()
                max_val = valid_data.max()
                median_val = valid_data.median()
                std_val = valid_data.std()

                stats_text = f"Min: {min_val:.3f}, Max: {max_val:.3f}, Median: {median_val:.4f}, Sigma: {std_val:.4f}"
                ax.annotate(stats_text, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top')
            else:
                ax.annotate("Selected data is non-numeric", xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top')
        except Exception as e:
            ax.annotate(f"Error calculating statistics: {str(e)}", xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top')
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
    try:
        x_stats = data_x[x_param].dropna()
        y_stats = data_y[y_param].dropna()
        if is_numeric(x_stats) and is_numeric(y_stats):
            stats_x = f"{x_label} - Min: {x_stats.min():.3f}, Max: {x_stats.max():.3f}, Median: {x_stats.median():.4f}, Sigma: {x_stats.std():.4f}"
            stats_y = f"{y_label} - Min: {y_stats.min():.3f}, Max: {y_stats.max():.3f}, Median: {y_stats.median():.4f}, Sigma: {y_stats.std():.4f}"
            st.write(stats_x)
            st.write(stats_y)
        else:
            st.write("Selected data is non-numeric. Unable to calculate statistics.")
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
    st.pyplot(fig)
    
def handle_scatter_plot(survey_x, x_data_source, x_param, survey_y, y_data_source, y_param):
    data_x = surveys[survey_x]["data"] if x_data_source == "Original" else gaia_data[survey_x] if x_data_source == "Gaia" else tess_data[survey_x]
    data_y = surveys[survey_y]["data"] if y_data_source == "Original" else gaia_data[survey_y] if y_data_source == "Gaia" else tess_data[survey_y]

    data_x = data_x.drop_duplicates(subset='source_id', keep='first')
    data_y = data_y.drop_duplicates(subset='source_id', keep='first')
    y_param_map = data_y.set_index('source_id')[y_param].to_dict()

    data_x['y_param_mapped'] = data_x['source_id'].map(y_param_map)
    
    filtered_data_x = data_x.dropna(subset=['y_param_mapped'])
    filtered_data_y = pd.DataFrame()
    filtered_data_y[y_param] = filtered_data_x['y_param_mapped']
    
    title = f"Scatter Plot - {survey_x} vs {survey_y} ({x_data_source} vs {y_data_source})"
    
    plot_scatter(filtered_data_x, x_param, filtered_data_y, y_param, title)



        
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
survey1 = st.selectbox("Select Survey", ["All Surveys"] + list(surveys.keys()), key="survey1")
plot_original = st.checkbox("Original Survey")
plot_gaia = st.checkbox("Gaia")
plot_tess = st.checkbox("TESS")

if plot_original or plot_gaia or plot_tess:
    if survey1 == "All Surveys":
        if plot_original:
            plot_hr_diagram(all_data, "Teff", "Lum", "logg", "HR Diagram - All Surveys (Original Data)", log_conversion=False)
        if plot_gaia:
            combined_gaia = pd.concat([gaia_data[key] for key in gaia_data])
            plot_hr_diagram(combined_gaia, "effective_temperature", "luminosity", "surface_gravity",
                            "HR Diagram - All Surveys (Gaia Data)", True)
        if plot_tess:
            combined_tess = pd.concat([tess_data[key] for key in tess_data])
            plot_hr_diagram(combined_tess, "Teff", "GAIAmag", "logg", "HR Diagram - All Surveys (TESS Data)", False, use_cmap=False)
   
    else:
        if plot_original:
            data_info = surveys[survey1]
            plot_hr_diagram(data_info["data"], data_info["Teff"], data_info["log_L"], data_info["logg"],
                            f"HR Diagram - {survey1} (Original Survey)", data_info["log_conversion"])
        if plot_gaia:
            gaia = gaia_data[survey1]
            plot_hr_diagram(gaia, "effective_temperature", "luminosity", "surface_gravity",
                            f"HR Diagram - {survey1} (Gaia Data)", True)
        if plot_tess:
            tess = tess_data[survey1]
            plot_hr_diagram(tess, "Teff", "GAIAmag", "logg", f"HR Diagram - {survey1} (TESS Data)", False, use_cmap=False)

st.header("Section 2: Distribution Plots")

survey2 = st.selectbox("Select Survey", ["All Surveys"] + list(surveys.keys()), key="survey2")
plot_original2 = st.checkbox("Plot Distributions from Original Survey")
plot_gaia2 = st.checkbox("Plot Distributions from Gaia")
plot_tess2 = st.checkbox("Plot Distributions from TESS")

columns2 = []
if survey2 == "All Surveys":
    if plot_original2:
        all_data = pd.read_csv("database/all_planetary_survey_original.csv")
        columns2 = all_data.columns
    elif plot_gaia2:
        combined_gaia = pd.concat([gaia_data[key] for key in gaia_data])
        columns2 = combined_gaia.columns
    elif plot_tess2:
        combined_tess = pd.concat([tess_data[key] for key in tess_data])
        columns2 = combined_tess.columns
else:
    if plot_original2:
        columns2 = surveys[survey2]["data"].columns
    elif plot_gaia2:
        columns2 = gaia_data[survey2].columns
    elif plot_tess2:
        columns2 = tess_data[survey2].columns

selected_columns = st.multiselect("Select Parameters to Plot", columns2)

if st.button("Plot Distributions"):
    if survey2 == "All Surveys":
        if plot_original2:
            all_data = pd.read_csv("database/all_planetary_survey_original.csv")
            plot_distribution(all_data, selected_columns, f"Distributions - All Surveys (Original Data)")
        if plot_gaia2:
            combined_gaia = pd.concat([gaia_data[key] for key in gaia_data])
            plot_distribution(combined_gaia, selected_columns, f"Distributions - All Surveys (Gaia Data)")
        if plot_tess2:
            combined_tess = pd.concat([tess_data[key] for key in tess_data])
            plot_distribution(combined_tess, selected_columns, f"Distributions - All Surveys (TESS Data)")
    else:
        if plot_original2:
            data_info = surveys[survey2]
            plot_distribution(data_info["data"], selected_columns, f"Distributions - {survey2} (Original Survey)")
        if plot_gaia2:
            gaia = gaia_data[survey2]
            plot_distribution(gaia, selected_columns, f"Distributions - {survey2} (Gaia Data)")
        if plot_tess2:
            tess = tess_data[survey2]
            plot_distribution(tess, selected_columns, f"Distributions - {survey2} (TESS Data)")

    
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
    data_x = data_x.drop_duplicates(subset='source_id', keep='first')
    data_y = data_y.drop_duplicates(subset='source_id', keep='first')
    valid_x_count = data_x[x_param4].dropna().shape[0]
    valid_y_count = data_y[y_param4].dropna().shape[0]
    st.write(f"Total valid entries for {survey_x4} {x_param4} from {x_data_source4}: {valid_x_count}")
    st.write(f"Total valid entries for {survey_y4} {y_param4} from {y_data_source4}: {valid_y_count}")
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
    #histnorm='probability density', #will comment this later if count is needed
    marker=dict(line=dict(color='black', width=1))
))
fig.add_trace(go.Histogram(
    x=data_y[param_y5], nbinsx=50, name=f"{survey_y5} {data_source_y5}",
    #histnorm='probability density', #will comment this later if count is needed
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

        if st.button(f"Perform {test_type} Test on Selected Ranges", key="perform_test_section5"):
            stat, message = perform_statistical_tests(data_x, param_x5, data_y, param_y5, test_type, auto=False, range_x=range_x, range_y=range_y)
            if stat is not None:
                st.write(f"{test_type} Statistic: {stat:.4f}, P-value: {message:.4f}")
            else:
                st.error("No data available in the selected range for one or both parameters.")
    else:
        st.error("Selected parameters must be numeric to perform the selected statistical test and select ranges.")
if st.button(f"Perform {test_type} Test on Overlapping Ranges (Auto Mode)", key="auto_test_section5"):
    stat, message = perform_statistical_tests(data_x, param_x5, data_y, param_y5, test_type)
    if stat is not None:
        st.write(f"{test_type} Statistic: {stat:.4f}, P-value: {message:.4f}")
    else:
        st.error(message)


st.header("Section 6: Golden Giants")

golden_giant_ptps['log_L'] = np.log10(golden_giant_ptps['luminosity'])

fig = px.scatter(
    golden_giant_ptps,
    x='teff',
    y='log_L',
    color='logg',
    labels={
        "teff": "Effective Temperature (Teff)",
        "log_L": "Logarithm of Luminosity (log L)",
        "logg": "Surface Gravity (logg)"
    },
    title="HR Diagram of Golden Giant Data",
    color_continuous_scale=px.colors.sequential.Viridis,
    range_color=[golden_giant_ptps['logg'].min(), golden_giant_ptps['logg'].max()]
)

fig.update_xaxes(autorange="reversed")

st.plotly_chart(fig, use_container_width=False)


survey_6 = st.selectbox("Select Survey for First Dataset", ["All Surveys"] + list(surveys.keys()), key="survey_6")
if survey_6 == "All Surveys":
    param_6 = st.selectbox("Select Parameter for First Dataset", all_data.columns, key="param_6")
    data_6 = all_data
    
else:
    data_source_6 = st.radio("Select Data Source for First Dataset", ["Original", "Gaia", "TESS"], key="data_source_6")
    data_6 = surveys[survey_6]["data"] if data_source_6 == "Original" else gaia_data[survey_6] if data_source_6 == "Gaia" else tess_data[survey_6]
    param_6 = st.selectbox("Select Parameter for First Dataset", data_6.columns, key="param_6")
    
param_golden = st.selectbox("Select Parameter from Golden Giant Data", golden_giant_ptps.columns, key="param_golden")

#if st.button("Plot Interactive Histograms", key="plot_interactive_histograms_6"): # if needed as a button, will uncomment this
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data_6[param_6], nbinsx=50, name=f"{survey_6} - {param_6}",
    histnorm='probability density', #will comment this later if count is needed
    marker=dict(color='blue', line=dict(color='black', width=1))
))
fig.add_trace(go.Histogram(
    x=golden_giant_ptps[param_golden], nbinsx=50, name="Golden Giant - {param_golden}",
    histnorm='probability density', #will comment this later if count is needed
    marker=dict(color='red', line=dict(color='black', width=1))
))
fig.update_layout(
    barmode='overlay',
    title_text='Interactive Distribution Comparison - Section 6',
    xaxis_title_text='Value',
    yaxis_title_text='Density', #count
)
fig.update_traces(opacity=0.6)
st.plotly_chart(fig, use_container_width=True)


test_type_6 = st.radio("Select Test Type for Comparison", ["KS", "MWU"], key="test_type_6")

if survey_6 == "All Surveys":
    data_6 = all_data
else:
    data_6 = surveys[survey_6]["data"] if data_source_6 == "Original" else gaia_data[survey_6] if data_source_6 == "Gaia" else tess_data[survey_6]

manual_selection_6 = st.checkbox("Manual Range Selection", key="manual_selection_6")

if manual_selection_6:
    if is_numeric(data_6[param_6]) and is_numeric(golden_giant_ptps[param_golden]):
        min_x6, max_x6 = float(data_6[param_6].min()), float(data_6[param_6].max())
        min_y6, max_y6 = float(golden_giant_ptps[param_golden].min()), float(golden_giant_ptps[param_golden].max())
        range_x6 = st.slider("Select Range for First Dataset", min_x6, max_x6, (min_x6, max_x6), key="range_x6")
        range_y6 = st.slider("Select Range for Golden Giant Dataset", min_y6, max_y6, (min_y6, max_y6), key="range_y6")

        if st.button(f"Perform {test_type_6} Test on Selected Ranges", key="perform_test_section6"):
            stat_6, message_6 = perform_statistical_tests(data_6, param_6, golden_giant_ptps, param_golden, test_type_6, auto=False, range_x=range_x6, range_y=range_y6)
            if stat_6 is not None:
                st.write(f"{test_type_6} Statistic: {stat_6:.4f}, P-value: {message_6:.4f}")
            else:
                st.error("No data available in the selected range for one or both parameters.")
    else:
        st.error("Selected parameters must be numeric to perform the selected statistical test and select ranges.")

if st.button(f"Perform {test_type_6} Test on Overlapping Ranges (Auto Mode)", key="auto_test_section6"):
    stat_6, message_6 = perform_statistical_tests(data_6, param_6, golden_giant_ptps, param_golden, test_type_6)
    if stat_6 is not None:
        st.write(f"{test_type_6} Statistic: {stat_6:.4f}, P-value: {message_6:.4f}")
    else:
        st.error(message_6)
