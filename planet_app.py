import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#@st.cache
def load_combine_csv_files():
    parameters = ['radius', 'logg', 'luminosity', 'mass', 'parallax', 'teff', 'metallicity']
    data_dict = {}
    for param in parameters:
        file_1 = pd.read_csv(f"database/golden_sample/{param}_1.csv")
        file_2 = pd.read_csv(f"database/golden_sample/{param}_2.csv")
        combined_data = pd.concat([file_1, file_2], ignore_index=True)
        data_dict[param] = combined_data
    golden_sample = pd.concat(data_dict.values(), axis=1)
    return golden_sample
    
def load_all_data():
    data = pd.read_csv('database/updated_exoplanet_data.csv')
    data_ps_planet = pd.read_csv('database/updated_exoplanet_data.csv')
    data_gg = load_combine_csv_files()
    data_ps_all = pd.read_csv('database/all_planetary_survey_original_unique_with_host.csv')
    return data, data_ps_planet, data_gg, data_ps_all


def filter_data(df, section, survey, filter_type='main'):
    is_giant = False
    st.sidebar.subheader(f"Filters for Section {section}")
    giants_key = f"giants_only_{filter_type}_section_{section}"
    if st.sidebar.checkbox("Giants only", key=giants_key):
        is_giant = True
        if 'log_g' in df.columns:
            df = df[df['log_g'] < 3.75]
        elif 'logg' in df.columns:
            df = df[df['logg'] < 3.75]

        #return df

    survey_mapping = {
        'All': None,
        'Lick': 'lick_GK_survey_with_gaia_id.csv',
        'EAPSNet1': 'EAPSNet1_stellar_params_with_gaia_id.csv',
        'EAPSNet2': 'EAPSNet2_stellar_params_with_gaia_id.csv',
        'EAPSNet3': 'EAPSNet3_stellar_params_with_gaia_id.csv',
        'Keck HIRES': 'keck_hires_with_gaia_id.csv',
        'PTPS': 'ptps_with_gaia_id.csv',
        'PPPS': 'PPPS_star_with_gaia_id.csv',
        'Express': 'express_post_MS_with_gaia_id.csv',
        'Coralie': 'coralie_star_with_gaia_id.csv'
    }

    if survey != 'All':
        source_file = survey_mapping.get(survey, None)
        if source_file:
            df = df[df['source_file'] == source_file]

    return df, is_giant

def plot_histogram(data, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data[column].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    return fig

def calculate_occurrence_error(counts, total_stars):
    occurrence_rates = counts / total_stars
    errors = np.sqrt((occurrence_rates * (1 - occurrence_rates)) / total_stars)
    return errors

def plot_occurrence_rates(df, surveys_df, param1, param2, bin_edges_param1, bin_edges_param2, scale_param1, scale_param2, normalize=False, show_error=False, is_giant=False):
    
    filtered_data = df[[param1, param2]].dropna()
    counts, xedges, yedges = np.histogram2d(filtered_data[param1], filtered_data[param2], bins=[bin_edges_param1, bin_edges_param2])

    total_stars = surveys_df.shape[0]
    if is_giant:
        total_stars = surveys_df[surveys_df['logg'] <= 3.75].shape[0]
    
    occurrence_rates = counts / total_stars

    errors = calculate_occurrence_error(counts, total_stars)

    if normalize:
        param1_bin_sizes = np.diff(bin_edges_param1)
        param2_bin_sizes = np.diff(bin_edges_param2)
        occurrence_rates /= np.outer(param1_bin_sizes, param2_bin_sizes)
        errors /= np.outer(param1_bin_sizes, param2_bin_sizes)
        
    occurrence_rates *= 100
    errors *= 100

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(bin_edges_param2, bin_edges_param1, occurrence_rates, shading='auto', cmap='Greys', edgecolor='black', linewidth=1)
    mesh.set_facecolor("none")
    
    if 'log' in scale_param1.lower():
        ax.set_yscale('log')
        ax.set_yticks(bin_edges_param1)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        
    if 'log' in scale_param2.lower():
        ax.set_xscale('log')
        ax.set_xticks(bin_edges_param2)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.set_xticks(bin_edges_param2)
    ax.set_xticklabels([f'{x:.2f}' for x in bin_edges_param2])
    ax.set_yticks(bin_edges_param1)
    ax.set_yticklabels([f'{y:.2f}' for y in bin_edges_param1])

    for i in range(len(bin_edges_param1) - 1):
        for j in range(len(bin_edges_param2) - 1):
            
            if scale_param1 == "Logarithmic" and scale_param2 == "Logarithmic":
                x_center = np.sqrt(bin_edges_param2[j] * bin_edges_param2[j+1])
                y_center = np.sqrt(bin_edges_param1[i] * bin_edges_param1[i+1])
            elif scale_param1 == "Logarithmic":
                x_center = (bin_edges_param2[j] + bin_edges_param2[j+1]) / 2
                y_center = np.sqrt(bin_edges_param1[i] * bin_edges_param1[i+1])
            elif scale_param2 == "Logarithmic":
                x_center = np.sqrt(bin_edges_param2[j] * bin_edges_param2[j+1])
                y_center = (bin_edges_param1[i] + bin_edges_param1[i+1]) / 2
            else:
                x_center = (bin_edges_param2[j] + bin_edges_param2[j+1]) / 2
                y_center = (bin_edges_param1[i] + bin_edges_param1[i+1]) / 2

            percentage_value = occurrence_rates[i, j]
            ax.text(x_center, y_center, f'{percentage_value:.3f}%', color='black', ha='center', va='center', fontsize=15)

            if show_error:
                if scale_param1 == "Linear" and scale_param2 == "Linear":
                    error_y_position = y_center - 0.15 * (bin_edges_param1[i+1] - bin_edges_param1[i])
                elif scale_param1 == "Linear" and scale_param2 == "Logarithmic":
                    error_y_position = y_center - 0.15 * (bin_edges_param1[i+1] - bin_edges_param1[i])
                else:
                    error_y_position = y_center - 0.25 * y_center
                error_value = errors[i, j]
                ax.text(x_center, error_y_position, f'±{error_value:.3f}%', color='gray', ha='center', va='center', fontsize=10)


    ax.set_xlabel(param2, fontsize=15)
    ax.set_ylabel(param1, fontsize=15)
    
    ax.set_title('Normalized Planet Occurrence Rates' if normalize else 'Planet Occurrence Rates')

    return fig

    
def update_efficiency_plots(selected_data, data_gg, data_ps_planet, param1, param2, bins_x, bins_y):
    col1, _ = get_column_name_and_scale(param1, 'ps_all')
    col2, _ = get_column_name_and_scale(param2, 'ps_all')
    col1_gg, _ = get_column_name_and_scale(param1, 'gg')
    col2_gg, _ = get_column_name_and_scale(param2, 'gg')
    col1_ps, _ = get_column_name_and_scale(param1, 'ps')
    col2_ps, _ = get_column_name_and_scale(param2, 'ps')

    xedges = np.linspace(selected_data[col1].min(), selected_data[col1].max(), bins_x + 1)
    yedges = np.linspace(selected_data[col2].min(), selected_data[col2].max(), bins_y + 1)

    n_ps_counts = np.zeros((bins_x, bins_y))
    n_g_counts = np.zeros((bins_x, bins_y))
    n_ps_occ_counts = np.zeros((bins_x, bins_y))

    for i in range(bins_x):
        for j in range(bins_y):
            bin_x_min, bin_x_max = xedges[i], xedges[i + 1]
            bin_y_min, bin_y_max = yedges[j], yedges[j + 1]

            n_ps_counts[i, j] = selected_data[(selected_data[col1] >= bin_x_min) & (selected_data[col1] < bin_x_max) &
                                              (selected_data[col2] >= bin_y_min) & (selected_data[col2] < bin_y_max)].shape[0]
            n_g_counts[i, j] = data_gg[(data_gg[col1_gg] >= bin_x_min) & (data_gg[col1_gg] < bin_x_max) &
                                       (data_gg[col2_gg] >= bin_y_min) & (data_gg[col2_gg] < bin_y_max)].shape[0]
            n_ps_occ_counts[i, j] = data_ps_planet[(data_ps_planet[col1_ps] >= bin_x_min) & (data_ps_planet[col1_ps] < bin_x_max) &
                                                   (data_ps_planet[col2_ps] >= bin_y_min) & (data_ps_planet[col2_ps] < bin_y_max)].shape[0]

    total_ps_in_bins = n_ps_counts.sum()
    total_gg_in_bins = n_g_counts.sum()
    #total_ps_planet_in_bins = n_ps_occ_counts.sum()
    
    n_ps_norm = n_ps_counts / total_ps_in_bins if total_ps_in_bins > 0 else n_ps_counts
    n_g_norm = n_g_counts / total_gg_in_bins if total_gg_in_bins > 0 else n_g_counts
    occ_rate = n_ps_occ_counts / total_ps_in_bins if total_ps_in_bins > 0 else n_ps_occ_counts #dividing by stars with/without exoplanet to get occurrence rate

    # st.write(f"Stars with planets in each bin: {n_ps_occ_counts}")
    # st.write(f"Stars in each bin from Planet Search: {n_ps_counts}")
    # st.write(f"Stars in each bin from Gaia: {n_g_counts}")
    
    
    sigma_n_ps = np.sqrt((n_ps_norm * (1 - n_ps_norm)) / total_ps_in_bins)
    sigma_n_g = np.sqrt((n_g_norm * (1 - n_g_norm)) / total_gg_in_bins)
    sigma_occ_rate = np.sqrt((occ_rate * (1 - occ_rate)) / total_ps_in_bins)

    
    eta = np.divide(n_ps_norm, n_g_norm, out=np.zeros_like(n_ps_norm), where=n_g_norm != 0)
    sigma_eta = eta * np.sqrt((sigma_n_ps / n_ps_norm) ** 2 + (sigma_n_g / n_g_norm) ** 2)

    
    eta_new = np.divide(n_g_norm, n_ps_norm, out=np.zeros_like(n_g_norm), where=n_ps_norm != 0)
    sigma_eta_new = eta_new * np.sqrt((sigma_n_g / n_g_norm) ** 2 + (sigma_n_ps / n_ps_norm) ** 2)

    fig, ax = plt.subplots()
    for i in range(bins_x):
        for j in range(bins_y):
            eta_val = eta[i, j] if not np.isnan(eta[i, j]) else 0
            eta_new_val = eta_new[i,j] if not np.isnan(eta_new[i,j]) else 0
            x_center = (xedges[i] + xedges[i + 1]) / 2
            y_center = (yedges[j] + yedges[j + 1]) / 2
            ax.text(x_center, y_center, f'N_Occ: {occ_rate[i, j]:.4f}\nN_ps: {n_ps_norm[i, j]:.4f}\nN_g: {n_g_norm[i, j]:.4f}\n\u03B7: {eta_val:.4f}', color='blue', ha='center', va='center')
            #ax.text(x_center, y_center, f'N_Occ: {occ_rate[i, j]:.4f}\nN_ps: {n_ps_norm[i, j]:.4f}\nN_g: {n_g_norm[i, j]:.4f}\n\u03B7: {eta_new_val:.4f}', color='blue', ha='center', va='center')
            
    ax.set_xlim([xedges[0], xedges[-1]])
    ax.set_ylim([yedges[0], yedges[-1]])
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.grid(True)

    plt.xticks(xedges, np.round(xedges, 3))
    plt.yticks(yedges, np.round(yedges, 3))
    
    ax.set_title('Dynamic Efficiency Plot')
    st.pyplot(fig)
    return eta, eta_new, occ_rate, sigma_eta, sigma_eta_new, sigma_occ_rate


def section4_main(data_ps_all, data_gg, data_ps_planet):
    st.header("Section 4: Interactive Data Selection and Analysis")
    params = ['Mass', 'Teff', 'Fe/H', 'log_g', 'radius', 'parallax']
    st.sidebar.subheader("Section 4 Configuration")
    survey4 = st.sidebar.selectbox("Select Survey", ['All', 'Lick', 'EAPSNet1', 'EAPSNet2', 'EAPSNet3', 'Keck HIRES', 'PTPS', 'PPPS', 'Express', 'Coralie'], key='survey4')
    filtered_data_ps_all,_ = filter_data(data_ps_all.copy(), "4", survey4)
    filtered_data_ps_all = filtered_data_ps_all[filtered_data_ps_all['Teff'] != 0.0]
    x_param = st.sidebar.selectbox("Select X-axis Parameter", params, index=0, key="x_param_section4")
    y_param = st.sidebar.selectbox("Select Y-axis Parameter", params, index=1, key="y_param_section4")
    
    bins_x = st.sidebar.number_input("Number of bins for X-axis", min_value=1, value=3, key="bins_x_section4")
    bins_y = st.sidebar.number_input("Number of bins for Y-axis", min_value=1, value=3, key="bins_y_section4")
    
    x_col, x_scale = get_column_name_and_scale(x_param, 'ps_all')
    y_col, y_scale = get_column_name_and_scale(y_param, 'ps_all')
    

    fig = px.scatter(filtered_data_ps_all, x=x_col, y=y_col, title="Select data points for efficiency analysis")
    #fig = px.scatter(filtered_data_ps_all, x=x_col, y=y_col, title="Select data points for efficiency analysis",
                 #color=filtered_data_ps_all['host'].map({True: 'Host', False: 'Non-host'}))

    event_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    if event_data and "selection" in event_data:
        selected_indices = event_data["selection"]["point_indices"]
        if selected_indices:
            selected_data = filtered_data_ps_all.iloc[selected_indices]
            eta, eta_new, occ_rate, sigma_eta, sigma_eta_new, sigma_occ_rate = update_efficiency_plots(selected_data, data_gg, data_ps_planet, x_param, y_param, bins_x, bins_y)
            #eta_new = 1/eta
            k = 1
            log_eta_new = np.log10(k + eta_new)
            log_eta = np.log10(k + eta)
            # st.write("log_eta:", log_eta)

            total_occ_rate = np.sum(occ_rate)
            total_sigma_occ_rate = np.sqrt(np.sum(sigma_occ_rate**2))
            

            #corrected_occ_rate = np.sum(occ_rate / log_eta) / (np.sum(1/log_eta)) if np.sum(log_eta) > 0 else 0
            #corrected_occ_rate = np.sum(occ_rate/log_eta) / 1/(np.sum(log_eta)) if np.sum(log_eta) > 0 else 0
            #corrected_occ_rate = (np.sum(np.divide(occ_rate, log_eta, where=log_eta != 0)) / np.sum(np.divide(1, log_eta, where=log_eta != 0)) if np.sum(log_eta) > 0 else 0)
            #corrected_occ_rate = (np.sum(occ_rate * log_eta_new) / np.sum(log_eta) if np.sum(log_eta_new) > 0 else 0)
            #corrected_occ_rate = np.sum(log_eta_new * occ_rate) / np.sum(log_eta_new) if np.sum(log_eta_new) > 0 else 0
            corrected_occ_rate = np.sum(eta_new * occ_rate) #/ np.sum(eta_new) if np.sum(eta_new) > 0 else 0
            #corrected_occ_rate = np.sum(eta * occ_rate) / np.sum(eta) if np.sum(eta) > 0 else 0
            sigma_corrected_occ_rate = np.sqrt(np.sum((eta_new ** 2) * (sigma_occ_rate ** 2) + (occ_rate ** 2) * (sigma_eta_new ** 2)))
            #sigma_corrected_occ_rate = np.sqrt(np.sum((sigma_occ_rate / log_eta) ** 2 + (occ_rate * sigma_eta / ((k + eta) * np.log(10) * log_eta**2)) ** 2)) / np.sum(np.divide(1, log_eta, where=log_eta != 0))

            
            st.write(f"Uncorrected Occurrence Rate: {total_occ_rate:.6f} ± {total_sigma_occ_rate:.6f}")
            st.write(f"Corrected Occurrence Rate: {corrected_occ_rate:.6f} ± {sigma_corrected_occ_rate:.6f}")


            st.markdown("""
    **Corrected Occurrence Rate Formula:**
    $$
    %\\text{Corrected Occurrence Rate} = \\frac{\\sum \\left(N\\_{\\text{Occ}_i} \\times \\log\\left(k + \\frac{1}{\\eta_i}\\right)\\right)}{\\sum \\log\\left(k + \\frac{1}{\\eta_i}\\right)}
    %\\text{Corrected Occurrence Rate} = \\frac{\\sum \\left(\\frac{1}{\\log(k + \\eta_i)} \\times N\\_{\\text{Occ}_i}\\right)}{\\sum \\frac{1}{\\log(k + \\eta_i)}}
    %\\text{Corrected Occurrence Rate} = \\frac{\\sum \\left(\\log(k + \\frac{1}{\\eta_i}) \\times N\\_{\\text{Occ}_i}\\right)}{\\sum \\log(k + \\frac{1}{\\eta_i})}
    \\text{Corrected Occurrence Rate} = {\\sum \\left(\\frac{1}{\\eta_i} \\times N\\_{\\text{Occ}_i}\\right)}

    %\\text{Corrected Occurrence Rate} = \\frac{\\sum \\left(\\frac{N\\_{\\text{Occ}_i}}{\\eta_i}\\right)}{\\sum \\left(\\frac{1}{\\eta_i}\\right)}
    $$
    Where:
    - $\\eta_i$ is the efficiency of detection for bin $i$, defined as:
      $$
      \\eta_i = \\frac{N_{ps_i}}{N_{g_i}}
      $$
      - $N_{ps_i}$ is the normalised number of stars from the planetary survey in bin $i$.
      - $N_{g_i}$ is the normalised number of stars from the Gaia Golden Sample in bin $i$.
    - $N\\_{\\text{Occ}_i}$ is the occurrence rate for bin $i$, calculated as:
      $$
      N\\_{\\text{Occ}_i} = \\frac{N_{*p_i}}{\\sum N_{*}}
      $$
      Where:
      - $N_{*p_i}$ is the number of stars hosting a planet in bin $i$.
      - $\\sum N_{*}$ is the total number of stars in the survey across all bins of the selection.
    """)


        else:
            st.write("No data selected. Please select data points in the graph.")
    else:
        st.write("No data selected. Please select data points in the graph.")



def section2_settings(data, section):
    parameters = ['mass', 'radius', 'orbital_period', 'semi_major_axis', 'eccentricity']
    parameter1 = st.sidebar.selectbox(f"Select Parameter 1 for Analysis (Section {section})", parameters, key=f'param1_section_{section}')
    parameter2 = st.sidebar.selectbox(f"Select Parameter 2 for Analysis (Section {section})", [p for p in parameters if p != parameter1], key=f'param2_section_{section}')

    scale_param1 = st.sidebar.selectbox(f"Scale for {parameter1}", ["Linear", "Logarithmic"], key=f'scale_param1_section_{section}')
    scale_param2 = st.sidebar.selectbox(f"Scale for {parameter2}", ["Linear", "Logarithmic"], key=f'scale_param2_section_{section}')

    custom_scale_param1 = st.sidebar.radio(f"Custom scale for {parameter1}?", ["No", "Yes"], key=f'custom_scale_param1_section_{section}')
    custom_scale_param2 = st.sidebar.radio(f"Custom scale for {parameter2}?", ["No", "Yes"], key=f'custom_scale_param2_section_{section}')

    show_error = st.sidebar.checkbox("Show Error", value=False, key=f'show_error_section_{section}')

    if custom_scale_param1 == "Yes":
        min_param1 = st.sidebar.number_input(f"Minimum {parameter1}:", value=float(data[parameter1].min()), key=f'min_param1_section_{section}')
        max_param1 = st.sidebar.number_input(f"Maximum {parameter1}:", value=float(data[parameter1].max()), key=f'max_param1_section_{section}')
    else:
        min_param1, max_param1 = data[parameter1].min(), data[parameter1].max()

    if custom_scale_param2 == "Yes":
        min_param2 = st.sidebar.number_input(f"Minimum {parameter2}:", value=float(data[parameter2].min()), key=f'min_param2_section_{section}')
        max_param2 = st.sidebar.number_input(f"Maximum {parameter2}:", value=float(data[parameter2].max()), key=f'max_param2_section_{section}')
    else:
        min_param2, max_param2 = data[parameter2].min(), data[parameter2].max()

    bins_param1 = st.sidebar.number_input(f"Number of bins for {parameter1}", min_value=1, max_value=50, value=3, step=1, key=f'bins_param1_section_{section}')
    bins_param2 = st.sidebar.number_input(f"Number of bins for {parameter2}", min_value=1, max_value=50, value=3, step=1, key=f'bins_param2_section_{section}')

    if scale_param1 == "Logarithmic":
        bin_edges_param1 = np.logspace(np.log10(min_param1), np.log10(max_param1), bins_param1 + 1)
    else:
        bin_edges_param1 = np.linspace(min_param1, max_param1, bins_param1 + 1)

    if scale_param2 == "Logarithmic":
        bin_edges_param2 = np.logspace(np.log10(min_param2), np.log10(max_param2), bins_param2 + 1)
    else:
        bin_edges_param2 = np.linspace(min_param2, max_param2, bins_param2 + 1)

    return parameter1, parameter2, bin_edges_param1, bin_edges_param2, scale_param1, scale_param2, show_error

def section3_settings(data, section):
    parameters = ['Mass', 'Teff', 'Fe/H', 'log_g', 'radius', 'parallax']
    param1 = st.sidebar.selectbox(f"Select X-axis parameter (Section {section})", parameters, key=f'param1_section_{section}')
    param2 = st.sidebar.selectbox(f"Select Y-axis parameter (Section {section})", parameters, key=f'param2_section_{section}')

    #col1, scale1 = get_column_name_and_scale(param1, 'ps')
    #col2, scale2 = get_column_name_and_scale(param2, 'ps')
    col1, scale1 = get_column_name_and_scale(param1, 'ps_all')
    col2, scale2 = get_column_name_and_scale(param2, 'ps_all')

    custom_scale_param1 = st.sidebar.radio(f"Custom scale for {param1}?", ["No", "Yes"], key=f'custom_scale_param1_section_{section}')
    if custom_scale_param1 == "Yes":
        min_param1 = st.sidebar.number_input(f"Minimum {param1}:", value=float(data[col1].min()), key=f'min_param1_section_{section}')
        max_param1 = st.sidebar.number_input(f"Maximum {param1}:", value=float(data[col1].max()), key=f'max_param1_section_{section}')
    else:
        min_param1, max_param1 = data[col1].min(), data[col1].max()

    custom_scale_param2 = st.sidebar.radio(f"Custom scale for {param2}?", ["No", "Yes"], key=f'custom_scale_param2_section_{section}')
    if custom_scale_param2 == "Yes":
        min_param2 = st.sidebar.number_input(f"Minimum {param2}:", value=float(data[col2].min()), key=f'min_param2_section_{section}')
        max_param2 = st.sidebar.number_input(f"Maximum {param2}:", value=float(data[col2].max()), key=f'max_param2_section_{section}')
    else:
        min_param2, max_param2 = data[col2].min(), data[col2].max()

    show_error = st.sidebar.checkbox("Show Error", value=False, key=f'show_error_section_{section}')

    bins = st.sidebar.number_input('Number of bins', min_value=1, value=3, key=f'bins_section_{section}')
    xedges = np.linspace(min_param1, max_param1, bins + 1)
    yedges = np.linspace(min_param2, max_param2, bins + 1)

    return param1, param2, xedges, yedges, show_error


def main():
    st.title("Exoplanet Data Analysis")

    data, data_ps_planet, data_gg, data_ps_all = load_all_data()

    st.sidebar.subheader("Section 1: Histogram Filters")
    survey1 = st.sidebar.selectbox("Select Survey (Section 1)", ['All', 'Lick', 'EAPSNet1', 'EAPSNet2', 'EAPSNet3', 'Keck HIRES', 'PTPS', 'PPPS', 'Express', 'Coralie'], key='survey1')
    filtered_data,_ = filter_data(data.copy(), "1", survey1)
    st.header("Section 1: Histogram")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    column_to_plot = st.selectbox("Select Column for Histogram", numeric_columns, key='column_hist_1')
    hist_figure = plot_histogram(filtered_data, column_to_plot)
    st.pyplot(hist_figure)
    column_data = filtered_data[column_to_plot].dropna()
    range_of_data = f"{column_data.min()} - {column_data.max()}"
    median_of_data = column_data.median()
    std_deviation = column_data.std(ddof=1)

    st.write(f"Total number of samples: {len(column_data)}")
    st.write(f"Range of the {column_to_plot}: {range_of_data}")
    st.write(f"Median of the {column_to_plot}: {median_of_data}")
    st.write(f"Standard Deviation (Dispersion) of the {column_to_plot}: {std_deviation:.3f}")

    st.sidebar.subheader("Section 2: Planetary 2D Histogram Settings")
    parameter1, parameter2, bin_edges_param1, bin_edges_param2, scale_param1, scale_param2, show_error = section2_settings(data, "2")
    survey2 = st.sidebar.selectbox("Select Survey (Section 2)", ['All', 'Lick', 'EAPSNet1', 'EAPSNet2', 'EAPSNet3', 'Keck HIRES', 'PTPS', 'PPPS', 'Express', 'Coralie'], key='survey2')
    filtered_data,is_giant = filter_data(data.copy(), "2", survey2)
    
    st.header("Section 2: Occurrence Rate")
    occurrence_figure = plot_occurrence_rates(filtered_data, data_ps_all, parameter1, parameter2, bin_edges_param1, bin_edges_param2, scale_param1, scale_param2, normalize=False, show_error=show_error, is_giant=is_giant)
    st.pyplot(occurrence_figure)
    
    
    st.header("Section 3: Planetary Search Efficiency")
    st.sidebar.header('Section 3: Parameter Selection')
    survey3 = st.sidebar.selectbox("Select Survey (Section 3)", ['All', 'Lick', 'EAPSNet1', 'EAPSNet2', 'EAPSNet3', 'Keck HIRES', 'PTPS', 'PPPS', 'Express', 'Coralie'], key='survey3')
    
    #filtered_data_ps_planet,_ = filter_data(data_ps_planet.copy(), "3: Planetary Search Data", survey3)
    filtered_data_ps,_ = filter_data(data_ps_all.copy(), "3: Planetary Search Data", survey3)
    
    filtered_data_gg,_ = filter_data(data_gg.copy(), "3: Golden Sample Data", 'All')
    
    #param1, param2, xedges, yedges, show_error = section3_settings(filtered_data_ps_planet, "3")
    #col1_ps_planet, scale1_ps_planet = get_column_name_and_scale(param1, 'ps')
    #col2_ps_planet, scale2_ps_planet = get_column_name_and_scale(param2, 'ps')
    
    param1, param2, xedges, yedges, show_error = section3_settings(filtered_data_ps, "3")
    col1_ps, scale1_ps = get_column_name_and_scale(param1, 'ps_all')
    col2_ps, scale2_ps = get_column_name_and_scale(param2, 'ps_all')
    
    col1_gg, scale1_gg = get_column_name_and_scale(param1, 'gg')
    col2_gg, scale2_gg = get_column_name_and_scale(param2, 'gg')

    #data1_ps_planet, data2_ps_planet = filtered_data_ps_planet[col1_ps_planet], filtered_data_ps_planet[col2_ps_planet]
    
    data1_ps, data2_ps = filtered_data_ps[col1_ps], filtered_data_ps[col2_ps]
    
    data1_gg, data2_gg = filtered_data_gg[col1_gg], filtered_data_gg[col2_gg]

    bins = len(xedges) - 1
    #n_ps_planet_counts = np.zeros((bins, bins))
    n_ps_counts = np.zeros((bins, bins))
    n_g_counts = np.zeros((bins, bins))

    for i in range(bins):
        for j in range(bins):
            bin_x_min, bin_x_max = xedges[j], xedges[j + 1]
            bin_y_min, bin_y_max = yedges[i], yedges[i + 1]
            
            #n_ps_planet_counts[i, j] = np.sum((data1_ps_planet >= bin_x_min) & (data1_ps_planet < bin_x_max) & (data2_ps_planet >= bin_y_min) & (data2_ps_planet < bin_y_max))
            n_ps_counts[i, j] = np.sum((data1_ps >= bin_x_min) & (data1_ps < bin_x_max) & (data2_ps >= bin_y_min) & (data2_ps < bin_y_max))
            n_g_counts[i, j] = np.sum((data1_gg >= bin_x_min) & (data1_gg < bin_x_max) & (data2_gg >= bin_y_min) & (data2_gg < bin_y_max))

    #total_ps_planet_in_bins = np.sum(n_ps_planet_counts)
    total_ps_in_bins = np.sum(n_ps_counts)
    total_gg_in_bins = np.sum(n_g_counts)
    
    #n_ps_planet_norm = n_ps_planet_counts / total_ps_planet_in_bins if total_ps_planet_in_bins > 0 else n_ps_planet_counts #no. of stars with exoplanet/total of same from all bins
    n_ps_norm = n_ps_counts / total_ps_in_bins if total_ps_in_bins > 0 else n_ps_counts #no. of stars with or without exoplanet/total of same from all bins

    n_g_norm = n_g_counts / total_gg_in_bins if total_gg_in_bins > 0 else n_g_counts

    #Ask Prof if taking them as Binomial distribution is right?
    sigma_n_ps = np.sqrt((n_ps_norm * (1 - n_ps_norm)) / total_ps_in_bins)
    sigma_n_g = np.sqrt((n_g_norm * (1 - n_g_norm)) / total_gg_in_bins)

    #eta = np.divide(n_ps_planet_norm, n_g_norm, out=np.zeros_like(n_ps_planet_norm), where=n_g_norm != 0)
    eta = np.divide(n_ps_norm, n_g_norm, out=np.zeros_like(n_ps_norm), where=n_g_norm != 0)
    sigma_eta = eta * np.sqrt((sigma_n_ps / n_ps_norm) ** 2 + (sigma_n_g / n_g_norm) ** 2)
    
    fig, ax = plt.subplots()
    for i in range(bins):
        for j in range(bins):
            eta_val = eta[i, j] if not np.isnan(eta[i, j]) else 0
            sigma_eta_val = sigma_eta[i, j] if not np.isnan(sigma_eta[i, j]) else 0
            x_center = (xedges[j] + xedges[j + 1]) / 2
            y_center = (yedges[i] + yedges[i + 1]) / 2

            if show_error:
                n_ps_text = f'N_ps: {n_ps_norm[i, j]:.4f}'# ± {sigma_n_ps[i, j]:.4f}'
                n_g_text = f'N_g: {n_g_norm[i, j]:.4f}' #± {sigma_n_g[i, j]:.4f}'
                eta_text = f'\u03B7: {eta_val:.4f} ± {sigma_eta_val:.4f}'
                ax.text(x_center, y_center, f'{n_ps_text}\n{n_g_text}\n{eta_text}', color='blue', ha='center', va='center', fontsize=8)
            else:
                ax.text(x_center, y_center, f'N_ps: {n_ps_norm[i, j]:.4f}\nN_g: {n_g_norm[i, j]:.4f}\n\u03B7: {eta_val:.4f}',color='blue', ha='center', va='center')
            
            #ax.text(x_center, y_center, f'N_psOcc: {n_ps_planet_norm[i, j]:.4f}\nN_g: {n_g_norm[i, j]:.4f}\n\u03B7: {eta_val:.4f}',color='blue', ha='center', va='center')
            
            
    ax.set_xlim([xedges[0], xedges[-1]])
    ax.set_ylim([yedges[0], yedges[-1]])
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.grid(True)

    xedgeslabel = np.round(xedges, 3)
    yedgeslabel = np.round(yedges, 3)

    plt.xticks(xedges, xedgeslabel)
    plt.yticks(yedges, yedgeslabel)

    st.pyplot(fig)


    section4_main(data_ps_all, data_gg, data_ps_planet)

def get_column_name_and_scale(param, dataset):
    mapping = {
        'ps': {'Mass': ('star_mass', 'log'), 'Teff': ('star_teff', 'log'),
               'Fe/H': ('star_metallicity', 'linear'), 'log_g': ('log_g', 'linear'),
               'radius': ('star_radius', 'log'), 'parallax': ('parallax', 'log')},
        'gg': {'Mass': ('mass', 'log'), 'Teff': ('teff', 'log'),
               'Fe/H': ('metallicity', 'linear'), 'log_g': ('logg', 'linear'),
               'radius': ('radius', 'log'), 'parallax': ('parallax', 'log')},
        'ps_all': {'Mass': ('Mass', 'log'), 'Teff': ('Teff', 'log'),
                   'Fe/H': ('Fe/H', 'linear'), 'log_g': ('logg', 'linear'),
                   'radius': ('log_Radius', 'linear'), 'parallax': ('plx', 'log')}
    }
    return mapping[dataset][param]


if __name__ == "__main__":
    main()
