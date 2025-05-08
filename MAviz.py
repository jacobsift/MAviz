# streamlit_app.py (or MAviz.py)
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import io # Needed for reading uploaded file

# --- Configuration ---
# Define columns of interest (ensure these match your CSV)
MIMIC_PEPTIDE_COL = 'mimic_Peptide' 
HLA_COL = 'MHC'             
MIMIC_AFF_COL = 'mimic_Aff(nM)'
CANCER_AFF_COL = 'cancer_Aff(nM)'
MIMIC_EL_COL = 'mimic_%Rank_EL'
CANCER_EL_COL = 'cancer_%Rank_EL'

# Define HLA Categories
CRITICAL_HLAS = ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-A*03:01']
NICE_TO_HAVE_HLAS = ['HLA-A*01:01', 'HLA-B*07:02', 'HLA-B*35:01']

# Define Color Mapping 
COLOR_MAP = {'Mimic': '#E69F00', 'Cancer': '#56B4E9'} 

# --- Helper Functions ---

# @st.cache_data # Consider adding back if loading is slow after testing
def process_uploaded_data(uploaded_file_obj):
    """Loads and preprocesses data from an uploaded file object."""
    try:
        df = pd.read_csv(uploaded_file_obj) 
        st.success(f"Successfully loaded uploaded data! Shape: {df.shape}") 
        required_cols = [MIMIC_PEPTIDE_COL, HLA_COL, MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Error: Missing required columns: {missing_cols}")
            return None 
        for col in [MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]:
             if col in df.columns and df[col].dtype == 'object':
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        if HLA_COL in df.columns and df[HLA_COL].dtype == 'object':
            # Clean HLA names: strip whitespace AND handle potential variations like HLA-A:03-01 vs HLA-A*03:01 if necessary
            df[HLA_COL] = df[HLA_COL].str.strip()
            # Example correction (if needed, adjust based on your actual data):
            # df[HLA_COL] = df[HLA_COL].str.replace('HLA-A:03-01', 'HLA-A*03:01', regex=False) 
        return df
    except Exception as e:
        st.error(f"Error reading or processing the uploaded CSV file: {e}")
        return None 

def reshape_for_plotting(df, selected_metric):
    """Reshapes data for easier plotting of mimic vs cancer values."""
    if selected_metric == 'Affinity (nM)':
        id_vars = [MIMIC_PEPTIDE_COL, HLA_COL]
        value_vars = [MIMIC_AFF_COL, CANCER_AFF_COL]
        var_name = 'Source'
        value_name = 'Affinity (nM)'
        log_y = True 
    elif selected_metric == '%Rank EL':
        id_vars = [MIMIC_PEPTIDE_COL, HLA_COL]
        value_vars = [MIMIC_EL_COL, CANCER_EL_COL]
        var_name = 'Source'
        value_name = '%Rank EL'
        log_y = False 
    else: return pd.DataFrame(), False, None, None 

    required_plot_cols = id_vars + value_vars
    missing_plot_cols = [col for col in required_plot_cols if col not in df.columns]
    if missing_plot_cols:
        st.warning(f"Cannot generate plot for '{selected_metric}'. Missing columns: {missing_plot_cols}")
        return pd.DataFrame(), False, None, None

    other_id_vars = [col for col in df.columns if col not in value_vars and col not in id_vars]
    all_id_vars = id_vars + other_id_vars
        
    try:
        df_melted = pd.melt(df, id_vars=all_id_vars, value_vars=value_vars, 
                            var_name=var_name, value_name=value_name)
    except KeyError as e:
        st.error(f"Error during data reshaping (melt): Missing column {e}. Available columns: {df.columns.tolist()}")
        return pd.DataFrame(), False, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during data reshaping: {e}")
        return pd.DataFrame(), False, None, None

    df_melted[var_name] = df_melted[var_name].replace({
        MIMIC_AFF_COL: 'Mimic', CANCER_AFF_COL: 'Cancer',
        MIMIC_EL_COL: 'Mimic', CANCER_EL_COL: 'Cancer'
    })
    df_melted.dropna(subset=[value_name], inplace=True)
    return df_melted, log_y, value_name, var_name

def generate_plots(df_filtered, selected_metric, color_map):
    """Generates the box plot and histogram for a given metric with specified colors."""
    df_melted, use_log_scale, value_col_name, source_col_name = reshape_for_plotting(df_filtered, selected_metric)
    if df_melted.empty:
        st.warning(f"No valid data points found for '{selected_metric}' with current filters.")
        return None, None 
    fig_box, fig_hist = None, None 

    # --- Plot 1: Box Plot ---
    if df_melted[HLA_COL].nunique() > 0:
        num_hlas = df_melted[HLA_COL].nunique()
        # Adjust height dynamically, ensure minimum height
        plot_height = min(2500, max(400, 40 * num_hlas)) # Increased multiplier slightly
        try:
            if use_log_scale: hla_order = df_melted.groupby(HLA_COL)[value_col_name].median().sort_values().index.tolist()
            else: hla_order = sorted(df_melted[HLA_COL].unique())
            
            fig_box = px.box(
                df_melted, x=value_col_name, y=HLA_COL, color=source_col_name,
                title=f"{selected_metric} Distribution per HLA (Mimic vs Cancer)",
                labels={value_col_name: selected_metric, HLA_COL: "HLA Type", source_col_name: "Source"},
                orientation='h', log_x=use_log_scale, category_orders={HLA_COL: hla_order}, 
                height=plot_height, points=False, 
                color_discrete_map=color_map 
            )
            # *** MODIFICATION: Increase y-axis label size ***
            fig_box.update_yaxes(
                categoryorder='array', 
                categoryarray=hla_order, 
                tickfont=dict(size=12) # Increased size
            ) 
            fig_box.update_xaxes(title_text=selected_metric)
        except Exception as e: st.error(f"Error generating box plot for {selected_metric}: {e}")

    # --- Plot 2: Histogram ---
    try:
        fig_hist = px.histogram(
            df_melted, x=value_col_name, color=source_col_name,
            title=f"Overall Distribution of {selected_metric} (Mimic vs Cancer)",
            labels={value_col_name: selected_metric, source_col_name: "Source"},
            marginal="rug", histnorm='percent', barmode='overlay', opacity=0.7,
            log_x=use_log_scale,
            color_discrete_map=color_map 
        )
        fig_hist.update_layout(xaxis_title=selected_metric, yaxis_title="Percentage")
    except Exception as e: st.error(f"Error generating histogram for {selected_metric}: {e}")
    return fig_box, fig_hist

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") 
st.title("🧬 Mimic Peptide Binding Explorer")
st.markdown("Explore Affinity (nM) and %Rank EL distributions for selected mimics across different HLA types.")

# --- File Uploader ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader(
    "Upload your 'final_300_mimics_all_original_data.csv' file:", type=["csv"])

# Initialize session state for HLA selections if it doesn't exist
if 'hla_selected_states' not in st.session_state:
    st.session_state.hla_selected_states = {}

# --- Main Processing Logic ---
if uploaded_file is not None:
    df_original = process_uploaded_data(uploaded_file)
    if df_original is not None:
        
        # --- Sidebar for User Inputs ---
        st.sidebar.header("⚙️ Filters")

        if MIMIC_PEPTIDE_COL not in df_original.columns or HLA_COL not in df_original.columns:
            st.sidebar.error(f"Required columns ('{MIMIC_PEPTIDE_COL}', '{HLA_COL}') not found.")
            st.stop()

        # Mimic Peptide Selection
        mimic_list = ['All'] + sorted(df_original[MIMIC_PEPTIDE_COL].unique().tolist())
        selected_mimics = st.sidebar.multiselect(
            "Select Mimic Peptide(s):", options=mimic_list, default=['All'] )

        # Handle 'All' selection for mimics
        if 'All' in selected_mimics or not selected_mimics: 
            filtered_df_mimics = df_original.copy()
        else:
            filtered_df_mimics = df_original[df_original[MIMIC_PEPTIDE_COL].isin(selected_mimics)]

        # *** MODIFIED: HLA Category Selection with Individual Checkboxes ***
        st.sidebar.markdown("---") 
        st.sidebar.header("HLA Selection")
        
        all_available_hlas_in_filtered_data = sorted(filtered_df_mimics[HLA_COL].unique())
        
        # Categorize available HLAs
        critical_set = set(CRITICAL_HLAS)
        nice_to_have_set = set(NICE_TO_HAVE_HLAS)
        
        hlas_in_data_critical = sorted([h for h in all_available_hlas_in_filtered_data if h in critical_set])
        hlas_in_data_nice = sorted([h for h in all_available_hlas_in_filtered_data if h in nice_to_have_set])
        hlas_in_data_other = sorted([h for h in all_available_hlas_in_filtered_data if h not in critical_set and h not in nice_to_have_set])

        # Function to create expander with checkboxes and select/deselect all
        def create_hla_expander(title, hla_list, category_key):
            with st.sidebar.expander(f"{title} ({len(hla_list)} available)", expanded=False):
                # Select/Deselect All buttons
                col1, col2 = st.columns(2)
                select_all_key = f"select_all_{category_key}"
                deselect_all_key = f"deselect_all_{category_key}"
                
                if col1.button("Select All", key=select_all_key, use_container_width=True):
                    for hla in hla_list:
                        st.session_state.hla_selected_states[hla] = True
                if col2.button("Deselect All", key=deselect_all_key, use_container_width=True):
                     for hla in hla_list:
                        st.session_state.hla_selected_states[hla] = False
                        
                # Individual checkboxes
                for hla in hla_list:
                    # Initialize state if not present, default to True
                    if hla not in st.session_state.hla_selected_states:
                        st.session_state.hla_selected_states[hla] = True
                    # Create checkbox, linking its value to session state
                    st.session_state.hla_selected_states[hla] = st.checkbox(hla, value=st.session_state.hla_selected_states[hla], key=f"chk_{hla}")

        # Create the expanders
        create_hla_expander("Critical HLAs", hlas_in_data_critical, "critical")
        create_hla_expander("Nice-to-have HLAs", hlas_in_data_nice, "nice")
        create_hla_expander("Other HLAs", hlas_in_data_other, "other")

        # Build list of HLAs to display based on session state
        hlas_to_display = [hla for hla, selected in st.session_state.hla_selected_states.items() if selected and hla in all_available_hlas_in_filtered_data]
            
        # Filter DataFrame based on selected HLAs
        if not hlas_to_display: 
             filtered_df_final = pd.DataFrame(columns=filtered_df_mimics.columns) 
             st.sidebar.warning("No HLAs selected.")
        else:
             filtered_df_final = filtered_df_mimics[filtered_df_mimics[HLA_COL].isin(hlas_to_display)]
        
        st.sidebar.markdown("---") 
        
        # --- Checkboxes to control plot visibility ---
        st.sidebar.header("📊 Select Plots to Display")
        show_affinity_plots = st.sidebar.checkbox("Affinity (nM) Plots", value=True)
        show_el_rank_plots = st.sidebar.checkbox("%Rank EL Plots", value=True)
        
        st.sidebar.markdown("---") 
        st.sidebar.info(f"Displaying data for **{filtered_df_final[MIMIC_PEPTIDE_COL].nunique()}** mimics and **{filtered_df_final[HLA_COL].nunique()}** HLAs.")

        # --- Main Panel for Visualizations ---
        st.header("2. Explore Distributions")

        if filtered_df_final.empty:
            st.warning("No data matches the current filter selections (check Mimic and HLA selections).")
        else:
            # --- Affinity Plots ---
            if show_affinity_plots:
                st.markdown("### Affinity (nM) Distributions")
                aff_box_fig, aff_hist_fig = generate_plots(filtered_df_final, 'Affinity (nM)', COLOR_MAP)
                if aff_box_fig: st.plotly_chart(aff_box_fig, use_container_width=True)
                else: st.info("Box plot for Affinity could not be generated.")
                if aff_hist_fig: st.plotly_chart(aff_hist_fig, use_container_width=True)
                else: st.info("Histogram for Affinity could not be generated.")
                st.markdown("---") 

            # --- %Rank EL Plots ---
            if show_el_rank_plots:
                st.markdown("### %Rank EL Distributions")
                el_box_fig, el_hist_fig = generate_plots(filtered_df_final, '%Rank EL', COLOR_MAP)
                if el_box_fig: st.plotly_chart(el_box_fig, use_container_width=True)
                else: st.info("Box plot for %Rank EL could not be generated.")
                if el_hist_fig: st.plotly_chart(el_hist_fig, use_container_width=True)
                else: st.info("Histogram for %Rank EL could not be generated.")
                st.markdown("---") 
                
            if not show_affinity_plots and not show_el_rank_plots:
                 st.info("Select at least one plot type from the sidebar.")

            # --- Display Filtered Data Table ---
            st.header("3. Filtered Data View")
            display_cols = [MIMIC_PEPTIDE_COL, HLA_COL, MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL, 'cancer_Peptide', 'source_file_name']
            display_cols_exist = [col for col in display_cols if col in filtered_df_final.columns]
            st.dataframe(filtered_df_final[display_cols_exist])
            with st.expander("Show Full Data Table"):
                 st.dataframe(filtered_df_final)

else:
    st.info("Please upload the CSV data file to begin exploration.")

# --- How to Run ---
st.sidebar.markdown("---")
st.sidebar.header("How to Run")
st.sidebar.markdown("""
1.  **Save:** Save this code as `streamlit_app.py` (or `MAviz.py`).
2.  **Create `requirements.txt`:** Make a file named `requirements.txt` in the same directory with this content:
    ```txt
    streamlit
    pandas
    plotly
    numpy
    ```
3.  **GitHub:** Push `streamlit_app.py` and `requirements.txt` to a **public** GitHub repository. **Do NOT push the CSV data file.**
4.  **Deploy:** Go to [Streamlit Community Cloud](https://streamlit.io/cloud), connect your GitHub repo, select the repo/branch, ensure the main file path matches your script name (e.g., `streamlit_app.py` or `MAviz.py`), and deploy.
5.  **Use:** Open the deployed app URL. Use the file uploader to load your `final_300_mimics_all_original_data.csv` file.
6.  **Share:** Share the `.streamlit.app` URL with your colleagues. They will also need to upload the CSV file when they use the app.
""")
st.sidebar.markdown("---")

