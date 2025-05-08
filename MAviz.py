# streamlit_app.py (or MAviz.py)
import streamlit as st
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go # No longer needed
import os
import numpy as np
import io 

# --- Configuration ---
MIMIC_PEPTIDE_COL = 'mimic_Peptide' 
CANCER_PEPTIDE_COL = 'cancer_Peptide' 
CANCER_ACC_COL = 'cancer_acc' 
HLA_COL = 'MHC'             
MIMIC_AFF_COL = 'mimic_Aff(nM)'
CANCER_AFF_COL = 'cancer_Aff(nM)'
MIMIC_EL_COL = 'mimic_%Rank_EL'
CANCER_EL_COL = 'cancer_%Rank_EL'

CRITICAL_HLAS = ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-A*03:01']
NICE_TO_HAVE_HLAS = ['HLA-A*01:01', 'HLA-B*07:02', 'HLA-B*35:01']

COLOR_MAP = {'Mimic': '#E69F00', 'Cancer': '#56B4E9'} 

# --- Helper Functions ---

# @st.cache_data 
def process_uploaded_data(uploaded_file_obj):
    """Loads and preprocesses data from an uploaded file object."""
    try:
        df = pd.read_csv(uploaded_file_obj) 
        st.success(f"Successfully loaded uploaded data! Shape: {df.shape}") 
        required_cols = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL, MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Error: Missing required columns: {missing_cols}")
            return None 
        for col in [MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]:
             if col in df.columns and df[col].dtype == 'object':
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        if HLA_COL in df.columns and df[HLA_COL].dtype == 'object':
            df[HLA_COL] = df[HLA_COL].str.strip()
        for col in [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL]:
            if col in df.columns and df[col].dtype == 'object':
                 df[col] = df[col].str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading or processing the uploaded CSV file: {e}")
        return None 

def reshape_for_plotting(df, selected_metric):
    """Reshapes data for easier plotting of mimic vs cancer values."""
    id_vars = [MIMIC_PEPTIDE_COL, HLA_COL]
    if CANCER_PEPTIDE_COL in df.columns: id_vars.append(CANCER_PEPTIDE_COL)
    if CANCER_ACC_COL in df.columns: id_vars.append(CANCER_ACC_COL)
        
    if selected_metric == 'Affinity (nM)':
        value_vars = [MIMIC_AFF_COL, CANCER_AFF_COL]
        var_name = 'Source'
        value_name = 'Affinity (nM)'
        log_y = True # Use log_x in horizontal plot
    elif selected_metric == '%Rank EL':
        value_vars = [MIMIC_EL_COL, CANCER_EL_COL]
        var_name = 'Source'
        value_name = '%Rank EL'
        log_y = False # Use log_x in horizontal plot
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

# *** MODIFIED generate_plots function ***
def generate_plots(df_filtered, selected_metric, color_map, num_selected_mimics):
    """
    Generates the box plot and histogram for a given metric with specified colors.
    Adjusts box plot point visibility based on the number of selected mimics.
    """
    df_melted, use_log_scale, value_col_name, source_col_name = reshape_for_plotting(df_filtered, selected_metric)
    if df_melted.empty:
        st.warning(f"No valid data points found for '{selected_metric}' with current filters.")
        return None, None 
    fig_box, fig_hist = None, None 

    # --- Plot 1: Box Plot ---
    if df_melted[HLA_COL].nunique() > 0:
        num_hlas = df_melted[HLA_COL].nunique()
        plot_height = min(2500, max(400, 40 * num_hlas)) 
        
        # Determine point visibility based on mimic selection
        box_points_setting = 'all' if num_selected_mimics == 1 else False 
        
        try:
            # Determine sorting order for HLAs
            if use_log_scale: 
                median_vals = df_melted.groupby(HLA_COL)[value_col_name].median()
                # Handle cases where median might be NaN or Inf after log transform if data is sparse/extreme
                hla_order = median_vals.replace([np.inf, -np.inf], np.nan).dropna().sort_values().index.tolist()
                missing_hlas = [h for h in df_melted[HLA_COL].unique() if h not in hla_order]
                hla_order.extend(sorted(missing_hlas)) 
            else: 
                hla_order = sorted(df_melted[HLA_COL].unique())
            
            fig_box = px.box(
                df_melted, x=value_col_name, y=HLA_COL, color=source_col_name,
                title=f"{selected_metric} Distribution per HLA (Mimic vs Cancer)",
                labels={value_col_name: selected_metric, HLA_COL: "HLA Type", source_col_name: "Source"},
                orientation='h', log_x=use_log_scale, category_orders={HLA_COL: hla_order}, 
                height=plot_height, 
                points=box_points_setting, # Use the conditional setting
                color_discrete_map=color_map 
            )
            
            # Update layout for clarity
            fig_box.update_yaxes(
                categoryorder='array', 
                categoryarray=hla_order, 
                tickfont=dict(size=12),
                showgrid=False, # Explicitly set to False as requested
                # gridwidth=1, 
                # gridcolor='LightGrey'
            ) 
            fig_box.update_xaxes(title_text=selected_metric)
            fig_box.update_layout(
                margin=dict(l=150), # Add left margin for long HLA labels if needed
                # Ensure background is transparent if needed, or set explicitly
                # paper_bgcolor='rgba(0,0,0,0)', 
                # plot_bgcolor='rgba(0,0,0,0)' 
                )

            # Dashed lines removed
                 
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

# Initialize session state 
if 'hla_selected_states' not in st.session_state: st.session_state.hla_selected_states = {}
if 'cancer_acc_selected_states' not in st.session_state: st.session_state.cancer_acc_selected_states = {}

# --- Main Processing Logic ---
if uploaded_file is not None:
    df_original = process_uploaded_data(uploaded_file)
    if df_original is not None:
        
        # --- Sidebar for User Inputs ---
        st.sidebar.header("⚙️ Filters")

        req_cols_check = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL]
        if not all(col in df_original.columns for col in req_cols_check):
            st.sidebar.error(f"Required columns missing: Need {req_cols_check}")
            st.stop()

        # Mimic Peptide Selection
        mimic_list = ['All'] + sorted(df_original[MIMIC_PEPTIDE_COL].unique().tolist())
        selected_mimics_list = st.sidebar.multiselect( 
            "Select Mimic Peptide(s):", options=mimic_list, default=['All'] )

        # Determine number of selected mimics (excluding 'All')
        num_mimics_selected_for_plot = 0
        if 'All' not in selected_mimics_list and selected_mimics_list:
            num_mimics_selected_for_plot = len(selected_mimics_list)
        elif 'All' in selected_mimics_list or not selected_mimics_list:
             num_mimics_selected_for_plot = df_original[MIMIC_PEPTIDE_COL].nunique() 

        if 'All' in selected_mimics_list or not selected_mimics_list: filtered_df_step1 = df_original.copy()
        else: filtered_df_step1 = df_original[df_original[MIMIC_PEPTIDE_COL].isin(selected_mimics_list)]

        # Cancer Peptide Selection
        st.sidebar.markdown("---") 
        cancer_peptide_list = ['All'] + sorted(filtered_df_step1[CANCER_PEPTIDE_COL].unique().tolist())
        selected_cancer_peptides = st.sidebar.multiselect(
             "Select Cancer Peptide(s):", options=cancer_peptide_list, default=['All'] )
        
        if 'All' not in selected_cancer_peptides and selected_cancer_peptides:
             filtered_df_step2 = filtered_df_step1[filtered_df_step1[CANCER_PEPTIDE_COL].isin(selected_cancer_peptides)]
        else: filtered_df_step2 = filtered_df_step1.copy()

        # Cancer Accession (Gene) Selection
        st.sidebar.markdown("---")
        st.sidebar.header("Cancer Accession (Gene) Selection")
        available_cancer_accs = sorted(filtered_df_step2[CANCER_ACC_COL].unique())
        if not available_cancer_accs:
             st.sidebar.info("No Cancer Accessions available.")
             filtered_df_step3 = filtered_df_step2.copy() 
        else:
            with st.sidebar.expander(f"Select Cancer Accessions ({len(available_cancer_accs)} available)", expanded=False):
                col1_acc, col2_acc = st.columns(2)
                select_all_acc_key = f"select_all_cancer_acc"; deselect_all_acc_key = f"deselect_all_cancer_acc"
                select_all_acc_pressed = col1_acc.button("Select All", key=select_all_acc_key, use_container_width=True)
                deselect_all_acc_pressed = col2_acc.button("Deselect All", key=deselect_all_acc_key, use_container_width=True)
                cancer_accs_to_display = []
                for acc in available_cancer_accs:
                    if acc not in st.session_state.cancer_acc_selected_states: st.session_state.cancer_acc_selected_states[acc] = True
                    if select_all_acc_pressed: st.session_state.cancer_acc_selected_states[acc] = True
                    if deselect_all_acc_pressed: st.session_state.cancer_acc_selected_states[acc] = False
                    is_selected = st.checkbox(acc, value=st.session_state.cancer_acc_selected_states[acc], key=f"chk_acc_{acc}")
                    st.session_state.cancer_acc_selected_states[acc] = is_selected
                    if is_selected: cancer_accs_to_display.append(acc)
            if not cancer_accs_to_display: 
                 filtered_df_step3 = pd.DataFrame(columns=filtered_df_step2.columns); st.sidebar.warning("No Cancer Accessions selected.")
            else: filtered_df_step3 = filtered_df_step2[filtered_df_step2[CANCER_ACC_COL].isin(cancer_accs_to_display)]

        # HLA Category Selection with Individual Checkboxes
        st.sidebar.markdown("---") 
        st.sidebar.header("HLA Selection")
        all_available_hlas_in_filtered_data = sorted(filtered_df_step3[HLA_COL].unique())
        critical_set = set(CRITICAL_HLAS); nice_to_have_set = set(NICE_TO_HAVE_HLAS)
        hlas_in_data_critical = sorted([h for h in all_available_hlas_in_filtered_data if h in critical_set])
        hlas_in_data_nice = sorted([h for h in all_available_hlas_in_filtered_data if h in nice_to_have_set])
        hlas_in_data_other = sorted([h for h in all_available_hlas_in_filtered_data if h not in critical_set and h not in nice_to_have_set])

        def create_hla_expander(title, hla_list, category_key):
            if not hla_list: st.sidebar.markdown(f"_{title} (0 available)_"); return 
            with st.sidebar.expander(f"{title} ({len(hla_list)} available)", expanded=False):
                col1, col2 = st.columns(2)
                select_all_key = f"select_all_{category_key}"; deselect_all_key = f"deselect_all_{category_key}"
                select_all_pressed = col1.button("Select All", key=select_all_key, use_container_width=True)
                deselect_all_pressed = col2.button("Deselect All", key=deselect_all_key, use_container_width=True)
                for hla in hla_list:
                    if hla not in st.session_state.hla_selected_states: st.session_state.hla_selected_states[hla] = True
                    if select_all_pressed: st.session_state.hla_selected_states[hla] = True
                    if deselect_all_pressed: st.session_state.hla_selected_states[hla] = False
                    st.session_state.hla_selected_states[hla] = st.checkbox(hla, value=st.session_state.hla_selected_states[hla], key=f"chk_{hla}")

        create_hla_expander("Critical HLAs", hlas_in_data_critical, "critical")
        create_hla_expander("Nice-to-have HLAs", hlas_in_data_nice, "nice")
        create_hla_expander("Other HLAs", hlas_in_data_other, "other")

        hlas_to_display = [hla for hla, selected in st.session_state.hla_selected_states.items() if selected and hla in all_available_hlas_in_filtered_data]
        if not hlas_to_display: 
             filtered_df_final = pd.DataFrame(columns=filtered_df_step3.columns); st.sidebar.warning("No HLAs selected.")
        else: filtered_df_final = filtered_df_step3[filtered_df_step3[HLA_COL].isin(hlas_to_display)]
        
        st.sidebar.markdown("---") 
        st.sidebar.header("📊 Select Plots to Display")
        show_affinity_plots = st.sidebar.checkbox("Affinity (nM) Plots", value=True)
        show_el_rank_plots = st.sidebar.checkbox("%Rank EL Plots", value=True)
        st.sidebar.markdown("---") 
        st.sidebar.info(f"""
        Displaying data for:
        - **{filtered_df_final[MIMIC_PEPTIDE_COL].nunique()}** Mimics
        - **{filtered_df_final[CANCER_PEPTIDE_COL].nunique()}** Cancer Peptides 
        - **{filtered_df_final[CANCER_ACC_COL].nunique()}** Cancer Accessions
        - **{filtered_df_final[HLA_COL].nunique()}** HLAs
        """)

        # --- Main Panel for Visualizations ---
        st.header("2. Explore Distributions")
        if filtered_df_final.empty:
            st.warning("No data matches the current filter selections.")
        else:
            # Pass the number of selected mimics to generate_plots
            if show_affinity_plots:
                st.markdown("### Affinity (nM) Distributions")
                aff_box_fig, aff_hist_fig = generate_plots(filtered_df_final, 'Affinity (nM)', COLOR_MAP, num_mimics_selected_for_plot)
                if aff_box_fig: st.plotly_chart(aff_box_fig, use_container_width=True)
                else: st.info("Box plot for Affinity could not be generated.")
                if aff_hist_fig: st.plotly_chart(aff_hist_fig, use_container_width=True)
                else: st.info("Histogram for Affinity could not be generated.")
                st.markdown("---") 
            if show_el_rank_plots:
                st.markdown("### %Rank EL Distributions")
                el_box_fig, el_hist_fig = generate_plots(filtered_df_final, '%Rank EL', COLOR_MAP, num_mimics_selected_for_plot)
                if el_box_fig: st.plotly_chart(el_box_fig, use_container_width=True)
                else: st.info("Box plot for %Rank EL could not be generated.")
                if el_hist_fig: st.plotly_chart(el_hist_fig, use_container_width=True)
                else: st.info("Histogram for %Rank EL could not be generated.")
                st.markdown("---") 
            if not show_affinity_plots and not show_el_rank_plots:
                 st.info("Select at least one plot type from the sidebar.")

            st.header("3. Filtered Data View")
            display_cols = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL, MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL, 'source_file_name']
            display_cols_exist = [col for col in display_cols if col in filtered_df_final.columns]
            st.dataframe(filtered_df_final[display_cols_exist])
            with st.expander("Show Full Data Table"):
                 st.dataframe(filtered_df_final)
else:
    st.info("Please upload the CSV data file to begin exploration.")

# --- How to Run ---
# (Instructions remain the same)
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

