# streamlit_app.py (or MAviz.py)
import streamlit as st
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go # No longer needed
import os
import numpy as np
import io # Already imported, good for handling bytes
# For Excel export
# Ensure 'openpyxl' is in your requirements.txt:
# streamlit
# pandas
# plotly
# numpy
# openpyxl


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

@st.cache_data # Caching enabled for data loading and initial processing
def process_uploaded_data(uploaded_file_obj):
    """Loads and preprocesses data from an uploaded file object."""
    try:
        df = pd.read_csv(uploaded_file_obj)
        st.success(f"Successfully loaded uploaded data! Shape: {df.shape}")
        # Check for absolutely essential columns first
        if MIMIC_PEPTIDE_COL not in df.columns or HLA_COL not in df.columns: # HLA_COL is needed for box plots
            st.error(f"Essential columns '{MIMIC_PEPTIDE_COL}' and/or '{HLA_COL}' are missing. Cannot proceed with full functionality.")
            if MIMIC_PEPTIDE_COL not in df.columns:
                return None # Cannot proceed without mimic peptide
        
        all_expected_cols = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL, MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
        missing_cols = [col for col in all_expected_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Warning: Some optional columns are missing: {missing_cols}. Some features might be limited or unavailable.")

        # Convert relevant columns to numeric, coercing errors
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
    id_vars_base = []
    if MIMIC_PEPTIDE_COL in df.columns:
        id_vars_base.append(MIMIC_PEPTIDE_COL)
    else:
        st.error(f"Essential column '{MIMIC_PEPTIDE_COL}' is missing. Cannot reshape data for plotting.")
        return pd.DataFrame(), None, None 

    if HLA_COL in df.columns: 
        id_vars_base.append(HLA_COL)

    id_vars = id_vars_base[:] 
    if CANCER_PEPTIDE_COL in df.columns: id_vars.append(CANCER_PEPTIDE_COL)
    if CANCER_ACC_COL in df.columns: id_vars.append(CANCER_ACC_COL)

    if selected_metric == 'Affinity (nM)':
        value_vars_config = [MIMIC_AFF_COL, CANCER_AFF_COL]
        var_name = 'Source'
        value_name = 'Affinity (nM)'
    elif selected_metric == '%Rank EL':
        value_vars_config = [MIMIC_EL_COL, CANCER_EL_COL]
        var_name = 'Source'
        value_name = '%Rank EL'
    else: return pd.DataFrame(), None, None 

    value_vars = [v_col for v_col in value_vars_config if v_col in df.columns]
    if not value_vars or len(value_vars) < len(value_vars_config) :
        missing_val_cols = [v for v in value_vars_config if v not in df.columns]
        st.warning(f"One or more critical value columns for '{selected_metric}' are missing: {missing_val_cols}. Plotting may be incomplete or fail.")
        if not value_vars:
            return pd.DataFrame(), None, None
    
    id_vars = [col for col in id_vars if col in df.columns] 

    other_id_vars = [col for col in df.columns if col not in value_vars and col not in id_vars]
    all_id_vars = id_vars + other_id_vars
    all_id_vars = [col for col in all_id_vars if col in df.columns] 

    try:
        df_melted = pd.melt(df, id_vars=all_id_vars, value_vars=value_vars,
                            var_name=var_name, value_name=value_name)
    except Exception as e: 
        st.error(f"Error during data reshaping (pd.melt): {e}")
        st.error(f"Attempted id_vars: {all_id_vars}, value_vars: {value_vars}")
        return pd.DataFrame(), None, None

    df_melted[var_name] = df_melted[var_name].replace({
        MIMIC_AFF_COL: 'Mimic', CANCER_AFF_COL: 'Cancer',
        MIMIC_EL_COL: 'Mimic', CANCER_EL_COL: 'Cancer'
    })
    df_melted[value_name] = pd.to_numeric(df_melted[value_name], errors='coerce')
    df_melted.dropna(subset=[value_name], inplace=True)
        
    return df_melted, value_name, var_name

def generate_plots(df_filtered, selected_metric, color_map):
    """
    Generates the box plot and histogram for a given metric with specified colors.
    """
    df_melted, value_col_name, source_col_name = reshape_for_plotting(df_filtered, selected_metric)
    
    if value_col_name is None or source_col_name is None:
        return None, None

    use_log_x_boxplot = True if selected_metric == 'Affinity (nM)' else False

    if df_melted.empty:
        st.warning(f"No valid data points found for '{selected_metric}' (after reshaping and initial NaN drop) with current filters to generate plots.")
        return None, None
    fig_box, fig_hist = None, None

    actual_mimics_in_plot_data = 0
    if MIMIC_PEPTIDE_COL in df_filtered.columns: 
        actual_mimics_in_plot_data = df_filtered[MIMIC_PEPTIDE_COL].nunique()
    
    box_points_setting = 'all' if actual_mimics_in_plot_data == 1 else False

    # --- Plot 1: Box Plot ---
    if HLA_COL in df_melted.columns and df_melted[HLA_COL].nunique() > 0 :
        num_hlas = df_melted[HLA_COL].nunique()
        plot_height = min(2500, max(400, 40 * num_hlas * 2)) 

        try:
            df_boxplot_data = df_melted.copy()
            df_boxplot_data[value_col_name] = pd.to_numeric(df_boxplot_data[value_col_name], errors='coerce')
            df_boxplot_data.dropna(subset=[value_col_name], inplace=True) 

            if df_boxplot_data.empty:
                st.warning(f"Boxplot data empty for '{selected_metric}' after ensuring numeric values.")
                hla_order = sorted(df_melted[HLA_COL].unique()) 
            elif use_log_x_boxplot: 
                df_for_median = df_boxplot_data[df_boxplot_data[value_col_name] > 0]
                if not df_for_median.empty:
                    median_vals = df_for_median.groupby(HLA_COL)[value_col_name].median()
                    hla_order = median_vals.replace([np.inf, -np.inf], np.nan).dropna().sort_values().index.tolist()
                else:
                    hla_order = [] 
                missing_hlas = [h for h in df_boxplot_data[HLA_COL].unique() if h not in hla_order]
                hla_order.extend(sorted(missing_hlas))
            else: 
                hla_order = sorted(df_boxplot_data[HLA_COL].unique())

            if not df_boxplot_data.empty:
                fig_box = px.box(
                    df_boxplot_data, x=value_col_name, y=HLA_COL, color=source_col_name,
                    title=f"{selected_metric} Distribution per HLA (Mimic vs Cancer)",
                    labels={value_col_name: selected_metric, HLA_COL: "HLA Type", source_col_name: "Source"},
                    orientation='h', log_x=use_log_x_boxplot, 
                    category_orders={HLA_COL: hla_order},
                    height=plot_height, points=box_points_setting, color_discrete_map=color_map
                )

                if hla_order and len(hla_order) > 0:
                    fig_box.add_shape(
                        type="line", xref="paper", x0=0, x1=1, y0=-0.5, y1=-0.5,
                        layer="below", line=dict(color="DarkGray", width=1.5, dash="solid")
                    )
                    for i in range(len(hla_order)):
                        fig_box.add_shape(
                            type="line", xref="paper", x0=0, x1=1, y0=i + 0.5, y1=i + 0.5,
                            layer="below", line=dict(color="DarkGray", width=1.5, dash="solid")
                        )
                fig_box.update_yaxes(
                    categoryorder='array', categoryarray=hla_order,
                    tickfont=dict(size=18), showgrid=False,
                )
                fig_box.update_xaxes(title_text=selected_metric)
                fig_box.update_layout(margin=dict(l=150, t=50, b=50, r=30))
            else:
                fig_box = None 

        except Exception as e: 
            st.error(f"Error generating box plot for {selected_metric}: {e}")
            fig_box = None 
    elif HLA_COL not in df_melted.columns:
        st.info(f"HLA column ('{HLA_COL}') not found in the data for '{selected_metric}'. Skipping box plot.")
        fig_box = None

    # --- Plot 2: Histogram ---
    df_hist_data = df_melted.copy()
    df_hist_data[value_col_name] = pd.to_numeric(df_hist_data[value_col_name], errors='coerce') 
    df_hist_data.dropna(subset=[value_col_name], inplace=True)

    if df_hist_data.empty: 
        st.warning(f"Cannot generate histogram for '{selected_metric}': Data is empty after ensuring numeric values.")
    else:
        try:
            if selected_metric == 'Affinity (nM)':
                st.info("Note: Affinity (nM) histogram is displayed on a linear scale due to rendering issues with log scale.")
                fig_hist = px.histogram(
                    df_hist_data, x=value_col_name, color=source_col_name,
                    title=f"Overall Distribution of {selected_metric} (Mimic vs Cancer) [Linear Scale]",
                    labels={value_col_name: selected_metric, source_col_name: "Source"},
                    barmode='overlay', 
                    opacity=0.7,
                    log_x=False, 
                    marginal='rug', 
                    nbins=None, 
                    color_discrete_map=color_map
                )
                fig_hist.update_layout(xaxis_title=selected_metric + " (Linear Scale)", yaxis_title="Count")

            elif selected_metric == '%Rank EL':
                fig_hist = px.histogram(
                    df_hist_data, x=value_col_name, color=source_col_name,
                    title=f"Overall Distribution of {selected_metric} (Mimic vs Cancer)",
                    labels={value_col_name: selected_metric, source_col_name: "Source"},
                    barmode='overlay', 
                    opacity=0.7,
                    log_x=False, 
                    marginal='rug', 
                    nbins=None, 
                    color_discrete_map=color_map
                )
                fig_hist.update_layout(xaxis_title=selected_metric, yaxis_title="Count")
            
            else: 
                st.warning(f"Unknown metric '{selected_metric}' for histogram generation.")
                fig_hist = None

        except Exception as e: 
            st.error(f"Error generating histogram for {selected_metric}: {e}")
            fig_hist = None 

    return fig_box, fig_hist

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🧬 Mimic Peptide Binding Explorer")
st.markdown("Explore Affinity (nM) and %Rank EL distributions for selected mimics across different HLA types.")

# --- File Uploader ---
st.header("Upload Data") 
uploaded_file = st.file_uploader(
    "Upload your CSV data file (ensure it includes required columns like 'mimic_Peptide', 'MHC', 'mimic_Aff(nM)', etc.):", type=["csv"])

# Initialize session state for filters if they don't exist
if 'hla_selected_states' not in st.session_state: st.session_state.hla_selected_states = {}
if 'cancer_acc_selected_states' not in st.session_state: st.session_state.cancer_acc_selected_states = {}

# --- Main Processing Logic ---
if uploaded_file is not None:
    df_original = process_uploaded_data(uploaded_file) # This call is now cached
    if df_original is not None and not df_original.empty: 

        # --- Display Options ---
        st.markdown("---")
        st.subheader("📊 Display Options")
        col1_display, col2_display, col3_display = st.columns(3)
        with col1_display:
            show_affinity_plots = st.checkbox("Affinity (nM) Plots", value=True, key="cb_aff_plots")
        with col2_display:
            show_el_rank_plots = st.checkbox("%Rank EL Plots", value=True, key="cb_el_plots")
        with col3_display:
            show_filtered_data_view = st.checkbox("Filtered Data View", value=False, key="cb_data_view") 
        st.markdown("---")


        # --- Sidebar Filters ---
        st.sidebar.header("⚙️ Filters")

        if MIMIC_PEPTIDE_COL not in df_original.columns: 
            st.sidebar.error(f"Essential column '{MIMIC_PEPTIDE_COL}' is missing. Cannot proceed.")
            st.stop()

        # --- Mimic Must Bind To Filter ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Mimic Must Bind To HLA(s)")
        df_after_must_bind_filter = df_original.copy() 

        if HLA_COL in df_original.columns:
            all_hlas_for_must_bind_filter = sorted(df_original[HLA_COL].unique().tolist())
            if not all_hlas_for_must_bind_filter:
                st.sidebar.info("No HLAs available in the data to apply the 'Must Bind To' filter.")
            else:
                selected_must_bind_hlas = st.sidebar.multiselect(
                    "Only include mimics that bind to at least one of these selected HLAs:",
                    options=all_hlas_for_must_bind_filter,
                    default=[], 
                    key="must_bind_hlas_multiselect"
                )

                if selected_must_bind_hlas:
                    mimics_meeting_criteria_df = df_original[df_original[HLA_COL].isin(selected_must_bind_hlas)]
                    mimics_to_keep = mimics_meeting_criteria_df[MIMIC_PEPTIDE_COL].unique()
                    
                    if mimics_to_keep.size > 0:
                        df_after_must_bind_filter = df_original[df_original[MIMIC_PEPTIDE_COL].isin(mimics_to_keep)].copy()
                        st.sidebar.caption(f"{len(mimics_to_keep)} mimic(s) meet the 'must bind' criteria.")
                    else:
                        st.sidebar.warning("No mimics found that bind to the selected 'must bind' HLA(s). Subsequent data will be empty.")
                        df_after_must_bind_filter = pd.DataFrame(columns=df_original.columns) 
        else:
            st.sidebar.warning(f"Column '{HLA_COL}' not found. 'Mimic Must Bind To' filter cannot be applied.")

        # --- Binding Threshold Filters ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Binding Thresholds")
        
        default_max_affinity_input_val = 500.0  # Default for the number_input widget
        default_max_el_rank_input_val = 2.0     # Default for the number_input widget

        df_for_threshold_application = df_after_must_bind_filter.copy()
        
        # --- Affinity Threshold ---
        enable_affinity_filter = st.sidebar.checkbox("Filter by Max Affinity (nM)", value=False, key="enable_aff_filter") # Off by default
        affinity_cols_present = any(col in df_for_threshold_application.columns for col in [MIMIC_AFF_COL, CANCER_AFF_COL])
        min_aff_data_display, max_aff_data_display = None, None

        if affinity_cols_present and not df_for_threshold_application.empty:
            aff_values_list = []
            if MIMIC_AFF_COL in df_for_threshold_application.columns:
                aff_values_list.append(df_for_threshold_application[MIMIC_AFF_COL].dropna())
            if CANCER_AFF_COL in df_for_threshold_application.columns:
                aff_values_list.append(df_for_threshold_application[CANCER_AFF_COL].dropna())
            
            if aff_values_list:
                all_aff_values_for_display = pd.concat(aff_values_list)
                if not all_aff_values_for_display.empty:
                    min_aff_data_display = all_aff_values_for_display.min()
                    max_aff_data_display = all_aff_values_for_display.max()
                    st.sidebar.caption(f"Affinity Data Range: {min_aff_data_display:.2f} to {max_aff_data_display:.2f} nM")
                else:
                    st.sidebar.caption("No numeric Affinity (nM) data to display range.")
            else: 
                st.sidebar.caption("Affinity columns found but no numeric data.")
        elif not df_for_threshold_application.empty: 
             st.sidebar.caption(f"'{MIMIC_AFF_COL}' or '{CANCER_AFF_COL}' not in current data.")
        
        selected_max_affinity = default_max_affinity_input_val 
        if enable_affinity_filter:
            if affinity_cols_present and not df_for_threshold_application.empty:
                slider_max_aff = max_aff_data_display if max_aff_data_display is not None else 10000.0
                if slider_max_aff <= 0: slider_max_aff = 10000.0 

                selected_max_affinity = st.sidebar.number_input(
                    f"Max Affinity (nM) (e.g., <= {default_max_affinity_input_val}):",
                    min_value=0.0,
                    max_value=float(slider_max_aff), 
                    value=default_max_affinity_input_val,
                    step=10.0,
                    key="max_affinity_filter_input",
                    help="Filters data where Mimic AND Cancer (if present) Aff(nM) are <= this value. Lower is better."
                )
            elif df_for_threshold_application.empty:
                st.sidebar.info("No data to apply affinity filter to (enable checkbox ignored).")
            else: 
                st.sidebar.info(f"Affinity columns not found, cannot filter by affinity (enable checkbox ignored).")

        # --- EL Rank Threshold ---
        enable_el_rank_filter = st.sidebar.checkbox("Filter by Max %Rank EL", value=False, key="enable_el_filter") # Off by default
        el_rank_cols_present = any(col in df_for_threshold_application.columns for col in [MIMIC_EL_COL, CANCER_EL_COL])
        min_el_data_display, max_el_data_display = None, None

        if el_rank_cols_present and not df_for_threshold_application.empty:
            el_values_list = []
            if MIMIC_EL_COL in df_for_threshold_application.columns:
                el_values_list.append(df_for_threshold_application[MIMIC_EL_COL].dropna())
            if CANCER_EL_COL in df_for_threshold_application.columns:
                el_values_list.append(df_for_threshold_application[CANCER_EL_COL].dropna())

            if el_values_list:
                all_el_values_for_display = pd.concat(el_values_list)
                if not all_el_values_for_display.empty:
                    min_el_data_display = all_el_values_for_display.min()
                    max_el_data_display = all_el_values_for_display.max()
                    st.sidebar.caption(f"%Rank EL Data Range: {min_el_data_display:.2f} to {max_el_data_display:.2f}")
                else:
                    st.sidebar.caption("No numeric %Rank EL data to display range.")
            else:
                st.sidebar.caption("%Rank EL columns found but no numeric data.")
        elif not df_for_threshold_application.empty:
            st.sidebar.caption(f"'{MIMIC_EL_COL}' or '{CANCER_EL_COL}' not in current data.")

        selected_max_el_rank = default_max_el_rank_input_val 
        if enable_el_rank_filter:
            if el_rank_cols_present and not df_for_threshold_application.empty:
                slider_max_el = max_el_data_display if max_el_data_display is not None else 100.0
                if slider_max_el <= 0: slider_max_el = 100.0

                selected_max_el_rank = st.sidebar.number_input(
                    f"Max %Rank EL (e.g., <= {default_max_el_rank_input_val}):",
                    min_value=0.0,
                    max_value=float(slider_max_el), 
                    value=default_max_el_rank_input_val,
                    step=0.1,
                    key="max_el_rank_filter_input",
                    help="Filters data where Mimic AND Cancer (if present) %Rank EL are <= this value. Lower is better."
                )
            elif df_for_threshold_application.empty:
                st.sidebar.info("No data to apply %Rank EL filter to (enable checkbox ignored).")
            else: 
                st.sidebar.info(f"%Rank EL columns not found, cannot filter by %Rank EL (enable checkbox ignored).")

        # Apply threshold filters conditionally
        df_after_thresholds = df_for_threshold_application.copy()

        if enable_affinity_filter and affinity_cols_present and not df_for_threshold_application.empty:
            mimic_aff_cond = pd.Series(True, index=df_after_thresholds.index)
            if MIMIC_AFF_COL in df_after_thresholds.columns:
                mimic_aff_cond = (pd.to_numeric(df_after_thresholds[MIMIC_AFF_COL], errors='coerce') <= selected_max_affinity)
            
            cancer_aff_cond = pd.Series(True, index=df_after_thresholds.index)
            if CANCER_AFF_COL in df_after_thresholds.columns:
                cancer_aff_cond = (pd.to_numeric(df_after_thresholds[CANCER_AFF_COL], errors='coerce') <= selected_max_affinity)
            df_after_thresholds = df_after_thresholds[mimic_aff_cond & cancer_aff_cond]

        if enable_el_rank_filter and el_rank_cols_present and not df_for_threshold_application.empty:
            mimic_el_cond = pd.Series(True, index=df_after_thresholds.index)
            if MIMIC_EL_COL in df_after_thresholds.columns:
                mimic_el_cond = (pd.to_numeric(df_after_thresholds[MIMIC_EL_COL], errors='coerce') <= selected_max_el_rank)

            cancer_el_cond = pd.Series(True, index=df_after_thresholds.index)
            if CANCER_EL_COL in df_after_thresholds.columns:
                cancer_el_cond = (pd.to_numeric(df_after_thresholds[CANCER_EL_COL], errors='coerce') <= selected_max_el_rank)
            df_after_thresholds = df_after_thresholds[mimic_el_cond & cancer_el_cond]
        
        # --- Select Mimic Peptide(s) Filter (operates on df_after_thresholds) ---
        st.sidebar.markdown("---") 
        st.sidebar.subheader("Select Mimic Peptide(s)")
        
        mimic_list_options = []
        if MIMIC_PEPTIDE_COL in df_after_thresholds.columns and not df_after_thresholds.empty:
             mimic_list_options = sorted(df_after_thresholds[MIMIC_PEPTIDE_COL].unique().tolist())

        if not mimic_list_options:
            st.sidebar.info("No mimic peptides available after applying preceding filters.")
            mimic_list_for_multiselect = ['All'] 
        else:
            mimic_list_for_multiselect = ['All'] + mimic_list_options
        
        selected_mimics_list = st.sidebar.multiselect(
            "Filter by specific Mimic Peptide(s):", 
            options=mimic_list_for_multiselect, 
            default=['All'],
            key="select_mimics_multiselect"
        )

        if 'All' in selected_mimics_list or not selected_mimics_list or not mimic_list_options:
            filtered_df_step1 = df_after_thresholds.copy()
        else:
            filtered_df_step1 = df_after_thresholds[df_after_thresholds[MIMIC_PEPTIDE_COL].isin(selected_mimics_list)]

        # --- Cancer Antigen Substring Filter ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧬 Cancer Antigen Sequence Filter")
        df_for_antigen_filter = filtered_df_step1.copy()
        pasted_antigen_sequence = st.sidebar.text_area(
            "Paste Cancer Antigen Sequence (optional):", 
            key="pasted_antigen_seq",
            height=100, 
            help="Filters for cancer peptides that are substrings of, or contain, this sequence."
        ).strip()

        df_after_antigen_substring_filter = df_for_antigen_filter.copy() 

        if pasted_antigen_sequence and CANCER_PEPTIDE_COL in df_for_antigen_filter.columns and not df_for_antigen_filter.empty:
            # Ensure cancer peptide column is string and handle NaNs for safe processing
            df_for_antigen_filter_safe = df_for_antigen_filter.copy() # Work on a copy for this operation
            df_for_antigen_filter_safe[CANCER_PEPTIDE_COL] = df_for_antigen_filter_safe[CANCER_PEPTIDE_COL].astype(str).fillna('')
            
            # Filter condition: pasted sequence is in cancer_peptide OR cancer_peptide is in pasted_sequence
            condition = (
                df_for_antigen_filter_safe[CANCER_PEPTIDE_COL].apply(lambda cp: pasted_antigen_sequence in cp if cp else False) |
                df_for_antigen_filter_safe[CANCER_PEPTIDE_COL].apply(lambda cp: cp in pasted_antigen_sequence if cp and pasted_antigen_sequence else False) # ensure cp is not empty for 'in'
            )
            df_after_antigen_substring_filter = df_for_antigen_filter[condition] # Apply condition to original df_for_antigen_filter
            
            if df_after_antigen_substring_filter.empty:
                st.sidebar.warning("No cancer peptides match the pasted antigen sequence criteria.")
            else:
                st.sidebar.caption(f"{len(df_after_antigen_substring_filter)} rows match the antigen sequence filter.")
        elif pasted_antigen_sequence and CANCER_PEPTIDE_COL not in df_for_antigen_filter.columns:
            st.sidebar.warning(f"'{CANCER_PEPTIDE_COL}' not found, cannot apply antigen sequence filter.")
        elif pasted_antigen_sequence and df_for_antigen_filter.empty:
            st.sidebar.info("No data available to apply antigen sequence filter (due to previous filters).")
        # If pasted_antigen_sequence is empty, df_after_antigen_substring_filter remains a copy of df_for_antigen_filter


        # --- Cancer Peptide Filter (now operates on df_after_antigen_substring_filter) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Select Cancer Peptide(s)")
        if CANCER_PEPTIDE_COL in df_after_antigen_substring_filter.columns and not df_after_antigen_substring_filter.empty:
            cancer_peptide_list_options = sorted(df_after_antigen_substring_filter[CANCER_PEPTIDE_COL].unique().tolist())
            if not cancer_peptide_list_options:
                st.sidebar.info("No Cancer Peptides available for current selections.")
                filtered_df_step2 = df_after_antigen_substring_filter.copy()
            else:
                cancer_peptide_list_for_multiselect = ['All'] + cancer_peptide_list_options
                selected_cancer_peptides = st.sidebar.multiselect(
                    "Filter by specific Cancer Peptide(s):", 
                    options=cancer_peptide_list_for_multiselect, 
                    default=['All'],
                    key="select_cancer_peptides_multiselect"
                )
                if 'All' not in selected_cancer_peptides and selected_cancer_peptides:
                    filtered_df_step2 = df_after_antigen_substring_filter[df_after_antigen_substring_filter[CANCER_PEPTIDE_COL].isin(selected_cancer_peptides)]
                else:
                    filtered_df_step2 = df_after_antigen_substring_filter.copy()
        elif df_after_antigen_substring_filter.empty:
            st.sidebar.info("Previous filters resulted in no data for Cancer Peptide selection.")
            filtered_df_step2 = df_after_antigen_substring_filter.copy() 
        else:
            st.sidebar.warning(f"Column '{CANCER_PEPTIDE_COL}' not found. Skipping Cancer Peptide filter.")
            filtered_df_step2 = df_after_antigen_substring_filter.copy()


        # --- Cancer Accession Filter ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Cancer Accession (Gene)")
        if CANCER_ACC_COL in filtered_df_step2.columns and not filtered_df_step2.empty:
            available_cancer_accs = sorted(filtered_df_step2[CANCER_ACC_COL].unique())
            if not available_cancer_accs:
                st.sidebar.info("No Cancer Accessions available for current selections.")
                filtered_df_step3 = filtered_df_step2.copy()
            else:
                with st.sidebar.expander(f"Select Cancer Accessions ({len(available_cancer_accs)} available)", expanded=False):
                    col1_acc, col2_acc = st.columns(2)
                    if col1_acc.button("Select All", key="btn_select_all_cancer_acc", use_container_width=True):
                        for acc in available_cancer_accs: st.session_state.cancer_acc_selected_states[acc] = True
                    if col2_acc.button("Deselect All", key="btn_deselect_all_cancer_acc", use_container_width=True):
                        for acc in available_cancer_accs: st.session_state.cancer_acc_selected_states[acc] = False

                    cancer_accs_to_display = []
                    for acc in available_cancer_accs:
                        if acc not in st.session_state.cancer_acc_selected_states:
                            st.session_state.cancer_acc_selected_states[acc] = True 
                        is_selected = st.checkbox(acc, value=st.session_state.cancer_acc_selected_states[acc], key=f"chk_acc_{acc}")
                        st.session_state.cancer_acc_selected_states[acc] = is_selected
                        if is_selected: cancer_accs_to_display.append(acc)

                if not cancer_accs_to_display and available_cancer_accs: 
                    filtered_df_step3 = pd.DataFrame(columns=filtered_df_step2.columns) 
                    st.sidebar.warning("No Cancer Accessions selected. Plots may be empty.")
                elif not available_cancer_accs: 
                    filtered_df_step3 = filtered_df_step2.copy() 
                else: 
                    filtered_df_step3 = filtered_df_step2[filtered_df_step2[CANCER_ACC_COL].isin(cancer_accs_to_display)]
        elif filtered_df_step2.empty:
            st.sidebar.info("Previous filters resulted in no data for Cancer Accession selection.")
            filtered_df_step3 = filtered_df_step2.copy()
        else:
            st.sidebar.warning(f"Column '{CANCER_ACC_COL}' not found. Skipping Cancer Accession filter.")
            filtered_df_step3 = filtered_df_step2.copy()

        # --- HLA Selection for Visualization ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("HLA Selection (for Visualization)")
        if HLA_COL in filtered_df_step3.columns and not filtered_df_step3.empty:
            all_available_hlas_in_filtered_data = sorted(filtered_df_step3[HLA_COL].unique())
            
            current_hla_set = set(all_available_hlas_in_filtered_data)
            for hla_key in list(st.session_state.hla_selected_states.keys()):
                if hla_key not in current_hla_set:
                    del st.session_state.hla_selected_states[hla_key]

            hlas_in_data_critical = sorted([h for h in all_available_hlas_in_filtered_data if h in CRITICAL_HLAS]) 
            hlas_in_data_nice = sorted([h for h in all_available_hlas_in_filtered_data if h in NICE_TO_HAVE_HLAS]) 
            hlas_in_data_other = sorted([h for h in all_available_hlas_in_filtered_data if h not in CRITICAL_HLAS and h not in NICE_TO_HAVE_HLAS])


            def create_hla_expander(title, hla_list_for_category, category_key_suffix):
                if not hla_list_for_category: 
                    st.sidebar.markdown(f"_{title} (0 available in current data for visualization)_")
                    return
                
                is_critical_category = (title == "Critical HLAs")
                has_critical_hlas = bool(hlas_in_data_critical) 

                is_only_this_category_with_hlas = \
                    (is_critical_category and has_critical_hlas and not bool(hlas_in_data_nice) and not bool(hlas_in_data_other)) or \
                    (title == "Nice-to-have HLAs" and bool(hlas_in_data_nice) and not has_critical_hlas and not bool(hlas_in_data_other)) or \
                    (title == "Other HLAs" and bool(hlas_in_data_other) and not has_critical_hlas and not bool(hlas_in_data_nice))

                default_expanded = (is_critical_category and has_critical_hlas) or is_only_this_category_with_hlas


                with st.sidebar.expander(f"{title} ({len(hla_list_for_category)} available)", expanded=default_expanded):
                    col1, col2 = st.columns(2)
                    if col1.button("Select All", key=f"btn_select_all_hla_{category_key_suffix}", use_container_width=True):
                        for hla in hla_list_for_category: st.session_state.hla_selected_states[hla] = True
                    if col2.button("Deselect All", key=f"btn_deselect_all_hla_{category_key_suffix}", use_container_width=True):
                        for hla in hla_list_for_category: st.session_state.hla_selected_states[hla] = False

                    for hla in hla_list_for_category:
                        if hla not in st.session_state.hla_selected_states:
                            st.session_state.hla_selected_states[hla] = True 
                        is_selected_hla = st.checkbox(hla, value=st.session_state.hla_selected_states[hla], key=f"chk_hla_{hla}")
                        st.session_state.hla_selected_states[hla] = is_selected_hla
            
            if not all_available_hlas_in_filtered_data:
                 st.sidebar.info("No HLAs available in the filtered data for visualization selection.")
                 filtered_df_final = filtered_df_step3.copy() 
            else:
                create_hla_expander("Critical HLAs", hlas_in_data_critical, "critical")
                create_hla_expander("Nice-to-have HLAs", hlas_in_data_nice, "nice")
                create_hla_expander("Other HLAs", hlas_in_data_other, "other")

                hlas_to_display_for_viz = [hla for hla, selected in st.session_state.hla_selected_states.items()
                                    if selected and hla in all_available_hlas_in_filtered_data] 

                if not hlas_to_display_for_viz and all_available_hlas_in_filtered_data: 
                    filtered_df_final = pd.DataFrame(columns=filtered_df_step3.columns) 
                    st.sidebar.warning("No HLAs selected for visualization. Plots may be empty.")
                elif not all_available_hlas_in_filtered_data : 
                    filtered_df_final = filtered_df_step3.copy() 
                else: 
                    if hlas_to_display_for_viz: 
                        filtered_df_final = filtered_df_step3[filtered_df_step3[HLA_COL].isin(hlas_to_display_for_viz)]
                    else: 
                        filtered_df_final = pd.DataFrame(columns=filtered_df_step3.columns) 
                        st.sidebar.warning("No HLAs selected for visualization. Plots may be empty.")
        elif filtered_df_step3.empty:
            st.sidebar.info("Previous filters resulted in no data for HLA visualization selection.")
            filtered_df_final = filtered_df_step3.copy()
        else: 
            st.sidebar.warning(f"Column '{HLA_COL}' not found in currently filtered data. Skipping HLA visualization filter.")
            filtered_df_final = filtered_df_step3.copy()

        # --- Sidebar Data Summary ---
        st.sidebar.markdown("---") 
        mimic_peptides_count = filtered_df_final[MIMIC_PEPTIDE_COL].nunique() if MIMIC_PEPTIDE_COL in filtered_df_final.columns and not filtered_df_final.empty else 0
        cancer_peptides_count = filtered_df_final[CANCER_PEPTIDE_COL].nunique() if CANCER_PEPTIDE_COL in filtered_df_final.columns and not filtered_df_final.empty else 0
        cancer_acc_count = filtered_df_final[CANCER_ACC_COL].nunique() if CANCER_ACC_COL in filtered_df_final.columns and not filtered_df_final.empty else 0
        hla_count = filtered_df_final[HLA_COL].nunique() if HLA_COL in filtered_df_final.columns and not filtered_df_final.empty else 0
        
        st.sidebar.info(f"""
        **Final Data for Display/Export:**
        - **{mimic_peptides_count}** Unique Mimic Peptides
        - **{cancer_peptides_count}** Unique Cancer Peptides
        - **{cancer_acc_count}** Unique Cancer Accessions
        - **{hla_count}** Unique HLAs
        - **{len(filtered_df_final)}** Total Data Rows
        """)

        # --- Main Page Content: Plots, Data, and Export ---
        if filtered_df_final.empty:
            st.warning("No data matches the current filter selections to display plots, data table, or to export.")
        else:
            # --- Explore Distributions ---
            st.header("Explore Distributions") 
            plots_displayed = False
            if show_affinity_plots:
                st.markdown("### Affinity (nM) Distributions")
                aff_box_fig, aff_hist_fig = generate_plots(filtered_df_final, 'Affinity (nM)', COLOR_MAP)
                if aff_box_fig: st.plotly_chart(aff_box_fig, use_container_width=True)
                if aff_hist_fig: st.plotly_chart(aff_hist_fig, use_container_width=True)
                st.markdown("---") 
                plots_displayed = True
            if show_el_rank_plots:
                st.markdown("### %Rank EL Distributions")
                el_box_fig, el_hist_fig = generate_plots(filtered_df_final, '%Rank EL', COLOR_MAP)
                if el_box_fig: st.plotly_chart(el_box_fig, use_container_width=True)
                if el_hist_fig: st.plotly_chart(el_hist_fig, use_container_width=True)
                st.markdown("---") 
                plots_displayed = True
            
            if not plots_displayed and (show_affinity_plots or show_el_rank_plots): 
                 st.info("Selected plots could not be generated with the current data filters (e.g., missing required columns or all data filtered out).")
            elif not show_affinity_plots and not show_el_rank_plots: 
                st.info("Select at least one plot type from the 'Display Options' above to show visualizations.")


            # --- Filtered Data View ---
            if show_filtered_data_view:
                st.header("Filtered Data View") 
                display_cols_options = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL,
                                        MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
                if 'source_file_name' in filtered_df_final.columns: 
                    display_cols_options.append('source_file_name')

                display_cols_exist = [col for col in display_cols_options if col in filtered_df_final.columns]
                
                if display_cols_exist: 
                    st.dataframe(filtered_df_final[display_cols_exist])
                else: 
                    st.info("Standard display columns not found. Showing all available columns from the filtered data.")
                    st.dataframe(filtered_df_final)

                with st.expander("Show Full Data Table (all columns from filtered data)"):
                    st.dataframe(filtered_df_final)
            elif not filtered_df_final.empty : 
                 st.info("The 'Filtered Data View' is currently hidden. You can enable it in the 'Display Options' above.")
            
            # --- Export Data Section ---
            st.markdown("---")
            st.header("Export Filtered Data") 
            
            export_filename_base = st.text_input("Enter filename (without extension):", "filtered_peptide_data", key="export_filename")
            export_format = st.radio("Select export format:", ("CSV (.csv)", "Excel (.xlsx)"), key="export_format_radio")

            file_extension = ".csv" if export_format == "CSV (.csv)" else ".xlsx"
            full_export_filename = f"{export_filename_base}{file_extension}"

            if export_format == "CSV (.csv)":
                try:
                    csv_data = filtered_df_final.to_csv(index=False).encode('utf-8')
                    mime_type = 'text/csv'
                    data_to_download = csv_data
                except Exception as e:
                    st.error(f"Error preparing CSV for download: {e}")
                    data_to_download = None
            else: # Excel
                try:
                    output_buffer = io.BytesIO()
                    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                        filtered_df_final.to_excel(writer, index=False, sheet_name='Filtered Data')
                    excel_data = output_buffer.getvalue()
                    mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    data_to_download = excel_data
                except Exception as e:
                    st.error(f"Error preparing Excel for download: {e}")
                    data_to_download = None
            
            if data_to_download:
                st.download_button(
                    label=f"Download {full_export_filename}",
                    data=data_to_download,
                    file_name=full_export_filename,
                    mime=mime_type,
                    key="download_button"
                )
            else:
                st.warning("Could not prepare data for download. Check for error messages above.")


    else: 
        if uploaded_file is not None: 
            st.error("Failed to process the uploaded data or the data is empty/invalid after initial processing. Please check the file format and content, ensuring essential columns like 'mimic_Peptide' and 'MHC' are present.")
else:
    st.info("👋 Welcome! Please upload your CSV data file to begin exploring peptide binding.")

st.sidebar.markdown("---")
