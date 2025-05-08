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

# @st.cache_data # Caching can be re-enabled if needed
def process_uploaded_data(uploaded_file_obj):
    """Loads and preprocesses data from an uploaded file object."""
    try:
        df = pd.read_csv(uploaded_file_obj)
        st.success(f"Successfully loaded uploaded data! Shape: {df.shape}")
        # Check for absolutely essential columns first
        if MIMIC_PEPTIDE_COL not in df.columns or HLA_COL not in df.columns: # HLA_COL is needed for box plots
            st.error(f"Essential columns '{MIMIC_PEPTIDE_COL}' and/or '{HLA_COL}' are missing. Cannot proceed with full functionality.")
            if MIMIC_PEPTIDE_COL not in df.columns:
                 return None 
            
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
        return pd.DataFrame(), None, None # Return None for value/source names

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
        
    return df_melted, value_name, var_name # Removed the boolean return as it's not used

# *** MODIFIED generate_plots function ***
def generate_plots(df_filtered, selected_metric, color_map, num_selected_mimics):
    """
    Generates the box plot and histogram for a given metric with specified colors.
    Uses separate logic for Affinity (nM) and %Rank EL histograms.
    Forces log_x=False for Affinity (nM) histogram to ensure rendering.
    """
    # Reshape data - ignore the boolean previously returned
    df_melted, value_col_name, source_col_name = reshape_for_plotting(df_filtered, selected_metric)
    
    # Handle case where reshaping failed
    if value_col_name is None or source_col_name is None:
        st.error(f"Failed to reshape data for metric '{selected_metric}'. Cannot generate plots.")
        return None, None

    use_log_x_boxplot = True if selected_metric == 'Affinity (nM)' else False

    if df_melted.empty:
        st.warning(f"No valid data points found for '{selected_metric}' (after reshaping and initial NaN drop) with current filters.")
        return None, None
    fig_box, fig_hist = None, None

    # --- Plot 1: Box Plot ---
    if HLA_COL in df_melted.columns and df_melted[HLA_COL].nunique() > 0 :
        num_hlas = df_melted[HLA_COL].nunique()
        plot_height = min(2500, max(400, 40 * num_hlas * 2))
        box_points_setting = 'all' if num_selected_mimics == 1 else False

        try:
            df_boxplot_data = df_melted.copy()
            # Ensure value column is numeric for box plot calculations
            df_boxplot_data[value_col_name] = pd.to_numeric(df_boxplot_data[value_col_name], errors='coerce')
            # Drop NaNs potentially introduced by coercion before calculating median/sorting
            df_boxplot_data.dropna(subset=[value_col_name], inplace=True) 

            if df_boxplot_data.empty:
                 st.warning(f"Boxplot data empty for '{selected_metric}' after ensuring numeric values.")
                 hla_order = sorted(df_melted[HLA_COL].unique()) # Fallback order
            elif use_log_x_boxplot: 
                df_for_median = df_boxplot_data[df_boxplot_data[value_col_name] > 0]
                if not df_for_median.empty:
                    median_vals = df_for_median.groupby(HLA_COL)[value_col_name].median()
                    hla_order = median_vals.replace([np.inf, -np.inf], np.nan).dropna().sort_values().index.tolist()
                else:
                    hla_order = [] 
                # Use original unique HLAs from df_boxplot_data for missing ones
                missing_hlas = [h for h in df_boxplot_data[HLA_COL].unique() if h not in hla_order]
                hla_order.extend(sorted(missing_hlas))
            else: 
                hla_order = sorted(df_boxplot_data[HLA_COL].unique())

            # Check if df_boxplot_data is still usable for plotting
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
                 fig_box = None # Set fig_box to None if data became empty


        except Exception as e: 
            st.error(f"Error generating box plot for {selected_metric}: {e}")
            fig_box = None 
    elif HLA_COL not in df_melted.columns:
        st.info(f"HLA column ('{HLA_COL}') not found in the data for '{selected_metric}'. Skipping box plot.")
        fig_box = None


    # --- Plot 2: Histogram ---
    df_hist_data = df_melted.copy()
    df_hist_data[value_col_name] = pd.to_numeric(df_hist_data[value_col_name], errors='coerce') 
    df_hist_data.dropna(subset=[value_col_name], inplace=True) # Drop NaNs again after coercion

    if df_hist_data.empty: 
         st.warning(f"Cannot generate histogram for '{selected_metric}': Data is empty after ensuring numeric values.")
    else:
        try:
            # --- Separate Histogram Logic ---
            if selected_metric == 'Affinity (nM)':
                # Force log_x=False for Affinity (nM) histogram as log_x=True was problematic
                # No need to filter non-positive values if scale is linear
                st.info("Note: Affinity (nM) histogram is displayed on a linear scale due to rendering issues with log scale.")
                fig_hist = px.histogram(
                    df_hist_data, x=value_col_name, color=source_col_name,
                    title=f"Overall Distribution of {selected_metric} (Mimic vs Cancer) [Linear Scale]",
                    labels={value_col_name: selected_metric, source_col_name: "Source"},
                    barmode='overlay', 
                    opacity=0.7,
                    log_x=False, # Force linear scale
                    marginal='rug', 
                    nbins=None, 
                    color_discrete_map=color_map
                )
                fig_hist.update_layout(xaxis_title=selected_metric + " (Linear Scale)", yaxis_title="Count")

            elif selected_metric == '%Rank EL':
                 # Use settings for %Rank EL: log_x=False, barmode='overlay', marginal='rug'
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
st.header("1. Upload Data")
uploaded_file = st.file_uploader(
    "Upload your CSV data file (ensure it includes required columns like 'mimic_Peptide', 'MHC', 'mimic_Aff(nM)', etc.):", type=["csv"])

# Initialize session state for filters if they don't exist
if 'hla_selected_states' not in st.session_state: st.session_state.hla_selected_states = {}
if 'cancer_acc_selected_states' not in st.session_state: st.session_state.cancer_acc_selected_states = {}

# --- Main Processing Logic ---
if uploaded_file is not None:
    df_original = process_uploaded_data(uploaded_file)
    if df_original is not None and not df_original.empty: 

        st.sidebar.header("⚙️ Filters")

        if MIMIC_PEPTIDE_COL not in df_original.columns:
            st.sidebar.error(f"Essential column '{MIMIC_PEPTIDE_COL}' is missing from uploaded data. Cannot proceed with filtering.")
            st.stop() 

        mimic_list = ['All'] + sorted(df_original[MIMIC_PEPTIDE_COL].unique().tolist())
        selected_mimics_list = st.sidebar.multiselect(
            "Select Mimic Peptide(s):", options=mimic_list, default=['All'] )

        num_mimics_selected_for_plot = 0
        if 'All' not in selected_mimics_list and selected_mimics_list:
            num_mimics_selected_for_plot = len(selected_mimics_list)
        elif 'All' in selected_mimics_list or not selected_mimics_list: 
            num_mimics_selected_for_plot = df_original[MIMIC_PEPTIDE_COL].nunique()

        if 'All' in selected_mimics_list or not selected_mimics_list:
            filtered_df_step1 = df_original.copy()
        else:
            filtered_df_step1 = df_original[df_original[MIMIC_PEPTIDE_COL].isin(selected_mimics_list)]

        st.sidebar.markdown("---")
        if CANCER_PEPTIDE_COL in filtered_df_step1.columns:
            cancer_peptide_list = ['All'] + sorted(filtered_df_step1[CANCER_PEPTIDE_COL].unique().tolist())
            selected_cancer_peptides = st.sidebar.multiselect(
                "Select Cancer Peptide(s):", options=cancer_peptide_list, default=['All'] )

            if 'All' not in selected_cancer_peptides and selected_cancer_peptides:
                filtered_df_step2 = filtered_df_step1[filtered_df_step1[CANCER_PEPTIDE_COL].isin(selected_cancer_peptides)]
            else:
                filtered_df_step2 = filtered_df_step1.copy()
        else:
            st.sidebar.warning(f"Column '{CANCER_PEPTIDE_COL}' not found. Skipping Cancer Peptide filter.")
            filtered_df_step2 = filtered_df_step1.copy()


        st.sidebar.markdown("---")
        st.sidebar.subheader("Cancer Accession (Gene)")
        if CANCER_ACC_COL in filtered_df_step2.columns:
            available_cancer_accs = sorted(filtered_df_step2[CANCER_ACC_COL].unique())
            if not available_cancer_accs:
                st.sidebar.info("No Cancer Accessions available for current peptide selections.")
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
        else:
            st.sidebar.warning(f"Column '{CANCER_ACC_COL}' not found. Skipping Cancer Accession filter.")
            filtered_df_step3 = filtered_df_step2.copy()

        st.sidebar.markdown("---")
        st.sidebar.subheader("HLA Selection")
        if HLA_COL in filtered_df_step3.columns:
            all_available_hlas_in_filtered_data = sorted(filtered_df_step3[HLA_COL].unique())
            hlas_in_data_critical = sorted([h for h in all_available_hlas_in_filtered_data if h in CRITICAL_HLAS]) 
            hlas_in_data_nice = sorted([h for h in all_available_hlas_in_filtered_data if h in NICE_TO_HAVE_HLAS]) 
            hlas_in_data_other = sorted([h for h in all_available_hlas_in_filtered_data if h not in CRITICAL_HLAS and h not in NICE_TO_HAVE_HLAS])


            def create_hla_expander(title, hla_list_for_category, category_key_suffix):
                if not hla_list_for_category: 
                    st.sidebar.markdown(f"_{title} (0 available in current data)_")
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
            
            create_hla_expander("Critical HLAs", hlas_in_data_critical, "critical")
            create_hla_expander("Nice-to-have HLAs", hlas_in_data_nice, "nice")
            create_hla_expander("Other HLAs", hlas_in_data_other, "other")

            hlas_to_display = [hla for hla, selected in st.session_state.hla_selected_states.items()
                               if selected and hla in all_available_hlas_in_filtered_data] 

            if not hlas_to_display and all_available_hlas_in_filtered_data: 
                filtered_df_final = pd.DataFrame(columns=filtered_df_step3.columns) 
                st.sidebar.warning("No HLAs selected. Plots may be empty.")
            elif not all_available_hlas_in_filtered_data : 
                 filtered_df_final = filtered_df_step3.copy() 
            else: 
                if hlas_to_display: 
                    filtered_df_final = filtered_df_step3[filtered_df_step3[HLA_COL].isin(hlas_to_display)]
                else: 
                     filtered_df_final = pd.DataFrame(columns=filtered_df_step3.columns)
                     st.sidebar.warning("No HLAs selected. Plots may be empty.")

        else:
            st.sidebar.warning(f"Column '{HLA_COL}' not found. Skipping HLA filtering and related plots.")
            filtered_df_final = filtered_df_step3.copy()


        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Select Plots to Display")
        show_affinity_plots = st.sidebar.checkbox("Affinity (nM) Plots", value=True)
        show_el_rank_plots = st.sidebar.checkbox("%Rank EL Plots", value=True)
        st.sidebar.markdown("---")

        mimic_peptides_count = filtered_df_final[MIMIC_PEPTIDE_COL].nunique() if MIMIC_PEPTIDE_COL in filtered_df_final.columns else 0
        cancer_peptides_count = filtered_df_final[CANCER_PEPTIDE_COL].nunique() if CANCER_PEPTIDE_COL in filtered_df_final.columns else 0
        cancer_acc_count = filtered_df_final[CANCER_ACC_COL].nunique() if CANCER_ACC_COL in filtered_df_final.columns else 0
        hla_count = filtered_df_final[HLA_COL].nunique() if HLA_COL in filtered_df_final.columns else 0
        
        st.sidebar.info(f"""
        Current Data View:
        - **{mimic_peptides_count}** Unique Mimic Peptides
        - **{cancer_peptides_count}** Unique Cancer Peptides
        - **{cancer_acc_count}** Unique Cancer Accessions
        - **{hla_count}** Unique HLAs
        - **{len(filtered_df_final)}** Total Data Rows
        """)

        st.header("2. Explore Distributions")
        if filtered_df_final.empty:
            st.warning("No data matches the current filter selections to display plots or data table.")
        else:
            if show_affinity_plots:
                st.markdown("### Affinity (nM) Distributions")
                aff_box_fig, aff_hist_fig = generate_plots(filtered_df_final, 'Affinity (nM)', COLOR_MAP, num_mimics_selected_for_plot)
                if aff_box_fig: st.plotly_chart(aff_box_fig, use_container_width=True)
                if aff_hist_fig: st.plotly_chart(aff_hist_fig, use_container_width=True)
                else: st.info(f"Histogram for Affinity (nM) could not be generated with current settings/data.") 
                st.markdown("---") 
            if show_el_rank_plots:
                st.markdown("### %Rank EL Distributions")
                el_box_fig, el_hist_fig = generate_plots(filtered_df_final, '%Rank EL', COLOR_MAP, num_mimics_selected_for_plot)
                if el_box_fig: st.plotly_chart(el_box_fig, use_container_width=True)
                if el_hist_fig: st.plotly_chart(el_hist_fig, use_container_width=True)
                else: st.info(f"Histogram for %Rank EL could not be generated with current settings/data.") 
                st.markdown("---") 
            if not show_affinity_plots and not show_el_rank_plots:
                st.info("Select at least one plot type from the sidebar to display visualizations.")

            st.header("3. Filtered Data View")
            display_cols_options = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL,
                                    MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
            if 'source_file_name' in filtered_df_final.columns: 
                display_cols_options.append('source_file_name')

            display_cols_exist = [col for col in display_cols_options if col in filtered_df_final.columns]
            
            if not filtered_df_final.empty:
                if display_cols_exist:
                    st.dataframe(filtered_df_final[display_cols_exist])
                else: 
                    st.info("Standard display columns not found. Showing all available columns from the filtered data.")
                    st.dataframe(filtered_df_final)
            else:
                st.info("Filtered data is empty for table view.")


            with st.expander("Show Full Data Table (all columns from filtered data)"):
                if not filtered_df_final.empty:
                    st.dataframe(filtered_df_final)
                else:
                    st.info("Filtered data is empty for full table view.")
    else: 
        if uploaded_file is not None: 
             st.error("Failed to process the uploaded data or the data is empty/invalid after initial processing. Please check the file format and content, ensuring essential columns like 'mimic_Peptide' and 'MHC' are present.")
else:
    st.info("👋 Welcome! Please upload your CSV data file to begin exploring peptide binding.")

st.sidebar.markdown("---")
st.sidebar.header("🚀 How to Run This App")
st.sidebar.markdown("""
1.  **Save Code:** Save this code as `streamlit_app.py` (or your preferred Python file name, e.g., `MAviz.py`).
2.  **Create `requirements.txt`:** In the same directory as your Python file, create a file named `requirements.txt` with the following content:
    ```txt
    streamlit
    pandas
    plotly
    numpy
    ```
3.  **Prepare Data:** Ensure your CSV data file is ready. It should contain columns like `mimic_Peptide`, `MHC`, `mimic_Aff(nM)`, etc. Optional columns like `cancer_Peptide` or `cancer_acc` will enable more filters.
4.  **Run Locally (Recommended for Testing):**
    * Open your terminal or command prompt.
    * Navigate to the directory where you saved the files.
    * Install requirements: `pip install -r requirements.txt`
    * Run the app: `streamlit run streamlit_app.py` (or your chosen filename).
5.  **Deploy to Streamlit Community Cloud (Optional):**
    * Push `streamlit_app.py` and `requirements.txt` to a **public** GitHub repository. **Do NOT push your sensitive CSV data file to a public repository.**
    * Go to [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub.
    * Click "New app", choose your repository, branch, and the main Python file.
    * Deploy the app.
    * When using the deployed app, you (and any users) will need to upload the CSV data file via the browser.
6.  **Share:** If deployed, share the `.streamlit.app` URL. Users will need to upload their own copy of the data CSV.
""")
st.sidebar.markdown("---")
