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
        required_cols = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL, MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Error: Missing required columns: {missing_cols}")
            return None
        # Convert relevant columns to numeric, coercing errors
        for col in [MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Strip whitespace from HLA column
        if HLA_COL in df.columns and df[HLA_COL].dtype == 'object':
            df[HLA_COL] = df[HLA_COL].str.strip()
        # Strip whitespace from peptide and accession columns
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
        # log_y = True # This was for box plot's x-axis, handled in generate_plots now
    elif selected_metric == '%Rank EL':
        value_vars = [MIMIC_EL_COL, CANCER_EL_COL]
        var_name = 'Source'
        value_name = '%Rank EL'
        # log_y = False # This was for box plot's x-axis, handled in generate_plots now
    else: return pd.DataFrame(), False, None, None # Return default False for log_x_hist if metric unknown

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
    
    # Determine if log scale should be used for the X-AXIS of the HISTOGRAM
    use_log_x_for_histogram = True if selected_metric == 'Affinity (nM)' else False
    
    return df_melted, use_log_x_for_histogram, value_name, var_name

# *** MODIFIED generate_plots function ***
def generate_plots(df_filtered, selected_metric, color_map, num_selected_mimics):
    """
    Generates the box plot and histogram for a given metric with specified colors.
    Adjusts box plot point visibility based on the number of selected mimics.
    Adds horizontal lines to delineate HLA groups in the box plot.
    Includes debugging for histogram data.
    """
    df_melted, use_log_x_hist_setting, value_col_name, source_col_name = reshape_for_plotting(df_filtered, selected_metric)
    
    use_log_x_boxplot = True if selected_metric == 'Affinity (nM)' else False

    if df_melted.empty:
        st.warning(f"No valid data points found for '{selected_metric}' (after reshaping) with current filters.")
        return None, None
    fig_box, fig_hist = None, None

    # --- Plot 1: Box Plot ---
    if df_melted[HLA_COL].nunique() > 0:
        num_hlas = df_melted[HLA_COL].nunique()
        plot_height = min(2500, max(400, 40 * num_hlas * 2))
        box_points_setting = 'all' if num_selected_mimics == 1 else False

        try:
            if use_log_x_boxplot: 
                df_melted[value_col_name] = pd.to_numeric(df_melted[value_col_name], errors='coerce')
                df_for_median = df_melted[df_melted[value_col_name] > 0] if use_log_x_boxplot else df_melted
                if not df_for_median.empty:
                    median_vals = df_for_median.groupby(HLA_COL)[value_col_name].median()
                    hla_order = median_vals.replace([np.inf, -np.inf], np.nan).dropna().sort_values().index.tolist()
                else:
                    hla_order = [] 
                missing_hlas = [h for h in df_melted[HLA_COL].unique() if h not in hla_order]
                hla_order.extend(sorted(missing_hlas))
            else: 
                hla_order = sorted(df_melted[HLA_COL].unique())

            fig_box = px.box(
                df_melted, x=value_col_name, y=HLA_COL, color=source_col_name,
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
                tickfont=dict(size=10), showgrid=False,
            )
            fig_box.update_xaxes(title_text=selected_metric)
            fig_box.update_layout(margin=dict(l=150, t=50, b=50, r=30))
        except Exception as e: st.error(f"Error generating box plot for {selected_metric}: {e}")

    # --- Plot 2: Histogram ---
    # Debugging info can be commented out or removed once the issue is resolved
    # st.markdown(f"--- \n ### Debug Info for '{selected_metric}' Histogram:")
    # st.write(f"Shape of `df_melted` before histogram processing: `{df_melted.shape}`")
    # ... (rest of debug info)

    df_hist_data = df_melted.copy()
    
    # Determine log_x for histogram, with override for Affinity (nM) for testing
    current_log_x_hist = use_log_x_hist_setting
    if selected_metric == 'Affinity (nM)':
        st.info("Testing Affinity (nM) histogram with log_x = False") # Inform user of the test
        current_log_x_hist = False # Override for testing

    if current_log_x_hist: # Only filter non-positive if log scale is actually being used
        df_hist_data[value_col_name] = pd.to_numeric(df_hist_data[value_col_name], errors='coerce')
        non_positive_count_before = (df_hist_data[value_col_name] <= 0).sum()
        if non_positive_count_before > 0:
            st.warning(f"For '{selected_metric}' histogram with log scale: Found {non_positive_count_before} non-positive value(s) in '{value_col_name}'. These will be excluded for log transformation.")
            df_hist_data = df_hist_data[df_hist_data[value_col_name] > 0]
        if df_hist_data.empty and non_positive_count_before > 0:
            st.error(f"Cannot generate log-scaled histogram for '{selected_metric}': All data points were non-positive.")
            return fig_box, None
    
    if df_hist_data.empty:
         st.warning(f"Cannot generate histogram for '{selected_metric}': Data became empty after filtering (if applicable).")
    else:
        try:
            fig_hist = px.histogram(
                df_hist_data, x=value_col_name, color=source_col_name,
                title=f"Overall Distribution of {selected_metric} (Mimic vs Cancer)",
                labels={value_col_name: selected_metric, source_col_name: "Source"},
                barmode='overlay', 
                opacity=0.7,
                log_x=current_log_x_hist, # Use the (potentially overridden) log setting
                nbins=50 if selected_metric == 'Affinity (nM)' and not current_log_x_hist else None, # nbins might behave differently with log
                color_discrete_map=color_map
            )
            fig_hist.update_layout(xaxis_title=selected_metric, yaxis_title="Count")
        except Exception as e: st.error(f"Error generating histogram for {selected_metric}: {e}")
    # st.markdown("---") # End of debug info section

    return fig_box, fig_hist

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🧬 Mimic Peptide Binding Explorer")
st.markdown("Explore Affinity (nM) and %Rank EL distributions for selected mimics across different HLA types.")

# --- File Uploader ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader(
    "Upload your 'final_300_mimics_all_original_data.csv' file (or similar format):", type=["csv"])

# Initialize session state for filters if they don't exist
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
            st.sidebar.error(f"One or more required columns for filtering are missing. Please ensure these columns are present: {req_cols_check}")
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
        cancer_peptide_list = ['All'] + sorted(filtered_df_step1[CANCER_PEPTIDE_COL].unique().tolist())
        selected_cancer_peptides = st.sidebar.multiselect(
            "Select Cancer Peptide(s):", options=cancer_peptide_list, default=['All'] )

        if 'All' not in selected_cancer_peptides and selected_cancer_peptides:
            filtered_df_step2 = filtered_df_step1[filtered_df_step1[CANCER_PEPTIDE_COL].isin(selected_cancer_peptides)]
        else:
            filtered_df_step2 = filtered_df_step1.copy()

        st.sidebar.markdown("---")
        st.sidebar.subheader("Cancer Accession (Gene)")
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

        st.sidebar.markdown("---")
        st.sidebar.subheader("HLA Selection")
        all_available_hlas_in_filtered_data = sorted(filtered_df_step3[HLA_COL].unique())

        critical_set = set(CRITICAL_HLAS); nice_to_have_set = set(NICE_TO_HAVE_HLAS)
        hlas_in_data_critical = sorted([h for h in all_available_hlas_in_filtered_data if h in critical_set])
        hlas_in_data_nice = sorted([h for h in all_available_hlas_in_filtered_data if h in nice_to_have_set])
        hlas_in_data_other = sorted([h for h in all_available_hlas_in_filtered_data if h not in critical_set and h not in nice_to_have_set])

        def create_hla_expander(title, hla_list, category_key_suffix):
            if not hla_list:
                st.sidebar.markdown(f"_{title} (0 available in current data)_")
                return
            with st.sidebar.expander(f"{title} ({len(hla_list)} available)", expanded=False):
                col1, col2 = st.columns(2)
                if col1.button("Select All", key=f"btn_select_all_hla_{category_key_suffix}", use_container_width=True):
                    for hla in hla_list: st.session_state.hla_selected_states[hla] = True
                if col2.button("Deselect All", key=f"btn_deselect_all_hla_{category_key_suffix}", use_container_width=True):
                    for hla in hla_list: st.session_state.hla_selected_states[hla] = False

                for hla in hla_list:
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
        elif not all_available_hlas_in_filtered_data:
            filtered_df_final = filtered_df_step3.copy()
            st.sidebar.info("No HLAs available for current filter selections.")
        else:
            filtered_df_final = filtered_df_step3[filtered_df_step3[HLA_COL].isin(hlas_to_display)]

        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Select Plots to Display")
        show_affinity_plots = st.sidebar.checkbox("Affinity (nM) Plots", value=True)
        show_el_rank_plots = st.sidebar.checkbox("%Rank EL Plots", value=True)
        st.sidebar.markdown("---")

        st.sidebar.info(f"""
        Current Data View:
        - **{filtered_df_final[MIMIC_PEPTIDE_COL].nunique()}** Unique Mimic Peptides
        - **{filtered_df_final[CANCER_PEPTIDE_COL].nunique()}** Unique Cancer Peptides
        - **{filtered_df_final[CANCER_ACC_COL].nunique()}** Unique Cancer Accessions
        - **{filtered_df_final[HLA_COL].nunique()}** Unique HLAs
        - **{len(filtered_df_final)}** Total Data Rows
        """)

        st.header("2. Explore Distributions")
        if filtered_df_final.empty:
            st.warning("No data matches the current filter selections. Please adjust filters or upload a compatible dataset.")
        else:
            if show_affinity_plots:
                st.markdown("### Affinity (nM) Distributions")
                aff_box_fig, aff_hist_fig = generate_plots(filtered_df_final, 'Affinity (nM)', COLOR_MAP, num_mimics_selected_for_plot)
                if aff_box_fig: st.plotly_chart(aff_box_fig, use_container_width=True)
                if aff_hist_fig: st.plotly_chart(aff_hist_fig, use_container_width=True)
                st.markdown("---") 
            if show_el_rank_plots:
                st.markdown("### %Rank EL Distributions")
                el_box_fig, el_hist_fig = generate_plots(filtered_df_final, '%Rank EL', COLOR_MAP, num_mimics_selected_for_plot)
                if el_box_fig: st.plotly_chart(el_box_fig, use_container_width=True)
                if el_hist_fig: st.plotly_chart(el_hist_fig, use_container_width=True)
                st.markdown("---") 
            if not show_affinity_plots and not show_el_rank_plots:
                st.info("Select at least one plot type from the sidebar to display visualizations.")

            st.header("3. Filtered Data View")
            display_cols_options = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL,
                                    MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
            if 'source_file_name' in filtered_df_final.columns:
                display_cols_options.append('source_file_name')

            display_cols_exist = [col for col in display_cols_options if col in filtered_df_final.columns]
            st.dataframe(filtered_df_final[display_cols_exist])

            with st.expander("Show Full Data Table (all columns from filtered data)"):
                st.dataframe(filtered_df_final)
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
3.  **Prepare Data:** Ensure your CSV data file (e.g., `final_300_mimics_all_original_data.csv`) is ready. It should contain columns like `mimic_Peptide`, `cancer_Peptide`, `MHC`, `mimic_Aff(nM)`, `cancer_Aff(nM)`, etc.
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
