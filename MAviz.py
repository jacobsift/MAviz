# streamlit_app.py (or MAviz.py)
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import io
import uuid # For unique temp table names

# Snowflake specific imports
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
# Ensure 'openpyxl', 'snowflake-connector-python', 'pyarrow<19.0.0'
# (and potentially 'sqlalchemy') are in your requirements.txt

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

@st.cache_data # Caching enabled for initial data loading and preprocessing
def process_uploaded_data(uploaded_file_obj):
    """Loads and preprocesses data from an uploaded file object."""
    try:
        st.info(f"Reading uploaded file '{uploaded_file_obj.name}'...")
        df = pd.read_csv(uploaded_file_obj)
        st.success(f"Successfully read file! Shape: {df.shape}. Preprocessing...")

        required_cols = [MIMIC_PEPTIDE_COL, HLA_COL]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            st.error(f"Essential columns {missing_required} are missing. Cannot proceed.")
            return None

        all_expected_cols = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL, MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
        missing_optional = [col for col in all_expected_cols if col not in df.columns and col not in missing_required]
        if missing_optional:
            st.warning(f"Optional columns missing: {missing_optional}. Some features/filters may be limited.")

        numeric_cols = [MIMIC_AFF_COL, CANCER_AFF_COL, MIMIC_EL_COL, CANCER_EL_COL]
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        string_cols = [MIMIC_PEPTIDE_COL, CANCER_PEPTIDE_COL, CANCER_ACC_COL, HLA_COL]
        for col in string_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        st.success(f"Preprocessing complete for '{uploaded_file_obj.name}'.")
        return df
    except Exception as e:
        st.error(f"Error reading or processing the uploaded CSV file: {e}")
        st.exception(e)
        return None

# --- Snowflake Processing Function ---
def process_data_in_snowflake(input_df, filters):
    """
    Uploads a DataFrame to a temporary Snowflake table, applies filters via SQL,
    and returns the resulting DataFrame.
    """
    if input_df.empty:
        st.warning("Input data for Snowflake processing is empty.")
        return pd.DataFrame()

    temp_table_name = f"TEMP_UPLOAD_{str(uuid.uuid4()).replace('-', '')}"
    conn = None
    try:
        conn = snowflake.connector.connect(**st.secrets["snowflake"])
        cs = conn.cursor()

        st.info(f"Uploading data ({input_df.shape[0]} rows) to temporary Snowflake table...")

        original_columns = input_df.columns.tolist()
        safe_columns = [f'"{col}"' for col in original_columns] # Quote all for safety
        df_upload = input_df.copy()
        df_upload.columns = safe_columns

        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df_upload,
            table_name=temp_table_name,
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
            quote_identifiers=False,
            auto_create_table=True,
            table_type='temporary'
        )

        if not success:
            st.error("Failed to upload data to temporary Snowflake table.")
            return pd.DataFrame()

        st.info(f"Data uploaded. Building and executing filter query...")

        # --- Build SQL WHERE Clause ---
        where_clauses = []
        sql_params = {} # Use parameters dictionary

        # Binding Thresholds
        if filters.get('enable_affinity_filter'):
            max_aff = filters.get('max_affinity', 500.0)
            # Reference quoted columns used in df_upload
            if f'"{MIMIC_AFF_COL}"' in df_upload.columns:
                 # Note: Using %(name)s style placeholders
                 where_clauses.append(f'("{MIMIC_AFF_COL}" <= %(max_aff)s OR "{MIMIC_AFF_COL}" IS NULL)')
            if f'"{CANCER_AFF_COL}"' in df_upload.columns:
                 where_clauses.append(f'("{CANCER_AFF_COL}" <= %(max_aff)s OR "{CANCER_AFF_COL}" IS NULL)')
            sql_params['max_aff'] = max_aff # Add value to params dict

        if filters.get('enable_el_rank_filter'):
            max_el = filters.get('max_el_rank', 2.0)
            if f'"{MIMIC_EL_COL}"' in df_upload.columns:
                 where_clauses.append(f'("{MIMIC_EL_COL}" <= %(max_el)s OR "{MIMIC_EL_COL}" IS NULL)')
            if f'"{CANCER_EL_COL}"' in df_upload.columns:
                 where_clauses.append(f'("{CANCER_EL_COL}" <= %(max_el)s OR "{CANCER_EL_COL}" IS NULL)')
            sql_params['max_el'] = max_el

        # Select Mimic Peptide(s)
        selected_mimics = filters.get('selected_mimics', ['All'])
        if selected_mimics and 'All' not in selected_mimics:
            param_name = "mimic_list"
            # Use %(name)s style placeholder
            where_clauses.append(f'"{MIMIC_PEPTIDE_COL}" IN %({param_name})s')
            sql_params[param_name] = tuple(selected_mimics) # Add tuple to params dict

        # Select Cancer Peptide(s)
        selected_cancer_peptides = filters.get('selected_cancer_peptides', ['All'])
        if selected_cancer_peptides and 'All' not in selected_cancer_peptides:
            param_name = "cancer_peptide_list"
            if f'"{CANCER_PEPTIDE_COL}"' in df_upload.columns:
                where_clauses.append(f'"{CANCER_PEPTIDE_COL}" IN %({param_name})s')
                sql_params[param_name] = tuple(selected_cancer_peptides)

        # Select Cancer Accession(s)
        selected_cancer_accs = filters.get('selected_cancer_accs', [])
        if selected_cancer_accs:
            param_name = "cancer_acc_list"
            if f'"{CANCER_ACC_COL}"' in df_upload.columns:
                where_clauses.append(f'"{CANCER_ACC_COL}" IN %({param_name})s')
                sql_params[param_name] = tuple(selected_cancer_accs)

        # Select HLA(s) for Visualization
        selected_hlas_viz = filters.get('selected_hlas_viz', [])
        if selected_hlas_viz:
            param_name = "hla_viz_list"
            if f'"{HLA_COL}"' in df_upload.columns:
                where_clauses.append(f'"{HLA_COL}" IN %({param_name})s')
                sql_params[param_name] = tuple(selected_hlas_viz)

        # Combine WHERE clauses
        sql_where = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Construct the raw query string with placeholders
        select_cols_sql = ", ".join(safe_columns)
        final_query_template = f'SELECT {select_cols_sql} FROM "{st.secrets["snowflake"]["database"]}"."{st.secrets["snowflake"]["schema"]}"."{temp_table_name}" WHERE {sql_where}'

        # **** FIX: Escape literal '%' signs in the query template ****
        # Replace single '%' with '%%' EXCEPT for the parameter placeholders like %(name)s
        # We can do this by temporarily replacing placeholders, escaping %, then restoring placeholders.
        # A simpler approach for this specific case is to just escape all % before execute
        final_query_escaped = final_query_template.replace('%', '%%')

        st.info("Executing query on Snowflake...")
        # st.write(f"Escaped Query Template:\n```sql\n{final_query_escaped}\n```") # Debug
        # st.write(f"Parameters: {sql_params}") # Debug

        # Execute using the escaped query string and the parameter dictionary
        cs.execute(final_query_escaped, sql_params)
        df_filtered = cs.fetch_pandas_all()

        # Rename columns back to original names
        df_filtered.columns = original_columns

        st.success(f"Filtered data received from Snowflake! Shape: {df_filtered.shape}")
        return df_filtered

    except snowflake.connector.errors.ProgrammingError as pe:
         st.error(f"Snowflake Programming Error (check SQL syntax, permissions, identifiers): {pe}")
         st.exception(pe)
         # Attempt to get more specific error info if available
         if "bind variable" in str(pe):
              st.error("This might be related to parameter binding. Check parameter names and types.")
         return pd.DataFrame()
    except ValueError as ve:
         # Catch the specific ValueError we saw before, though escaping should prevent it
         st.error(f"ValueError during SQL processing (likely parameter formatting): {ve}")
         st.exception(ve)
         return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during Snowflake processing: {e}")
        st.exception(e)
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


# --- Plotting and Other Functions ---
# (reshape_for_plotting and generate_plots remain the same as previous version)
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
        st.exception(e)
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
    if df_filtered.empty:
         st.warning(f"No data available to generate plots for {selected_metric}.")
         return None, None

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

            if df_boxplot_data.empty:
                st.warning(f"Boxplot data empty for '{selected_metric}' after reshaping/cleaning.")
                hla_order = []
            elif use_log_x_boxplot:
                df_for_median = df_boxplot_data[df_boxplot_data[value_col_name] > 0]
                if not df_for_median.empty:
                    median_vals = df_for_median.groupby(HLA_COL)[value_col_name].median()
                    hla_order = median_vals.replace([np.inf, -np.inf], np.nan).dropna().sort_values().index.tolist()
                else:
                    hla_order = []
                missing_hlas = sorted([h for h in df_boxplot_data[HLA_COL].unique() if h not in hla_order])
                hla_order.extend(missing_hlas)
            else:
                hla_order = sorted(df_boxplot_data[HLA_COL].unique())

            if not df_boxplot_data.empty:
                fig_box = px.box(
                    df_boxplot_data, x=value_col_name, y=HLA_COL, color=source_col_name,
                    title=f"{selected_metric} Distribution per HLA (Mimic vs Cancer)",
                    labels={value_col_name: selected_metric, HLA_COL: "HLA Type", source_col_name: "Source"},
                    orientation='h', log_x=use_log_x_boxplot,
                    category_orders={HLA_COL: hla_order} if hla_order else {},
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
            st.exception(e)
            fig_box = None
    elif HLA_COL not in df_melted.columns:
        st.info(f"HLA column ('{HLA_COL}') not found in the data for '{selected_metric}'. Skipping box plot.")
        fig_box = None

    # --- Plot 2: Histogram ---
    df_hist_data = df_melted

    if df_hist_data.empty:
        st.warning(f"Cannot generate histogram for '{selected_metric}': Data is empty after reshaping/cleaning.")
    else:
        try:
            if selected_metric == 'Affinity (nM)':
                st.info("Note: Affinity (nM) histogram is displayed on a linear scale due to rendering issues with log scale.")
                fig_hist = px.histogram(
                    df_hist_data, x=value_col_name, color=source_col_name,
                    title=f"Overall Distribution of {selected_metric} (Mimic vs Cancer) [Linear Scale]",
                    labels={value_col_name: selected_metric, source_col_name: "Source"},
                    barmode='overlay', opacity=0.7, log_x=False, marginal='rug', nbins=None, color_discrete_map=color_map
                )
                fig_hist.update_layout(xaxis_title=selected_metric + " (Linear Scale)", yaxis_title="Count")

            elif selected_metric == '%Rank EL':
                fig_hist = px.histogram(
                    df_hist_data, x=value_col_name, color=source_col_name,
                    title=f"Overall Distribution of {selected_metric} (Mimic vs Cancer)",
                    labels={value_col_name: selected_metric, source_col_name: "Source"},
                    barmode='overlay', opacity=0.7, log_x=False, marginal='rug', nbins=None, color_discrete_map=color_map
                )
                fig_hist.update_layout(xaxis_title=selected_metric, yaxis_title="Count")

            else:
                st.warning(f"Unknown metric '{selected_metric}' for histogram generation.")
                fig_hist = None

        except Exception as e:
            st.error(f"Error generating histogram for {selected_metric}: {e}")
            st.exception(e)
            fig_hist = None

    return fig_box, fig_hist

# --- Streamlit App Layout ---
# [ Remains the same as previous version - Omitted for brevity ]
# ... st.set_page_config ...
# ... st.title ...
# ... st.markdown ...

# --- File Uploader ---
# [ Remains the same as previous version - Omitted for brevity ]
# ... st.header("Upload Data") ...
# ... uploaded_file = st.file_uploader(...) ...

# --- Session State Init ---
# [ Remains the same as previous version - Omitted for brevity ]
# ... if 'hla_selected_states' not in st.session_state: ...

# --- Main Processing Logic ---
if uploaded_file is not None:
    df_initial_read = process_uploaded_data(uploaded_file)

    if df_initial_read is not None and not df_initial_read.empty:

        # --- Display Options ---
        # [ Remains the same as previous version - Omitted for brevity ]
        # ... st.markdown("---") ...
        # ... st.subheader("📊 Display Options") ...
        # ... checkboxes ...
        # ... st.markdown("---") ...

        # --- Sidebar Filters ---
        # [ Remains the same as previous version - Omitted for brevity ]
        # ... st.sidebar.header("⚙️ Filters") ...
        # ... Pre-Snowflake Filtering (Must Bind To HLA) ...
        # ... Filters for Snowflake SQL Query (Thresholds, Mimic Select, Antigen Seq, Cancer Pep, Cancer Acc, HLA Viz) ...
        # ... Collect snowflake_filters dictionary ...

        # --- Step 2: Call Snowflake Processing ---
        filtered_df_intermediate = pd.DataFrame()
        if df_ready_for_snowflake.empty:
             st.warning("Data is empty after 'Must Bind To' filter. Skipping Snowflake processing.")
             filtered_df_intermediate = pd.DataFrame(columns=df_initial_read.columns)
        else:
            with st.spinner("Processing data in Snowflake..."):
                filtered_df_intermediate = process_data_in_snowflake(df_ready_for_snowflake, snowflake_filters)

        # --- Step 3: Post-Snowflake Filtering (Pandas) ---
        filtered_df_final = filtered_df_intermediate.copy()
        if pasted_antigen_sequence and not filtered_df_intermediate.empty:
             if CANCER_PEPTIDE_COL in filtered_df_intermediate.columns:
                 try:
                     with st.spinner("Applying antigen sequence filter..."):
                         df_temp_antigen = filtered_df_intermediate.copy()
                         df_temp_antigen[CANCER_PEPTIDE_COL] = df_temp_antigen[CANCER_PEPTIDE_COL].astype(str).fillna('')
                         pasted_seq_upper = pasted_antigen_sequence.upper()
                         condition = (
                             df_temp_antigen[CANCER_PEPTIDE_COL].str.upper().apply(lambda cp: pasted_seq_upper in cp if cp else False) |
                             df_temp_antigen[CANCER_PEPTIDE_COL].str.upper().apply(lambda cp: cp in pasted_seq_upper if cp and pasted_seq_upper else False)
                         )
                         filtered_df_final = filtered_df_intermediate[condition]
                     if filtered_df_final.empty: st.sidebar.warning("No results remaining after antigen sequence filter.")
                     else: st.sidebar.caption(f"{len(filtered_df_final)} rows remain after antigen sequence filter.")
                 except Exception as e_antigen:
                     st.sidebar.error(f"Error applying antigen sequence filter: {e_antigen}")
                     st.exception(e_antigen)
                     filtered_df_final = filtered_df_intermediate.copy()
             else:
                 st.sidebar.warning(f"Cannot apply antigen sequence filter: '{CANCER_PEPTIDE_COL}' not found in results.")
                 filtered_df_final = filtered_df_intermediate.copy()

        # --- Sidebar Summary ---
        # [ Remains the same as previous version - Omitted for brevity ]
        # ... st.sidebar.markdown("---") ...
        # ... calculate counts ...
        # ... st.sidebar.info(...) ...

        # --- Main Page Content ---
        # [ Remains the same as previous version - Omitted for brevity ]
        # ... if filtered_df_final.empty: ...
        # ... else: ...
        # ...   Explore Distributions ...
        # ...   Filtered Data View ...
        # ...   Export Data Section ...

    else: # df_initial_read is None or empty
         if uploaded_file is not None:
             st.error("Failed to process the uploaded data. Please check the file format and content.")
else:
    st.info("👋 Welcome! Please upload your CSV data file to begin exploring peptide binding.")

st.sidebar.markdown("---")

