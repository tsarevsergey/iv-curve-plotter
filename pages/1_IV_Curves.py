import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.parser import parse_filename, handle_duplicates
from src.data_loader import load_dat_file, check_short_circuit
from src.calculations import calculate_metrics
import os
import glob

# Page Config
st.set_page_config(page_title="IV Curve Plotter", layout="wide")

# Title
st.title("IV Curve Plotter for Perovskite Photodetectors")

# Sidebar - Global Settings
st.sidebar.header("Settings")

# Light Source Params
st.sidebar.subheader("Measurement Parameters")
radiance = st.sidebar.number_input("Light Power (mW/cm²)", value=1.0, step=0.1)
# Convert mW/cm2 to W/cm2 for calculation
radiance_w_cm2 = radiance / 1000.0

area = st.sidebar.number_input("Device Area (cm²)", value=0.16, step=0.01)

# Wavelength mapping (could be expanded)
st.sidebar.markdown("---")
st.sidebar.subheader("Wavelengths (nm)")
wl_red = st.sidebar.number_input("Red (R)", value=626)
wl_green = st.sidebar.number_input("Green (G)", value=522)
wl_blue = st.sidebar.number_input("Blue (B)", value=461)

wavelength_map = {'R': wl_red, 'G': wl_green, 'B': wl_blue}

# Analysis Params
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Targets")
target_voltage_eqe = st.sidebar.number_input("Voltage for EQE (V)", value=0.0, step=0.1)
target_voltage_dark = st.sidebar.number_input("Voltage for Dark Current (V)", value=-0.5, step=0.1)

# Quality Control Params
st.sidebar.markdown("---")
st.sidebar.subheader("Quality Control")
filter_shorted = st.sidebar.checkbox("Filter Shorted Devices", value=False)
short_threshold = st.sidebar.number_input("Min Light/Dark Ratio @ -0.5V", value=3.0, step=0.5)

filter_bad_contact = st.sidebar.checkbox("Filter Bad Contact (Low EQE)", value=False)
min_eqe_threshold = st.sidebar.number_input("Min EQE (%)", value=1.0, step=0.1)

# Duplicate Handling
st.sidebar.markdown("---")
duplicate_strategy = st.sidebar.selectbox("Duplicate Handling", ["Keep Latest", "Keep Original", "Keep All"])

# Main Area - File Input
st.header("1. Load Data")

col_upload, col_folder = st.columns(2)

# 1. Drag and Drop
with col_upload:
    uploaded_files = st.file_uploader("Option A: Drag and drop .dat files", accept_multiple_files=True, type=['dat'])

# 2. Folder Input
with col_folder:
    folder_path = st.text_input("Option B: Paste Folder Path (Recursive Search)")
    load_folder_btn = st.button("Load from Folder")

# Initialize Session State for Files if not exists
if 'all_parsed_files' not in st.session_state:
    st.session_state['all_parsed_files'] = []

# Process Inputs
new_files = []

# Process Uploaded Files
if uploaded_files:
    for uploaded_file in uploaded_files:
        meta = parse_filename(uploaded_file.name)
        if meta:
            meta['file_source'] = 'upload'
            meta['file_obj'] = uploaded_file # Store reference to file object
            meta['file_path'] = None
            new_files.append(meta)

# Process Folder Files
if load_folder_btn and folder_path:
    if os.path.isdir(folder_path):
        # Recursive search
        found_files = glob.glob(os.path.join(folder_path, '**/*.dat'), recursive=True)
        st.toast(f"Found {len(found_files)} .dat files in folder.")
        
        for file_path in found_files:
            filename = os.path.basename(file_path)
            meta = parse_filename(filename)
            if meta:
                # Handle Subfolder in Batch Name
                rel_path = os.path.relpath(file_path, start=folder_path)
                parent_dir = os.path.dirname(rel_path)
                
                if parent_dir:
                    # Normalize path separators
                    parent_dir = parent_dir.replace(os.sep, '/')
                    meta['batch'] = f"{parent_dir}/{meta['batch']}"
                    # Update ID to ensure uniqueness across folders
                    meta['id'] = f"{meta['batch']}{meta['device']}_{meta['pixel']}{meta['color'] or ''}{meta['condition']}"
                
                meta['file_source'] = 'local'
                meta['file_obj'] = None
                meta['file_path'] = file_path
                new_files.append(meta)
    else:
        st.error("Invalid folder path.")

# Update Session State
if new_files:
    # Append new files to existing session state
    existing_ids = {f['id'] for f in st.session_state['all_parsed_files']}
    for f in new_files:
        if f['id'] not in existing_ids:
            st.session_state['all_parsed_files'].append(f)

# Main Logic
if st.session_state['all_parsed_files']:
    all_files = st.session_state['all_parsed_files']
    
    # Handle Duplicates (Global Strategy)
    unique_files = handle_duplicates(all_files, strategy=duplicate_strategy)
    
    st.info(f"Total Loaded: {len(unique_files)} unique measurements.")
    
    # Convert to DataFrame
    df_meta = pd.DataFrame(unique_files)
    
    if not df_meta.empty:
        # --- STEP 2: GLOBAL FILTERS ---
        st.header("2. Filter & Group")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            all_batches = sorted(df_meta['batch'].unique().tolist())
            selected_batches = st.multiselect("Filter by Batch", all_batches, default=all_batches)
            
        with col_f2:
            # Filter by Device (in selected batches)
            available_devices = sorted(df_meta[df_meta['batch'].isin(selected_batches)]['device'].unique().tolist())
            selected_devices = st.multiselect("Filter by Device", available_devices, default=available_devices)

        with col_f3:
            # Filter by Pixel
            all_pixels = sorted(df_meta['pixel'].unique().tolist())
            selected_pixels = st.multiselect("Filter by Pixel", all_pixels, default=all_pixels)
        
        # Apply Global Filters
        pre_filtered_meta = df_meta[
            (df_meta['batch'].isin(selected_batches)) & 
            (df_meta['device'].isin(selected_devices)) &
            (df_meta['pixel'].isin(selected_pixels))
        ].copy()
        
        # --- STEP 2.5: QUALITY CONTROL FILTERING ---
        # Apply QC *before* the editor so bad devices don't even show up (or are removed).
        # We need to identify (batch, device, pixel) tuples that fail QC.
        
        if filter_shorted or filter_bad_contact:
            st.info("Running Quality Control checks...")
            
            # Get unique pixels to check
            unique_pixels = pre_filtered_meta[['batch', 'device', 'pixel']].drop_duplicates()
            bad_pixels = set()
            
            # We need a progress bar here because this might involve file I/O
            qc_progress = st.progress(0)
            total_checks = len(unique_pixels)
            
            for idx, (i, row) in enumerate(unique_pixels.iterrows()):
                # Find Light and Dark files for this pixel
                pixel_files = pre_filtered_meta[
                    (pre_filtered_meta['batch'] == row['batch']) & 
                    (pre_filtered_meta['device'] == row['device']) & 
                    (pre_filtered_meta['pixel'] == row['pixel'])
                ]
                
                # We need at least one Light file to check QC (usually)
                light_file = pixel_files[pixel_files['condition'] == 'LIGHT'].head(1)
                dark_file = pixel_files[pixel_files['condition'] == 'DARK'].head(1)
                
                is_bad = False
                
                if not light_file.empty and not dark_file.empty:
                    # Load Data
                    try:
                        # Helper to load
                        def load_meta(m):
                            if m['file_source'] == 'upload':
                                m['file_obj'].seek(0)
                                return load_dat_file(m['file_obj'])
                            else:
                                return load_dat_file(m['file_path'])

                        df_light = load_meta(light_file.iloc[0])
                        df_dark = load_meta(dark_file.iloc[0])
                        
                        if df_light is not None and df_dark is not None:
                            from src.calculations import check_quality
                            
                            # Check Short
                            if filter_shorted:
                                qc = check_quality(df_light, df_dark, voltage_check=-0.5, threshold_ratio=short_threshold)
                                if qc['is_shorted']:
                                    is_bad = True
                            
                            # Check Bad Contact (if not already bad)
                            if not is_bad and filter_bad_contact:
                                # Calculate EQE for the first available wavelength map color
                                # We need to know the color of the light file
                                c = light_file.iloc[0]['color']
                                if c in wavelength_map:
                                    wl = wavelength_map[c]
                                    metrics = calculate_metrics(
                                        df_light, df_dark, 
                                        target_voltage_eqe, wl, 
                                        radiance=radiance_w_cm2, area=area
                                    )
                                    if metrics['eqe'] < min_eqe_threshold:
                                        is_bad = True
                    except Exception as e:
                        print(f"QC Error: {e}")
                
                if is_bad:
                    bad_pixels.add((row['batch'], row['device'], row['pixel']))
                
                qc_progress.progress((idx + 1) / total_checks)
            
            qc_progress.empty()
            
            if bad_pixels:
                st.warning(f"Filtered out {len(bad_pixels)} pixels due to QC.")
                # Filter pre_filtered_meta
                # Remove rows where (batch, device, pixel) is in bad_pixels
                # A quick way is to create a tuple index
                
                def is_kept(r):
                    return (r['batch'], r['device'], r['pixel']) not in bad_pixels
                
                pre_filtered_meta = pre_filtered_meta[pre_filtered_meta.apply(is_kept, axis=1)]

        # --- STEP 3: FINE-TUNE SELECTION & GROUPING ---
        with st.expander("3. Fine-tune Selection & Grouping", expanded=False):
            st.markdown("Use the table below to **uncheck specific pixels** or **rename groups** (to merge devices in the legend).")
            
            # Initialize editor_data in session state if not present
            if 'editor_data' not in st.session_state:
                 st.session_state['editor_data'] = pd.DataFrame(columns=['id', 'Include', 'Group Label'])
    
            # Merge current filtered data with saved editor state (to preserve edits)
            # We want all rows from pre_filtered_meta
            # And 'Include', 'Group Label' from session state if they exist for that ID
            
            # 1. Get saved state
            saved_edits = st.session_state['editor_data'][['id', 'Include', 'Group Label']]
            
            # 2. Merge (Left Join)
            merged_df = pd.merge(pre_filtered_meta, saved_edits, on='id', how='left')
            
            # 3. Fill Missing Values (for new files or files not in saved state)
            merged_df['Include'] = merged_df['Include'].fillna(True)
            
            # For Group Label, if NaN, generate default
            def get_default_group(row):
                if pd.isna(row['Group Label']):
                    return f"{row['batch']}_Dev{row['device']}"
                return row['Group Label']
                
            merged_df['Group Label'] = merged_df.apply(get_default_group, axis=1)
            
            # 4. Prepare DataFrame for Editor
            editor_input = merged_df[['Include', 'Group Label', 'batch', 'device', 'pixel', 'color', 'condition', 'id']].copy()
    
            # Display Editor
            edited_df = st.data_editor(
                editor_input, 
                column_config={
                    "Include": st.column_config.CheckboxColumn("Plot?", help="Uncheck to hide this specific curve", default=True),
                    "Group Label": st.column_config.TextColumn("Legend Group", help="Change this to group multiple devices together"),
                    "id": None # Hide ID
                },
                disabled=["batch", "device", "pixel", "color", "condition"],
                hide_index=True,
                use_container_width=True,
                height=300
            )
            
            # Update Session State with new edits (so they persist)
            # We only need to store ID, Include, Group Label
            st.session_state['editor_data'] = edited_df[['id', 'Include', 'Group Label']]
        
        # Final Filter based on Editor
        final_selection = edited_df[edited_df['Include'] == True]
        
        st.write(f"Plotting {len(final_selection)} curves.")

        # Load Data
        data_cache = {}
        results = []
        
        # Progress bar
        if len(final_selection) > 0:
            progress_bar = st.progress(0)
            
            # Convert to list of dicts for iteration
            selection_records = final_selection.to_dict('records')
            
            # Pre-load all necessary data to handle filters efficiently
            # Actually, we process row by row. If a row is filtered out by QC, we just don't add it to results/plots.
            
            filtered_results_count = 0
            
            for i, row in enumerate(selection_records):
                # Load data
                try:
                    # We need to find the original file info (path/obj) from unique_files using ID
                    orig_meta = next((item for item in unique_files if item['id'] == row['id']), None)
                    
                    if not orig_meta:
                        continue
                        
                    if orig_meta['file_source'] == 'upload':
                        orig_meta['file_obj'].seek(0)
                        df = load_dat_file(orig_meta['file_obj'])
                    else:
                        df = load_dat_file(orig_meta['file_path'])
                    
                    if df is not None:
                        # Check health (Basic short circuit check from data_loader - maybe redundant now?)
                        # status = check_short_circuit(df) 
                        
                        # Store in cache (we might need it for plotting even if it fails QC? No, if filtered, don't plot)
                        # But wait, we need Dark data for Light QC.
                        
                        # Calculation Logic
                        if row['condition'] == 'LIGHT' and row['color'] in wavelength_map:
                            # Find matching dark
                            dark_id = row['id'].replace('LIGHT', 'DARK')
                            dark_meta = next((item for item in unique_files if item['id'] == dark_id), None)
                            
                            if dark_meta:
                                if dark_meta['file_source'] == 'upload':
                                    dark_meta['file_obj'].seek(0)
                                    df_dark = load_dat_file(dark_meta['file_obj'])
                                else:
                                    df_dark = load_dat_file(dark_meta['file_path'])
                                
                                # QC is already done globally!
                                
                                wl = wavelength_map.get(row['color'], 0)
                                metrics = calculate_metrics(
                                    df, df_dark, 
                                    target_voltage_eqe, wl, 
                                    radiance=radiance_w_cm2, area=area
                                )
                                
                                # If passed QC, add to results and cache for plotting
                                data_cache[row['id']] = df
                                metrics.update(row) # Includes Group Label
                                metrics['type'] = 'LIGHT'
                                results.append(metrics)
                                filtered_results_count += 1
                        
                        elif row['condition'] == 'DARK':
                             # For Dark files, we usually plot them if their corresponding Light file is plotted?
                             # Or just plot them if selected.
                             # If we filter Light files based on QC, we might want to filter the Dark ones too?
                             # It's hard to link them backwards easily without a map.
                             # For now, let's just plot Dark files if they are selected, unless we want to apply "Shorted" check to them too?
                             # Short check needs Light data usually (Ratio).
                             # But we can check if Dark current is too high?
                             
                             # Let's just calculate metrics and add.
                             i_dark = calculate_metrics(None, df, target_voltage_dark, 0, area=area)['i_dark']
                             res = {
                                 'j_dark': abs(i_dark)/area,
                                 'voltage': target_voltage_dark,
                                 'type': 'DARK'
                             }
                             res.update(row)
                             results.append(res)
                             data_cache[row['id']] = df
                             filtered_results_count += 1

                except Exception as e:
                    print(f"Error processing {row.get('id')}: {e}")

                progress_bar.progress((i + 1) / len(selection_records))
            
        st.success(f"Data Processed! {filtered_results_count} curves plotted.")
        
        # Tabs for Visualization
        tab1, tab2 = st.tabs(["IV Curves", "Analysis"])
        
        with tab1:
            st.subheader("IV Curves")
            
            col1, col2 = st.columns(2)
            with col1:
                scale_type = st.radio("Y-Axis Scale", ["Linear", "Log"], index=1)
            
            # Plotting
            fig = go.Figure()
            
            # Track which groups have been added to legend to avoid duplicates
            legend_groups_added = set()
            
            # Color Map for Groups (Dynamic)
            unique_groups = final_selection['Group Label'].unique()
            # Use Plotly's default color cycle
            colors = px.colors.qualitative.Plotly
            group_color_map = {grp: colors[i % len(colors)] for i, grp in enumerate(unique_groups)}
            
            max_traces = 500
            trace_count = 0
            
            for i, row in enumerate(selection_records):
                if trace_count > max_traces:
                    st.warning(f"Plot truncated at {max_traces} curves.")
                    break
                    
                mid = row['id']
                if mid in data_cache:
                    df = data_cache[mid]
                    
                    grp_label = row['Group Label']
                    
                    # Line Style
                    line_dash = 'solid' if row['condition'] == 'LIGHT' else 'dash'
                    
                    # Legend Logic
                    show_legend = False
                    if grp_label not in legend_groups_added:
                        show_legend = True
                        legend_groups_added.add(grp_label)
                    
                    fig.add_trace(go.Scatter(
                        x=df['V'], 
                        y=abs(df['I']), 
                        mode='lines',
                        name=grp_label, # Name in legend
                        line=dict(dash=line_dash, color=group_color_map.get(grp_label, 'black'), width=3),
                        legendgroup=grp_label,
                        showlegend=show_legend,
                        hoverinfo='text',
                        text=f"{grp_label}<br>P{row['pixel']} ({row['condition']})<br>V: %{{x}}<br>I: %{{y}}"
                    ))
                    trace_count += 1
            
            if scale_type == 'Log':
                fig.update_yaxes(type="log")
                
            # Square Layout
            fig.update_layout(
                xaxis_title="Voltage (V)",
                yaxis_title="Current (A)",
                width=800,
                height=800,
                autosize=False,
                hovermode="closest",
                font=dict(size=18), # Base font size
                xaxis=dict(
                    title_font=dict(size=24),
                    tickfont=dict(size=20)
                ),
                yaxis=dict(
                    title_font=dict(size=24),
                    tickfont=dict(size=20)
                ),
                legend=dict(
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02, # Move legend to the right
                    font=dict(size=20)
                ),
                margin=dict(r=150) # Add right margin for legend
            )
            
            # Config for high-res download
            config = {
                'toImageButtonOptions': {
                    'format': 'png', # one of png, svg, jpeg, webp
                    'filename': 'iv_curve_plot',
                    'height': 800,
                    'width': 800,
                    'scale': 3 # Multiply title/legend/axis/canvas sizes by this factor
                }
            }
            
            st.plotly_chart(fig, use_container_width=False, config=config) # False to respect fixed width
            
        with tab2:
            st.subheader("Statistical Analysis")
            
            if results:
                df_res = pd.DataFrame(results)
                
                # Common Layout Update Function
                def update_box_layout(fig):
                    fig.update_layout(
                        font=dict(size=18),
                        title_font=dict(size=24),
                        xaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
                        yaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
                        legend=dict(font=dict(size=18)),
                        width=800,
                        height=600,
                        xaxis_title=None # Remove "Group Label" title
                    )
                    return fig

                # EQE Distribution
                st.markdown(f"### EQE Distribution at {target_voltage_eqe} V")
                df_eqe = df_res[df_res['type'] == 'LIGHT']
                
                if not df_eqe.empty:
                    # Define explicit color sequence to ensure consistency
                    color_seq = px.colors.qualitative.Plotly

                    fig_eqe = px.box(
                        df_eqe, 
                        x='Group Label', 
                        y='eqe', 
                        color='Group Label', 
                        points="all",
                        hover_data=['batch', 'device', 'pixel'],
                        title=f"EQE (%) @ {target_voltage_eqe}V",
                        color_discrete_sequence=color_seq
                    )
                    fig_eqe = update_box_layout(fig_eqe)
                    
                    # Static Image Toggle
                    if st.checkbox("Render as Static Image (Right-Click -> Copy)", key="static_eqe"):
                        try:
                            # Enforce white background for PPTX, use colorful template
                            # Enforce white background for PPTX
                            fig_eqe.update_layout(
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                font=dict(color="black")
                            )
                            img_bytes = fig_eqe.to_image(format="png", scale=2, width=800, height=600)
                            st.image(img_bytes, caption="High-Res Static Image", use_column_width=False, width=800)
                        except Exception as e:
                            st.error(f"Static image generation failed: {e}. Please restart the app to load the new dependencies.")
                            st.plotly_chart(fig_eqe, use_container_width=False, config=config)
                    else:
                        st.plotly_chart(fig_eqe, use_container_width=False, config=config)
                else:
                    st.info("No Light data for EQE.")
                
                # Dark Current Distribution
                st.markdown(f"### Dark Current Density at {target_voltage_dark} V")
                
                df_dark_res = df_res[df_res['type'] == 'DARK']
                if not df_dark_res.empty:
                    fig_dark = px.box(
                        df_dark_res, 
                        x='Group Label', 
                        y='j_dark', 
                        color='Group Label', 
                        points="all",
                        hover_data=['batch', 'device', 'pixel'],
                        title=f"Dark Current Density (A/cm²) @ {target_voltage_dark}V",
                        log_y=True,
                        color_discrete_sequence=color_seq
                    )
                    fig_dark = update_box_layout(fig_dark)
                    
                    if st.checkbox("Render as Static Image (Right-Click -> Copy)", key="static_dark"):
                        try:
                            # Enforce white background for PPTX, use colorful template
                            # Enforce white background for PPTX
                            fig_dark.update_layout(
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                font=dict(color="black")
                            )
                            img_bytes = fig_dark.to_image(format="png", scale=2, width=800, height=600)
                            st.image(img_bytes, caption="High-Res Static Image", use_column_width=False, width=800)
                        except Exception as e:
                            st.error(f"Static image generation failed: {e}. Please restart the app to load the new dependencies.")
                            st.plotly_chart(fig_dark, use_container_width=False, config=config)
                    else:
                        st.plotly_chart(fig_dark, use_container_width=False, config=config)
                else:
                    st.info("No Dark data found.")

            else:
                st.warning("No results calculated.")

else:
    st.info("Awaiting file upload or folder selection...")
