import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats, constants
import os
from src.calculations import analyze_iv_numpy

st.set_page_config(page_title="Voc Analysis", layout="wide")

st.title("Voc vs Light Intensity Analysis")

# --- Sidebar ---
st.sidebar.header("Configuration")

# Light Intensity Input
default_li = "0, 0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 2, 2.4"
li_input = st.sidebar.text_area("Light Intensities (mW/cm²)", value=default_li, help="Comma-separated values corresponding to columns 1..N in the data file.")

# Temperature
temp = st.sidebar.number_input("Temperature (K)", value=300.0)

# Constants
q = constants.elementary_charge
k = constants.Boltzmann
ktq = k * temp / q

# --- Main Area ---

st.subheader("1. Load Data")
st.markdown("Paste absolute file paths below (one per line) OR upload files.")

col1, col2 = st.columns(2)

with col1:
    file_paths_input = st.text_area("File Paths (One per line)", height=150)
    load_paths_btn = st.button("Load Paths")

with col2:
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

# Process Data
data_results = {}

def process_file(file_obj, filename, li_values):
    try:
        # Read file (space separated, skip header row 1 usually? VosLIGHT skips 1)
        # VosLIGHT: pd.read_csv(path, sep=r"\s+", engine="python", header=None, skiprows=1)
        # Let's try to be robust.
        if isinstance(file_obj, str):
             df = pd.read_csv(file_obj, sep=r"\s+", engine="python", header=None, skiprows=1)
        else:
             df = pd.read_csv(file_obj, sep=r"\s+", engine="python", header=None, skiprows=1)
        
        V = df.iloc[:, 0].to_numpy(float)
        
        # Check dimensions
        num_current_cols = len(df.columns) - 1
        if num_current_cols > len(li_values):
             st.warning(f"File {filename} has {num_current_cols} current columns but only {len(li_values)} light intensities defined. Using first {len(li_values)}.")
        elif num_current_cols < len(li_values):
             st.warning(f"File {filename} has {num_current_cols} current columns but {len(li_values)} light intensities defined. Truncating LI.")
             li_values = li_values[:num_current_cols]
        
        res_list = []
        for i in range(len(li_values)):
            I = df.iloc[:, i+1].to_numpy(float)
            params = analyze_iv_numpy(V, I)
            params['LightIntensity'] = li_values[i]
            res_list.append(params)
            
        return pd.DataFrame(res_list)
        
    except Exception as e:
        st.error(f"Error processing {filename}: {e}")
        return None

# Parse LI
try:
    li_values = [float(x.strip()) for x in li_input.split(',') if x.strip()]
except:
    st.error("Invalid Light Intensity format.")
    li_values = []

if li_values:
    # Load from Paths
    if load_paths_btn and file_paths_input:
        paths = [p.strip() for p in file_paths_input.split('\n') if p.strip()]
        for p in paths:
            if os.path.exists(p):
                df_res = process_file(p, os.path.basename(p), li_values)
                if df_res is not None:
                    data_results[os.path.basename(p)] = df_res
            else:
                st.error(f"File not found: {p}")

    # Load from Upload
    if uploaded_files:
        for f in uploaded_files:
            f.seek(0)
            df_res = process_file(f, f.name, li_values)
            if df_res is not None:
                data_results[f.name] = df_res

# --- Visualization ---
if data_results:
    st.success(f"Loaded {len(data_results)} files.")
    
    tab1, tab2, tab3 = st.tabs(["Voc vs Light", "Jsc vs Light", "FF vs Light"])
    
    # Colors
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
    
    with tab1:
        st.subheader("Voc vs Light Intensity")
        
        col_fit, col_plot = st.columns([1, 3])
        
        with col_fit:
            st.markdown("### Fit Settings")
            # Range Selection
            # Assuming all files have same LI points for simplicity of UI, 
            # but we should probably allow per-file or global index range.
            # Global index range is easier.
            n_points = len(li_values)
            fit_start = st.number_input("Fit Start Index", min_value=0, max_value=n_points-1, value=2)
            fit_end = st.number_input("Fit End Index", min_value=0, max_value=n_points-1, value=8)
        
        with col_plot:
            fig_voc = go.Figure()
            
            for i, (name, df) in enumerate(data_results.items()):
                color = colors[i % len(colors)]
                
                # Filter valid Voc
                df_valid = df.dropna(subset=['Voc'])
                
                # X axis: Light Intensity (log scale for plot usually?)
                # User script plots vs log(LI) but labels axis as LI.
                # Let's plot vs LI and set x-axis to log.
                
                # Data Points
                fig_voc.add_trace(go.Scatter(
                    x=df_valid['LightIntensity'],
                    y=df_valid['Voc'],
                    mode='markers',
                    name=name,
                    marker=dict(color=color, size=10)
                ))
                
                # Fit
                # Sort by LI to ensure slicing makes sense
                df_sorted = df_valid.sort_values('LightIntensity')
                
                # Apply Index Slice (User selection)
                # We slice based on the user's indices relative to the sorted valid data
                # If the user input LI list has 0s, they are likely at the start.
                
                if fit_end < len(df_sorted):
                    df_slice = df_sorted.iloc[fit_start:fit_end+1]
                else:
                    df_slice = df_sorted.iloc[fit_start:]
                
                # Filter out <= 0 for Log Fit
                df_fit = df_slice[df_slice['LightIntensity'] > 0]
                
                if len(df_fit) > 1:
                    x_fit = np.log(df_fit['LightIntensity']) # Natural log for physics
                    y_fit = df_fit['Voc']
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
                    
                    # n = slope * q / (k * T) -> slope / ktq
                    n_factor = slope / ktq
                    
                    # Generate fit line points for the whole range (for visual)
                    # But usually we just show fit line over fit range
                    # Let's show it over the fit range to be clear what was fitted
                    x_line_log = np.linspace(x_fit.min(), x_fit.max(), 10)
                    x_line_linear = np.exp(x_line_log)
                    y_line = intercept + slope * x_line_log
                    
                    fig_voc.add_trace(go.Scatter(
                        x=x_line_linear,
                        y=y_line,
                        mode='lines',
                        name=f"{name} (n={n_factor:.2f})",
                        line=dict(color=color, dash='dash', width=3),
                        showlegend=True
                    ))
                    
                    # Highlight fitted points
                    fig_voc.add_trace(go.Scatter(
                        x=df_fit['LightIntensity'],
                        y=df_fit['Voc'],
                        mode='markers',
                        name=f"{name} (Fitted)",
                        marker=dict(color=color, symbol='circle-open', size=14, line=dict(width=2)),
                        showlegend=False
                    ))

            fig_voc.update_layout(
                xaxis_type="log",
                xaxis_title="Light Power (mW/cm²)",
                yaxis_title="Voc (V)",
                height=600,
                width=800,
                font=dict(size=18),
                xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
                yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
                legend=dict(font=dict(size=20))
            )
            
            config = {
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'voc_analysis_plot',
                    'height': 600,
                    'width': 800,
                    'scale': 3
                }
            }
            
            if st.checkbox("Render as Static Image (Right-Click -> Copy)", key="static_voc"):
                try:
                    # Enforce white background for PPTX
                    fig_voc.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(color="black")
                    )
                    img_bytes = fig_voc.to_image(format="png", scale=2, width=800, height=600)
                    st.image(img_bytes, caption="High-Res Static Image", use_column_width=False, width=800)
                except Exception as e:
                    st.error(f"Static image generation failed: {e}. Please restart the app.")
                    st.plotly_chart(fig_voc, config=config)
            else:
                st.plotly_chart(fig_voc, config=config)

    with tab2:
        st.subheader("Jsc vs Light Intensity")
        # Linear fit for Jsc
        
        fig_jsc = go.Figure()
        
        for i, (name, df) in enumerate(data_results.items()):
            color = colors[i % len(colors)]
            df_valid = df.dropna(subset=['Isc'])
            
            # Jsc is usually abs(Isc)
            y_val = abs(df_valid['Isc'])
            x_val = df_valid['LightIntensity']
            
            fig_jsc.add_trace(go.Scatter(
                x=x_val,
                y=y_val,
                mode='markers',
                name=name,
                marker=dict(color=color, size=12)
            ))
            
            # Linear Fit
            if len(df_valid) > 1:
                 slope, intercept, r, p, err = stats.linregress(x_val, y_val)
                 # Slope is roughly Responsivity (A/W) if area is 1cm2 and units match
                 
                 x_line = np.linspace(x_val.min(), x_val.max(), 100)
                 y_line = intercept + slope * x_line
                 
                 fig_jsc.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f"{name} (Slope={slope:.2e})",
                    line=dict(color=color, dash='dash', width=3)
                 ))
        
        fig_jsc.update_layout(
            xaxis_title="Light Power (mW/cm²)",
            yaxis_title="Isc (A)",
            height=600,
            width=800,
            font=dict(size=18),
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            legend=dict(font=dict(size=20))
        )
        
        if st.checkbox("Render as Static Image (Right-Click -> Copy)", key="static_jsc"):
            try:
                # Enforce white background for PPTX
                fig_jsc.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="black")
                )
                img_bytes = fig_jsc.to_image(format="png", scale=2, width=800, height=600)
                st.image(img_bytes, caption="High-Res Static Image", use_column_width=False, width=800)
            except Exception as e:
                st.error(f"Static image generation failed: {e}. Please restart the app.")
                st.plotly_chart(fig_jsc, config=config)
        else:
            st.plotly_chart(fig_jsc, config=config)

    with tab3:
        st.subheader("FF vs Light Intensity")
        
        fig_ff = go.Figure()
        
        for i, (name, df) in enumerate(data_results.items()):
            color = colors[i % len(colors)]
            df_valid = df.dropna(subset=['FF'])
            
            fig_ff.add_trace(go.Scatter(
                x=df_valid['LightIntensity'],
                y=df_valid['FF'],
                mode='lines+markers',
                name=name,
                marker=dict(color=color, size=12),
                line=dict(width=3)
            ))
            
        fig_ff.update_layout(
            xaxis_type="log",
            xaxis_title="Light Power (mW/cm²)",
            yaxis_title="Fill Factor",
            height=600,
            width=800,
            yaxis_range=[0, 1],
            font=dict(size=18),
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            legend=dict(font=dict(size=20))
        )
        
        if st.checkbox("Render as Static Image (Right-Click -> Copy)", key="static_ff"):
            try:
                # Enforce white background for PPTX
                fig_ff.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="black")
                )
                img_bytes = fig_ff.to_image(format="png", scale=2, width=800, height=600)
                st.image(img_bytes, caption="High-Res Static Image", use_column_width=False, width=800)
            except Exception as e:
                st.error(f"Static image generation failed: {e}. Please restart the app.")
                st.plotly_chart(fig_ff, config=config)
        else:
            st.plotly_chart(fig_ff, config=config)

else:
    st.info("Please load data to begin analysis.")
