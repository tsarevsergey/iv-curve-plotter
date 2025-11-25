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

# Device Area
area = st.sidebar.number_input("Device Area (cm²)", value=0.16, step=0.01)

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

# Initialize session state for paths
if 'loaded_paths' not in st.session_state:
    st.session_state['loaded_paths'] = []

if li_values:
    # Load from Paths
    if load_paths_btn and file_paths_input:
        new_paths = [p.strip() for p in file_paths_input.split('\n') if p.strip()]
        # Add new paths if not already present
        for p in new_paths:
            if p not in st.session_state['loaded_paths']:
                st.session_state['loaded_paths'].append(p)
    
    # Process paths from session state
    for p in st.session_state['loaded_paths']:
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
                
    # Add Clear Button
    if st.session_state['loaded_paths'] or uploaded_files:
        if st.sidebar.button("Clear Loaded Data"):
            st.session_state['loaded_paths'] = []
            st.rerun()

# --- Visualization ---
if data_results:
    st.success(f"Loaded {len(data_results)} files.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Voc vs Light", "Jsc vs Light", "FF vs Light", "Responsivity & LDR"])
    
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

    with tab4:
        st.subheader("Responsivity & LDR Analysis")
        
        st.markdown("""
        **LDR Definition**: $LDR = 20 \\log_{10}(I_{max} / I_{min})$ within a linearity deviation $\\Delta$.
        """)
        
        col_ldr_set, col_ldr_plot = st.columns([1, 3])
        
        with col_ldr_set:
            delta_tol = st.number_input("Tolerable Deviation (Δ)", value=0.03, step=0.01, format="%.3f")
            st.caption("Recommended: 0.01 for strict linear, 0.03 for quasi-linear.")
            
            show_alpha = st.checkbox("Show Linearity Coefficient (α)", value=True, key="chk_show_alpha")
            subtract_dark = st.checkbox("Subtract Dark Current", value=True, key="chk_subtract_dark")
            
        with col_ldr_plot:
            # We need to calculate R and LDR for each file
            
            # 1. Responsivity vs Light Intensity Plot
            fig_resp = go.Figure()
            
            # 2. Alpha vs Light Intensity Plot (if requested)
            fig_alpha = go.Figure() if show_alpha else None
            
            # 3. LDR Visualization (maybe on the Jsc plot or separate?)
            # Let's do a combined Jsc vs P plot with the Linear Fit range highlighted.
            fig_ldr = go.Figure()
            
            for i, (name, df) in enumerate(data_results.items()):
                color = colors[i % len(colors)]
                df_valid = df.dropna(subset=['Isc'])
                
                if df_valid.empty:
                    continue
                
                # Calculate Power (W)
                # LightIntensity is in mW/cm2.
                # Power (W) = Intensity (mW/cm2) * Area (cm2) / 1000
                
                # Identify Dark Current (Isc at LightIntensity == 0)
                # We assume 0 intensity is dark.
                dark_data = df[df['LightIntensity'] == 0]
                if subtract_dark and not dark_data.empty:
                    i_dark = abs(dark_data['Isc']).mean()
                else:
                    i_dark = 0.0
                
                # Use a copy to avoid SettingWithCopy warnings
                df_calc = df_valid.copy()
                
                # Calculate Photocurrent (Iph) = |Isc| - |Idark|
                df_calc['I_raw_abs'] = abs(df_calc['Isc'])
                df_calc['I_ph'] = df_calc['I_raw_abs'] - i_dark
                
                # Filter out non-positive photocurrents for Log analysis (and P=0)
                df_calc = df_calc[(df_calc['I_ph'] > 0) & (df_calc['LightIntensity'] > 0)]
                
                if df_calc.empty:
                    continue

                df_calc['Power_W'] = df_calc['LightIntensity'] * area / 1000.0
                df_calc['Responsivity'] = df_calc['I_ph'] / df_calc['Power_W']
                
                # Plot Responsivity
                fig_resp.add_trace(go.Scatter(
                    x=df_calc['LightIntensity'],
                    y=df_calc['Responsivity'],
                    mode='lines+markers',
                    name=f"{name} (Idark={i_dark:.1e}A)",
                    marker=dict(color=color, size=10),
                    line=dict(width=2)
                ))
                
                # Calculate Alpha (Slope of log-log)
                # Local slope or global? "its slope" usually means local alpha or fit alpha.
                # Let's calculate local alpha between points.
                if show_alpha and len(df_calc) > 1:
                    # Calculate log derivatives using I_ph
                    log_I = np.log10(df_calc['I_ph'].to_numpy())
                    log_P = np.log10(df_calc['LightIntensity'].to_numpy())
                    
                    # Central difference or forward difference
                    # Let's use midpoints for alpha
                    alpha_vals = np.diff(log_I) / np.diff(log_P)
                    # Midpoint P
                    p_mids = 10**((log_P[:-1] + log_P[1:]) / 2)
                    
                    fig_alpha.add_trace(go.Scatter(
                        x=p_mids,
                        y=alpha_vals,
                        mode='lines+markers',
                        name=f"{name} (α)",
                        marker=dict(color=color, symbol='triangle-up', size=10),
                        line=dict(dash='dot')
                    ))
                
                # LDR Calculation
                # Find largest contiguous range [start, end] where linear fit has relative error <= delta_tol
                
                P_arr = df_calc['Power_W'].to_numpy()
                I_arr = df_calc['I_ph'].to_numpy() # Use I_ph
                
                best_ldr = -1
                best_range = None
                best_slope = 0
                
                n_pts = len(P_arr)
                if n_pts >= 3:
                    # Brute force all sub-segments of length >= 3
                    for start in range(n_pts):
                        for end in range(start + 2, n_pts): # at least 3 points
                            sub_P = P_arr[start:end+1]
                            sub_I = I_arr[start:end+1]
                            
                            # Linear fit forced through zero? I = R * P
                            # Since we subtracted dark current, I ~ R * P is correct.
                            
                            # Least squares for I = S * P is S = sum(I*P) / sum(P^2)
                            S = np.sum(sub_I * sub_P) / np.sum(sub_P**2)
                            
                            I_fit = S * sub_P
                            
                            # Relative Deviation
                            # Handle div by zero if I_fit is 0 (unlikely)
                            dev = np.abs((sub_I - I_fit) / I_fit)
                            
                            if np.all(dev <= delta_tol):
                                # Valid range
                                # Calculate LDR
                                I_max = sub_I.max()
                                I_min = sub_I.min()
                                if I_min > 0:
                                    ldr_val = 20 * np.log10(I_max / I_min)
                                    if ldr_val > best_ldr:
                                        best_ldr = ldr_val
                                        best_range = (start, end)
                                        best_slope = S
                
                # Plot LDR on the third plot (Log-Log Iph vs P)
                # We plot the full data and the linear fit on the valid range
                
                # Full data
                fig_ldr.add_trace(go.Scatter(
                    x=df_calc['LightIntensity'],
                    y=df_calc['I_ph'],
                    mode='markers',
                    name=f"{name} Data",
                    marker=dict(color=color, size=8, opacity=0.5)
                ))
                
                if best_range:
                    s, e = best_range
                    # Plot the valid range with solid line
                    valid_P = df_calc['LightIntensity'].iloc[s:e+1]
                    valid_I = abs(df_calc['Isc']).iloc[s:e+1]
                    
                    # Fit line (using best_slope which is A/W vs Power)
                    # We need to plot vs Intensity (mW/cm2)
                    # I = S * Power = S * (Intensity * Area / 1000)
                    # So slope in plot vs Intensity is S * Area / 1000
                    
                    slope_plot = best_slope * area / 1000.0
                    y_fit_plot = slope_plot * valid_P
                    
                    fig_ldr.add_trace(go.Scatter(
                        x=valid_P,
                        y=y_fit_plot,
                        mode='lines',
                        name=f"{name} Fit (LDR={best_ldr:.1f} dB)",
                        line=dict(color=color, width=4)
                    ))
                    
                    # Annotate
                    st.write(f"**{name}**: LDR = **{best_ldr:.2f} dB** (Range: {valid_P.min():.2f} - {valid_P.max():.2f} mW/cm²)")
                else:
                    st.write(f"**{name}**: Could not find linear range with Δ <= {delta_tol}")

            # Render Plots
            st.markdown("### Responsivity (A/W)")
            fig_resp.update_layout(
                xaxis_type="log",
                xaxis_title="Light Power (mW/cm²)",
                yaxis_title="Responsivity (A/W)",
                height=500,
                width=800,
                font=dict(size=18)
            )
            st.plotly_chart(fig_resp, config=config)
            
            if show_alpha:
                st.markdown("### Linearity Coefficient (α)")
                fig_alpha.update_layout(
                    xaxis_type="log",
                    xaxis_title="Light Power (mW/cm²)",
                    yaxis_title="Alpha (Slope)",
                    height=400,
                    width=800,
                    font=dict(size=18)
                )
                # Add reference line at alpha=1
                fig_alpha.add_hline(y=1.0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_alpha, config=config)
                
            st.markdown("### LDR Analysis (Log-Log Isc)")
            fig_ldr.update_layout(
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title="Light Power (mW/cm²)",
                yaxis_title="Isc (A)",
                height=600,
                width=800,
                font=dict(size=18)
            )
            st.plotly_chart(fig_ldr, config=config)

else:
    st.info("Please load data to begin analysis.")
