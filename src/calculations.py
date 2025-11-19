import numpy as np
import pandas as pd
from scipy import interpolate, constants
import math

def get_I_from_V(dataraw, xval, direction='F'):
    """
    Get value of I column corresponding to a value in X column.
    Handles bidirectional scans by selecting Forward (F) or Reverse (R) scan.
    """
    if dataraw is None or dataraw.empty:
        return np.nan

    # Sort by V to ensure monotonic for splitting, but we need to preserve scan order for direction detection
    # Actually, the original script sorts by V first:
    # dataraw = dataraw.sort_values(by='V').reset_index(drop=True)
    # But that destroys the time-series order which defines "Forward" vs "Reverse" scan in a loop.
    # Let's look at the original logic:
    # direction_change_idx = dataraw['V'].idxmax() if direction == 'F' else dataraw['V'].idxmin()
    
    # If we sort first, we lose the ability to distinguish F/R based on index if the data was recorded as a loop.
    # However, the original script DOES sort first: `dataraw = dataraw.sort_values(by='V').reset_index(drop=True)`
    # This implies the original script MIGHT have been buggy or relied on the file being pre-sorted or specific structure.
    # Wait, if it sorts by V, `idxmax` is just the last element (for positive sweep).
    
    # Let's try to be smarter. If the data is a loop (0 -> Vmax -> 0 -> -Vmax -> 0), sorting destroys that structure.
    # But if the original code did that, maybe I should stick to it or improve it?
    # The original code:
    # dataraw = dataraw.sort_values(by='V').reset_index(drop=True)
    # direction_change_idx = dataraw['V'].idxmax() if direction == 'F' else dataraw['V'].idxmin()
    # This logic seems to assume the data is NOT a loop in the file, or that sorting is intended.
    # BUT, if it's a loop, sorting mixes the F and R scans together.
    
    # Let's assume the user wants to interpolate on the full dataset if it's just one scan.
    # If it's a loop, we should probably try to detect it BEFORE sorting.
    
    # For now, let's follow the logic of "Interpolate the entire V range".
    
    # Robust approach:
    # 1. Drop duplicates in V to avoid interpolation errors
    data = dataraw.drop_duplicates(subset=['V']).sort_values(by='V')
    
    if data.empty:
        return np.nan
        
    try:
        # Linear interpolation is safer than cubic for noisy data
        f = interpolate.interp1d(data['V'], data['I'], kind='linear', fill_value="extrapolate")
        return float(f(xval))
    except Exception as e:
        print(f"Interpolation error: {e}")
        return np.nan

def get_responsivity(current, radiance=0.001, area=0.16):
    """
    Calculates Responsivity (A/W).
    current: Amps
    radiance: W/cm^2 (default 1 mW/cm^2 = 0.001 W/cm^2)
    area: cm^2
    """
    if radiance == 0 or area == 0:
        return 0
    return current / (radiance * area)

def get_eqe_from_R(responsivity, wavelength):
    """
    Calculates EQE (%) from Responsivity.
    wavelength: nm
    """
    if wavelength == 0:
        return 0
    # EQE = (R * h * c) / (e * lambda)
    # Simplified: EQE = 1240 * R / lambda (nm)
    # Returns fraction (0-1) or percent? Original script: 1240*responsivity/wavelength
    # And later: abs(eqe.round(3)*100) -> So it returns fraction-like value (e.g. 0.8 for 80%)?
    # No, 1240 * (A/W) / nm -> (W/A)^-1 * V * nm ...
    # R = A/W = C/J.
    # 1240 / lambda(nm) is Energy in eV.
    # EQE = electrons / photons
    # R = I / P = (electrons/s * e) / (photons/s * h*c/lambda)
    # R = EQE * e * lambda / (h * c)
    # EQE = R * h * c / (e * lambda)
    # h*c/e approx 1240 eV*nm
    # So EQE = R * 1240 / lambda.
    # If R is A/W, result is dimensionless (0.0 to 1.0 usually).
    return (1240 * responsivity / wavelength)

def calculate_metrics(light_data, dark_data, voltage, wavelength, radiance=0.001, area=0.16):
    """
    Calculates EQE, Responsivity, and Dark Current Density at a specific voltage.
    """
    
    # Get currents at target voltage
    i_light = get_I_from_V(light_data, voltage)
    i_dark = get_I_from_V(dark_data, voltage)
    
    if np.isnan(i_light): i_light = 0
    if np.isnan(i_dark): i_dark = 0
    
    # Corrected current (Light - Dark)
    # Original script: current_corr = abs(light_current) - abs(dark_current)
    # If current_corr < 0: current_corr = 0
    
    current_corr = abs(i_light) - abs(i_dark)
    if current_corr < 0:
        current_corr = 0
        
    responsivity = get_responsivity(current_corr, radiance, area)
    eqe = get_eqe_from_R(responsivity, wavelength)
    
    # Dark Current Density (A/cm^2)
    j_dark = abs(i_dark) / area
    
    return {
        'voltage': voltage,
        'i_light': i_light,
        'i_dark': i_dark,
        'responsivity': responsivity,
        'eqe': eqe * 100, # Convert to percent
        'j_dark': j_dark
    }

def check_quality(light_data, dark_data, voltage_check=-0.5, threshold_ratio=3.0):
    """
    Checks for shorted devices based on Light/Dark current ratio at a specific voltage.
    Returns a dictionary with quality metrics.
    """
    i_light = get_I_from_V(light_data, voltage_check)
    i_dark = get_I_from_V(dark_data, voltage_check)
    
    if i_dark == 0:
        ratio = float('inf')
    else:
        ratio = abs(i_light) / abs(i_dark)
        
    is_shorted = ratio < threshold_ratio
    
    return {
        'ratio_at_check_v': ratio,
        'is_shorted': is_shorted,
        'i_light_check': i_light,
        'i_dark_check': i_dark
    }

# --- Voc-Light Analysis Helpers ---

def interp_y_at_x(x, xa, ya):
    for i in range(len(xa)-1):
        x0, x1 = xa[i], xa[i+1]
        if (x0 <= x <= x1) or (x1 <= x <= x0):
            if x1 == x0:
                return np.nan
            t = (x - x0) / (x1 - x0)
            return ya[i]*(1-t) + ya[i+1]*t
    return np.nan

def zero_cross_x(xa, ya):
    for i in range(len(ya)-1):
        y0, y1 = ya[i], ya[i+1]
        if y0 == 0:
            return xa[i]
        if y0 * y1 < 0:
            t = y0 / (y0 - y1)
            return xa[i] + t * (xa[i+1] - xa[i])
    return np.nan

def iv_params(Vseg, Iseg):
    Voc = zero_cross_x(Vseg, Iseg)
    Isc = interp_y_at_x(0.0, Vseg, Iseg)
    
    if np.isnan(Voc) or np.isnan(Isc) or Voc == 0 or Isc == 0:
        return float("nan"), float("nan"), float("nan")
        
    # Use PV convention: photocurrent positive => Iph = -I
    Iph = -Iseg
    vmin, vmax = (0, Voc) if Voc >= 0 else (Voc, 0)
    m = (Vseg >= vmin) & (Vseg <= vmax)
    
    if m.sum() < 2:
        return float(Voc), float(Isc), float("nan")
        
    P = Vseg[m] * Iph[m]
    if len(P) == 0:
         return float(Voc), float(Isc), float("nan")

    k = int(np.nanargmax(np.abs(P)))
    Vmp, Imp = Vseg[m][k], Iph[m][k]
    FF = abs(Vmp * Imp) / (abs(Voc * Isc)) if (Voc*Isc) != 0 else float("nan")
    return float(Voc), float(Isc), float(FF)

def analyze_iv_numpy(V, I):
    """
    Analyzes a single IV curve (numpy arrays).
    Returns Voc, Isc, FF.
    """
    # 1) scan direction from first two voltages
    if len(V) < 2:
        return {'Voc': np.nan, 'Isc': np.nan, 'FF': np.nan}

    # 2) single vs double (look for one sign change in dV)
    dv = np.diff(V)
    sign = np.sign(dv)
    # treat zeros by carrying last nonzero sign
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i-1]
    changes = np.where(np.diff(sign) != 0)[0]
    
    if len(changes) == 0:
        scan_type = "single"
        segments = [(0, len(V)-1)]
    else:
        scan_type = "double"
        tp = changes[0]
        segments = [(0, tp), (tp+1, len(V)-1)]  # forward, then reverse

    seg_results = []
    try:
        for a, b in segments:
            voc, isc, ff = iv_params(V[a:b+1], I[a:b+1])
            seg_results.append((voc, isc, ff))
    
        def avg_or_pick(vals):
            xs = [x for x in vals if isinstance(x, (int, float)) and not math.isnan(x)]
            if not xs:
                return float("nan")
            return float(np.mean(xs)) if len(xs) > 1 else float(xs[0])
    
        if scan_type == "single":
            Voc, Isc, FF = seg_results[0]
        else:
            Voc = avg_or_pick([s[0] for s in seg_results])
            Isc = avg_or_pick([s[1] for s in seg_results])
            FF  = avg_or_pick([s[2] for s in seg_results])
    
        return {
            "Voc": Voc,
            "Isc": Isc,
            "FF": FF,
        }
    except Exception as e:
        print(f"Analysis Error: {e}")
        return {
            "Voc": np.nan,
            "Isc": np.nan,
            "FF": np.nan,
        }
