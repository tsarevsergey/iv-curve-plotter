# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:05:03 2025

@author: freed
"""
# https://www.fluxim.com/measurement-techniques-perovskite-solar-cells#Open-circuit%20voltage%20versus%20light%20intensity
import numpy as np
import pandas as pd
from pathlib import Path
import math

from scipy import stats
from scipy import constants
import matplotlib.pyplot as plt

def analyze_iv_file(filepath: str) -> dict:
    """
    Columns expected: 'Voltage,V' and 'Current,A' (any whitespace-separated header is OK).
    Returns a dict with scan meta + Voc, Isc, FF.
    """
    df = pd.read_csv(filepath, sep=r"\s+", engine="python")
    # Normalize column names
    cols = {c.strip().lower(): c for c in df.columns}
    vcol = cols.get("voltage,v", df.columns[0])
    icol = cols.get("current,a", df.columns[1])
    V = df[vcol].to_numpy(float)
    I = df[icol].to_numpy(float)
    # 3) illumination from filename (default to LIGHT if absent)
    fname = Path(filepath).name
    u = fname.upper()
    illumination = "LIGHT" if ("LIGHT" in u or "DARK" not in u) else "DARK"

    # 1) scan direction from first two voltages
    scan_direction = "neg_to_pos" if V[1] > V[0] else "pos_to_neg"

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


    # ---- helpers ----
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
        k = int(np.nanargmax(np.abs(P)))
        Vmp, Imp = Vseg[m][k], Iph[m][k]
        FF = abs(Vmp * Imp) / (abs(Voc * Isc)) if (Voc*Isc) != 0 else float("nan")
        return float(Voc), float(Isc), float(FF)

    seg_results = []
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
        "filename": fname,
        "scan_direction": scan_direction,  # "neg_to_pos" or "pos_to_neg"
        "scan_type": scan_type,            # "single" or "double"
        "illumination": illumination,      # "DARK" or "LIGHT"
        "Voc": Voc,
        "Isc": Isc,
        "FF": FF,
    }

# --- examples on the attached files ---
print(analyze_iv_file(r"C:/Users/freed/polybox/Shared/PD data/2025/sep/STCR4/STCR4/STCR4P12MW_1BDARK.dat"))
print(analyze_iv_file(r"C:/Users/freed/polybox/Shared/PD data/2025/sep/STCR4/STCR4/STCR4P12MW_1BLIGHT.dat"))


#%%

    
    
def analyze_iv_numpy(vcol, icol) -> dict:
    """
    accepts separate columns with votlage and current in pandas dataframe format
    Columns expected: 'Voltage,V' and 'Current,A' (any whitespace-separated header is OK).
    Returns a dict with scan meta + Voc, Isc, FF.
    """
    V = vcol.to_numpy(float)
    I = icol.to_numpy(float)
    # 1) scan direction from first two voltages
    scan_direction = "neg_to_pos" if V[1] > V[0] else "pos_to_neg"

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


    # ---- helpers ----
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
        print(Iseg)
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
        k = int(np.nanargmax(np.abs(P)))
        Vmp, Imp = Vseg[m][k], Iph[m][k]
        FF = abs(Vmp * Imp) / (abs(Voc * Isc)) if (Voc*Isc) != 0 else float("nan")
        return float(Voc), float(Isc), float(FF)

    seg_results = []
    try:
        for a, b in segments:
            print(V[a:b+1])
            print(I[a:b+1])
            voc, isc, ff = iv_params(V[a:b+1], I[a:b+1])
            seg_results.append((voc, isc, ff))
    
        def avg_or_pick(vals):
            xs = [x for x in vals if isinstance(x, (int, float)) and not math.isnan(x)]
            if not xs:
                return float("nan")
            return float(np.mean(xs)) if len(xs) > 1 else float(xs[0])
            # return float(xs[1]) if len(xs) > 1 else float(xs[0])
    
        if scan_type == "single":
            Voc, Isc, FF = seg_results[0]
        else:
            Voc = avg_or_pick([s[0] for s in seg_results])
            Isc = avg_or_pick([s[1] for s in seg_results])
            FF  = avg_or_pick([s[2] for s in seg_results])
    
        return {
            "scan_direction": scan_direction,  # "neg_to_pos" or "pos_to_neg"
            "scan_type": scan_type,            # "single" or "double"    # "DARK" or "LIGHT"
            "Voc": Voc,
            "Isc": Isc,
            "FF": FF,
        }
    except:
        print('cant process the file')
        return {
            "scan_direction": np.nan,  # "neg_to_pos" or "pos_to_neg"
            "scan_type": np.nan,            # "single" or "double"    # "DARK" or "LIGHT"
            "Voc": np.nan,
            "Isc": np.nan,
            "FF": np.nan,
        }

def parametersvsLight(path, LI, listart = 3):
    
    iv_df = pd.read_csv(path, sep=r"\s+", engine="python", header=None, skiprows=1)
    V = iv_df.iloc[:, 0].to_numpy(float)
    I_mat = iv_df.iloc[:, 1:].to_numpy(float)
    
 
    
    # ---- Align lengths ---
    assert len(LI) == (len(iv_df.columns)-1)
    results = {}
    vcol = iv_df[0]
    for i in range(listart,(len(LI)+1)):    
        icol =iv_df[i]
        results[i] = analyze_iv_numpy(vcol, icol)
    voltages = [val['Voc']  for key,val in results.items()]
    currents = [np.abs(val['Isc'])  for key,val in results.items()]
    fill_factors  = [np.abs(val['FF'])  for key,val in results.items()]
    return voltages, currents, fill_factors

def plot_LI(x,y, label, color, fit = True):
    x =np.log(x)
    # y= np.log(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[2:8], y[2:8])                                                                                  
    #The resulting slope value (m)= nkT/q, therefore n= m*q/kT.
    plt.plot(x, y, 'o' , color = color)  
    if fit:
        plt.plot(x, intercept + slope*x, label = label +', n= ' + str((slope/ktq).round(2)), color = color)
    else:
        plt.plot(x, intercept + slope*x, label = label , color = color)
    
    
    

#%%


#paths = list of paths with datafiles from Voc-Light
paths = [r"C:/Users/freed/polybox/Shared/PD data/2025/nov/STBGP2/STBGP2/LI/STGB153RD.dat",
           r"C:/Users/freed/polybox/Shared/PD data/2025/nov/STBGP2/STBGP2/LI/STGB16P6.dat",
          # r"C:/Users/freed/OneDrive - ETH Zurich/Dinh  Bao Dan's files - Dan's data_ETHz/BDAE-BDABplusstability/BDAB/BDAD 100 24h/BDAB8P2.dat"
         ]
#corresponding light intensity arrays
LI = np.array([0, 0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 2, 2.4], dtype=float)
T = 300
q = constants.elementary_charge
k = constants.Boltzmann
ktq = k*T/q
  #this is the styling that I use (from seaborn package)
plt.style.use('seaborn-ticks')
x_scale = LI[2:]

#figure size. Change it for bigger picture size (x,y)
plt.figure(figsize= (7,7))

voltages, currents, fill_factors = parametersvsLight(paths[0], LI)
plot_LI(x_scale,voltages,label = 'MA', color ='blue')

voltages, currents, fill_factors = parametersvsLight(paths[1], LI)
plot_LI(x_scale,voltages,label = 'FA', color ='red')
# voltages, currents, fill_factors = parametersvsLight(paths[2], LI)

# plot_LI(x_scale,voltages,label = 'TPPO', color ='orange')
 
plt.legend(fontsize = 26)
plt.ylim(0.2,1.4)
plt.ylabel('Voc, V', fontsize = 26)
plt.xlabel(r'Light power, mW cm$^{-2}$', fontsize = 26)
plt.xticks(np.log(x_scale[:-1:2]), x_scale[:-1:2], fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(fontsize = 26)
plt.show()  
#%%

#FF vs LI

x_scale = LI[2:]

#figure size. Change it for bigger picture size (x,y)
plt.figure(figsize= (7,7))

voltages, currents, fill_factors = parametersvsLight(paths[0], LI)
plot_LI(x_scale,fill_factors,label = 'MA', color ='blue', fit = False)

voltages, currents, fill_factors = parametersvsLight(paths[1], LI)
# plot_LI(x_scale,fill_factors,label = 'NiO new', color ='red',fit = False)
# voltages, currents, fill_factors= parametersvsLight(paths[2], LI)

plot_LI(x_scale,fill_factors,label = 'FA', color ='orange',fit = False)
 
plt.legend(fontsize = 26)
plt.ylim(0,1)
plt.ylabel('FF, %', fontsize = 26)
plt.xlabel(r'Light power, mW cm$^{-2}$', fontsize = 26)
plt.xticks(np.log(x_scale[:-1:2]), x_scale[:-1:2], fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(fontsize = 26)
plt.show()  
#%%

#FF vs LI

x_scale = LI[2:]

#figure size. Change it for bigger picture size (x,y)
plt.figure(figsize= (7,7))

voltages, currents, fill_factors = parametersvsLight(paths[0], LI)
# plot_LI(x_scale,fill_factors,label = 'MA', color ='blue', fit = False)

# voltages, currents, fill_factors = parametersvsLight(paths[1], LI)
# plot_LI(x_scale,fill_factors,label = 'NiO new', color ='red',fit = False)
# voltages, currents, fill_factors= parametersvsLight(paths[2], LI)

plot_LI(np.exp(x_scale),currents,label = 'FA', color ='orange')
 
plt.legend(fontsize = 26)
plt.ylim(0,10e-5)
plt.ylabel('FF, %', fontsize = 26)
plt.xlabel(r'Light power, mW cm$^{-2}$', fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(fontsize = 26)
plt.show()  


