import pandas as pd
import numpy as np

def load_dat_file(filepath):
    """
    Loads a .dat file into a pandas DataFrame.
    Expects two columns: Voltage (V) and Current (I).
    """
    try:
        # Read with whitespace delimiter, skipping the header row
        data = pd.read_csv(filepath, delim_whitespace=True, names=('V', 'I'), skiprows=1)
        
        # Ensure numeric
        data['V'] = pd.to_numeric(data['V'], errors='coerce')
        data['I'] = pd.to_numeric(data['I'], errors='coerce')
        data = data.dropna()
        
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def check_short_circuit(data):
    """
    Checks if the device appears to be shorted or open based on heuristics.
    Returns a status string: 'OK', 'SHORT', 'OPEN', or 'NOISY'
    
    Heuristics from original script (commented out there, but useful here):
    # if (abs(data[data['V'] > 1e-5]['I']).max() > 3e-5) or (abs(data[data['V'] >0 ]['I']).max() < 6e-8):
    #     continue
    # if (abs(data[data['V'] <0 ]['I']).max()> 7e-5):
    #     continue
    """
    if data is None or data.empty:
        return 'EMPTY'

    try:
        # Extract subsets
        pos_bias = data[data['V'] > 1e-5]
        neg_bias = data[data['V'] < -1e-5]
        
        max_pos_current = abs(pos_bias['I']).max() if not pos_bias.empty else 0
        max_neg_current = abs(neg_bias['I']).max() if not neg_bias.empty else 0
        
        # Heuristic thresholds (can be adjusted via settings later if needed)
        # These are based on the commented out lines in IVcurve_V3.py
        
        # High leakage/Short?
        if max_pos_current > 3e-5: 
            return 'HIGH_CURRENT' # Potential Short
            
        if max_neg_current > 7e-5:
            return 'HIGH_LEAKAGE'
            
        # Open circuit / No contact?
        if max_pos_current < 6e-8:
            return 'LOW_CURRENT' # Potential Open
            
        return 'OK'
    except Exception as e:
        return 'ERROR'
