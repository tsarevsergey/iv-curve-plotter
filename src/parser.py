import re
import pandas as pd
from pathlib import Path

def parse_filename(filename):
    """
    Parses the filename to extract metadata.
    Expected format: BatchDevice_PixelColorCondition.dat
    Example: BDAF1_3BDARK.dat or BDAF1_3BDARK_2.dat
    """
    # Remove extension
    stem = Path(filename).stem
    
    # Check for duplicate suffix (e.g., _2, _3)
    duplicate_match = re.search(r'_(\d+)$', stem)
    if duplicate_match:
        run_number = int(duplicate_match.group(1))
        base_stem = stem[:duplicate_match.start()]
    else:
        run_number = 1
        base_stem = stem

    # Split by underscore to separate BatchDevice and PixelColorCondition
    # Example: BDAF1_3BDARK -> BDAF1, 3BDARK
    parts = base_stem.split('_')
    
    if len(parts) < 2:
        return None # Invalid format

    batch_device = parts[0]
    pixel_color_condition = parts[1]

    # Extract Pixel (digits), Color (R/G/B), Condition (LIGHT/DARK)
    # Regex: Start with digits (Pixel), then one char (Color), then remaining (Condition)
    # Or: Pixel (digits), then remaining.
    
    # Heuristic from existing code:
    # pixel_number: digits
    # color: R/G/B
    # condition: LIGHT/DARK
    
    pixel_match = re.match(r'^(\d+)', pixel_color_condition)
    if not pixel_match:
        return None
        
    pixel = int(pixel_match.group(1))
    rest = pixel_color_condition[len(pixel_match.group(0)):]
    
    color = None
    if rest and rest[0] in ['R', 'G', 'B']:
        color = rest[0]
        condition = rest[1:]
    else:
        condition = rest
        
    # Extract Batch and Device Number
    # BDAF1 -> Batch: BDAF, Device: 1
    device_match = re.search(r'(\d+)$', batch_device)
    if device_match:
        device_num = int(device_match.group(1))
        batch = batch_device[:device_match.start()]
    else:
        device_num = 0
        batch = batch_device

    return {
        'filename': filename,
        'batch': batch,
        'device': device_num,
        'pixel': pixel,
        'color': color,
        'condition': condition,
        'run': run_number,
        'id': f"{batch}{device_num}_{pixel}{color}{condition}" # Unique ID for the measurement type
    }

def handle_duplicates(file_list, strategy='Keep Latest'):
    """
    Filters the list of parsed file dictionaries based on the duplicate strategy.
    strategy: 'Keep Original', 'Keep Latest', 'Keep All'
    """
    if strategy == 'Keep All':
        return file_list

    # Group by unique measurement ID (Batch+Device+Pixel+Color+Condition)
    grouped = {}
    for f in file_list:
        mid = f['id']
        if mid not in grouped:
            grouped[mid] = []
        grouped[mid].append(f)

    filtered_files = []
    for mid, files in grouped.items():
        if strategy == 'Keep Latest':
            # Sort by run number descending and take the first
            files.sort(key=lambda x: x['run'], reverse=True)
            filtered_files.append(files[0])
        elif strategy == 'Keep Original':
            # Sort by run number ascending and take the first (usually run 1)
            files.sort(key=lambda x: x['run'])
            filtered_files.append(files[0])
            
    return filtered_files
