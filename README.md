# IV Curve Plotter for Perovskite Photodetectors

A Streamlit-based application for analyzing and plotting IV curves, specifically designed for perovskite photodetector characterization.

## Features

- **Flexible Data Loading**: 
    - Drag-and-drop `.dat` files.
    - Recursive folder search to load entire datasets.
    - Automatic parsing of metadata (Batch, Device, Pixel, Color, Condition) from filenames.
- **Interactive Visualization**:
    - Dynamic IV curve plots using Plotly.
    - Linear and Logarithmic scales.
    - Grouping by device/batch with customizable legends.
- **Quality Control**:
    - Filter out shorted devices (Light/Dark current ratio check).
    - Filter out bad contacts (Low EQE check).
- **Analysis**:
    - Automatic calculation of Responsivity (A/W), EQE (%), and Dark Current Density (A/cm²).
    - Box plot distributions for statistical analysis.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

## File Naming Convention

The app expects filenames in the following format:
`{Batch}{Device}_{Pixel}{Color}{Condition}.dat`

Example: `BDAF1_3BDARK.dat`
- Batch: `BDAF`
- Device: `1`
- Pixel: `3`
- Color: `B` (Blue)
- Condition: `DARK`

## Dependencies

- streamlit
- pandas
- plotly
- scipy
- numpy
