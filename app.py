import streamlit as st

st.set_page_config(
    page_title="Perovskite Lab Tools",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Perovskite Photodetector Analysis Suite")

st.markdown("""
Welcome to the **Perovskite Lab Analysis Suite**. This application bundles multiple tools for characterizing and analyzing perovskite photodetectors.

### 👈 Select a Tool from the Sidebar

#### 1. [IV Curves](IV_Curves)
The main tool for processing and visualizing IV characteristics.
- **Features**:
    - Drag-and-drop or Folder loading.
    - Automatic metadata parsing.
    - EQE, Responsivity, and Dark Current calculations.
    - Quality Control filters (Shorts, Bad Contacts).
    - Interactive Plotly graphs.

#### 2. [Voc Analysis](Voc_Analysis)
*(Coming Soon)*
- Analyze $V_{oc}$ dependence on Light Intensity.
- Extract Ideality Factor ($n$).

---
**Version**: 1.0.0
**Maintainer**: Lab Automation Team
""")
