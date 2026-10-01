# ⚗️ Ethanol AI

Ethanol AI is a tool created within a research program called scientific initiation by researchers from UFSCar and DTU with funding from FAPESP. It is particularly useful for studying the behavior of different second-generation ethanol production processes when subjected to various operating conditions. This software implements hybrid machine learning models, previously trained using knowledge generated from previous research works at UFSCar and abroad, to predict the outcomes. Here, you can test different combinations of initial conditions, essentially finding the maximum possible yield for each situation.

## Run the App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ethanol-ai.streamlit.app/)

## App Instructions

Instructions for the application will be available here.

## Repository Structure

```
├── streamlit_app.py      # App entrypoint (deployed on Streamlit Community Cloud)
├── requirements.txt      # App dependencies (pinned versions, Python 3.12)
├── assets/style.css      # App styling (theme colors and fonts are in .streamlit/config.toml)
├── models/               # Trained ANNs loaded by the app
├── data/                 # Synthetic LHS datasets (used by the app to fit the scalers)
├── docs/references/      # Reference articles
└── research/             # Research notebooks (not used by the app)
    ├── 01_pretreatment/          # Hydrothermal kinetic model and LHS data generation
    ├── 02_enzymatic_hydrolysis/  # Hydrolysis models, experimental data and LHS
    ├── 03_fermentation/          # Fermentation model (Python + MATLAB reference)
    ├── 04_genetic_ann_search/    # Genetic algorithm search for ANN architectures
    ├── 05_back_optimization/     # Inverse optimization with the champion ANN
    └── legacy/                   # Early models (SVR, RF, first ANNs)
```

Notebooks under `research/` use paths relative to their own folder, so run them with the notebook's folder as the working directory.

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
