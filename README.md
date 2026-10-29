# Building Occupancy Prediction via Indoor Environmental Telemetry

**Unit:** MMA3001 – Numerical Methods and Machine Learning  
**Student:** [Your Name]  
**Dataset:** Dataset 2 – Monash Smart Infrastructure (Living Lab)

---

## Project Overview

Heating, ventilation, and air-conditioning (HVAC) systems are typically operated on rigid, static time schedules. This wastes significant energy conditioning completely empty lecture theatres, breakout rooms, and offices.

This project develops an automated computational pipeline that processes indoor air quality data (carbon dioxide, temperature, relative humidity, and volatile organic compounds) to estimate whether a room is occupied (`1 = Occupied`, `0 = Vacant`). This provides an edge-deployable signal for dynamic, demand-controlled ventilation.

---

## Repository Structure

```text
mma3001-building-occupancy-prediction/
├── data/
│   ├── raw/             # Raw sensor logs and metadata (git-ignored for size)
│   └── processed/       # Resampled 5-minute master dataset (master_dataset_5m.csv)
├── docs/                # Auto-generated HTML documentation from pdoc
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb   # Main notebook with all analysis
├── src/                 # Reusable Python source modules
│   ├── __init__.py
│   ├── preprocessing.py # JSON unpacking, deduplication, and interpolation
│   └── model.py         # Machine learning training pipeline
├── tests/               # Automated unit tests and test reports
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_report.xml  # PyTest JUnit XML report
├── .gitignore           # Prevents uploading large raw data files
├── LICENSE              # Open-source license agreement
├── README.md            # Project orientation and setup guide
└── requirements.txt     # Python package dependencies
