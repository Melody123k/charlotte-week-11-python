# Week 11 Class 4 — Feature Engineering

## 1. Project Overview
This project combines synthetic LAS well-log data and SCADA sensor data to create useful features for data analysis using Python, Pandas, NumPy and lasio.

## 2. Data Sources and Limitations
The project uses two synthetic datasets:
- `well_b_synthetic.las`
- `scada_synthetic.csv`

The assumed units are depth (m), pressure (psi), flow (m³/h), temperature (°C), gamma ray (API) and density (g/cm³).

The datasets are for learning purposes and do not represent real oilfield measurements.

## 3. How to Run the Project

Install the required libraries:

`pip install -r requirements.txt`

Run the pipeline:

`python feature_pipeline.py`

Run the tests:

`python -m unittest discover -v`

## 4. Feature Definitions
The project creates seven features:

- `pressure_missing_flag`: identifies missing pressure readings.
- `flow_missing_flag`: identifies missing flow readings.
- `pressure_delta_psi`: calculates pressure changes between samples.
- `pressure_var_3`: calculates rolling pressure variance over three samples.
- `temperature_delta_c`: calculates temperature changes between samples.
- `cumulative_sampled_flow`: calculates the running sum of sampled flow readings.
- `gamma_density_ratio`: calculates the ratio of gamma ray to density.

Features are calculated per well and in depth order. Missing sensor readings are preserved rather than automatically filled.

## 5. Validation and Outputs
The pipeline checks duplicate records, missing sensor readings, data types and LAS-SCADA merge coverage.

It produces:
- `outputs/engineered_features.csv`
- `outputs/quality_summary.csv`

Automated tests are used to check the feature calculations and data validation.

## 6. AWS Connection
CSV files are simple to store and share. Parquet is compressed and column-oriented, making it more efficient for analysing large datasets stored in Amazon S3.

## 7. Conclusion
This project demonstrates how well-log and SCADA data can be combined, validated and transformed into useful features for further analysis and future machine learning applications.