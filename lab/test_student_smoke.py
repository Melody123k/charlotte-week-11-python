
"""Week 11 Class 4 — Feature Engineering from LAS and SCADA."""

from pathlib import Path

import numpy as np
import pandas as pd
import lasio


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

SCADA_FILE = DATA_DIR / "scada_synthetic.csv"
LAS_FILE = DATA_DIR / "well_b_synthetic.las"

JOIN_KEYS = ["well_id", "depth_m"]


def find_column(df, candidates, description):
    """Find a column using accepted names."""

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Missing {description}. Available columns: {list(df.columns)}"
    )


def load_scada(path: Path) -> pd.DataFrame:
    """Load SCADA data and preserve missing sensor readings."""

    df = pd.read_csv(path)

    required_names = {
        "well_id": ["well_id"],
        "depth_m": ["depth_m"],
        "pressure_psi": ["pressure_psi"],
        "temperature_c": ["temperature_c"],
        "flow_m3_h": ["flow_m3_h"],
        "operating_mode": ["operating_mode"],
    }

    rename = {}

    for standard_name, candidates in required_names.items():
        original_name = find_column(
            df, candidates, standard_name
        )
        rename[original_name] = standard_name

    df = df.rename(columns=rename)

    df["well_id"] = df["well_id"].astype("string").str.strip()

    for column in [
        "depth_m",
        "pressure_psi",
        "temperature_c",
        "flow_m3_h",
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df["well_id"].isna().any() or df["well_id"].eq("").any():
        raise ValueError("Missing well IDs in SCADA data.")

    if df["depth_m"].isna().any():
        raise ValueError("Invalid or missing SCADA depths.")

    if df.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate SCADA well/depth keys found.")

    return df.sort_values(JOIN_KEYS).reset_index(drop=True)


def load_las(path: Path) -> pd.DataFrame:
    """Load LAS curves and standardise the required columns."""

    las = lasio.read(str(path))

    curve_names = [
        curve.mnemonic.strip()
        for curve in las.curves
    ]

    print("LAS curve names:", curve_names)

    lookup = {
        name.upper(): name
        for name in curve_names
    }

    depth_name = next(
        (lookup[name] for name in ["DEPT", "DEPTH", "MD"]
         if name in lookup),
        None,
    )

    gamma_name = next(
        (lookup[name] for name in ["GR", "GAMMA", "GAMMA_RAY"]
         if name in lookup),
        None,
    )

    density_name = next(
        (lookup[name] for name in ["RHOB", "RHOZ", "DENSITY"]
         if name in lookup),
        None,
    )

    if not all([depth_name, gamma_name, density_name]):
        raise ValueError(
            "Required LAS curves not found. "
            f"Available curves: {curve_names}"
        )

    df = las.df().reset_index()

    if depth_name not in df.columns:
        df = df.rename(columns={df.columns[0]: depth_name})

    df = df.rename(columns={
        depth_name: "depth_m",
        gamma_name: "gamma_api",
        density_name: "density_g_cm3",
    })

    try:
        well_id = las.well.WELL.value
    except (AttributeError, KeyError):
        well_id = None

    if well_id is None or not str(well_id).strip():
        raise ValueError("Could not find the WELL ID in LAS header.")

    df["well_id"] = str(well_id).strip()

    for column in [
        "depth_m",
        "gamma_api",
        "density_g_cm3",
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Convert the LAS null marker to missing values.
    try:
        null_value = float(las.well.NULL.value)

        for column in ["gamma_api", "density_g_cm3"]:
            df[column] = df[column].replace(
                null_value, np.nan
            )
    except (AttributeError, TypeError, ValueError):
        pass

    if df["depth_m"].isna().any():
        raise ValueError("LAS contains invalid depth values.")

    if df.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate LAS well/depth keys found.")

    return df[
        ["well_id", "depth_m", "gamma_api", "density_g_cm3"]
    ].sort_values(JOIN_KEYS).reset_index(drop=True)


def merge_inputs(las_df, scada_df):
    """Join the two datasets and validate merge coverage."""

    for name, frame in [("LAS", las_df), ("SCADA", scada_df)]:
        if frame.duplicated(JOIN_KEYS).any():
            raise ValueError(f"Duplicate keys found in {name}.")

    merged = pd.merge(
        las_df,
        scada_df,
        on=JOIN_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    print("\nMerge coverage:")
    print(merged["_merge"].value_counts())

    merged["merge_status"] = merged["_merge"].astype(str)
    merged = merged.drop(columns="_merge")

    return merged.sort_values(JOIN_KEYS).reset_index(drop=True)


def engineer_features(df):
    """Create the seven required features per well."""

    df = df.copy()
    df = df.sort_values(JOIN_KEYS).reset_index(drop=True)

    grouped = df.groupby("well_id", sort=False)

    # 1. Missing pressure flag.
    df["pressure_missing_flag"] = (
        df["pressure_psi"].isna().astype(int)
    )

    # 2. Missing flow flag.
    df["flow_missing_flag"] = (
        df["flow_m3_h"].isna().astype(int)
    )

    # 3. Pressure change from previous sample.
    df["pressure_delta_psi"] = grouped[
        "pressure_psi"
    ].diff()

    # 4. Trailing three-reading population variance.
    df["pressure_var_3"] = grouped[
        "pressure_psi"
    ].transform(
        lambda values: values.rolling(
            window=3,
            min_periods=3
        ).var(ddof=0)
    )

    # 5. Temperature change from previous sample.
    df["temperature_delta_c"] = grouped[
        "temperature_c"
    ].diff()

    # 6. Cumulative recorded flow-rate readings.
    df["cumulative_sampled_flow"] = grouped[
        "flow_m3_h"
    ].cumsum()

    # 7. Gamma-to-density ratio.
    df["gamma_density_ratio"] = (
        df["gamma_api"] /
        df["density_g_cm3"].replace(0, np.nan)
    )

    df["gamma_density_ratio"] = df[
        "gamma_density_ratio"
    ].replace([np.inf, -np.inf], np.nan)

    return df


def create_quality_summary(
    las_df,
    scada_df,
    merged_df,
    engineered_df,
):
    """Create row-count and missingness summary."""

    metrics = {
        "las_input_rows": len(las_df),
        "scada_input_rows": len(scada_df),
        "merged_rows": len(merged_df),
        "engineered_rows": len(engineered_df),
        "matched_rows": int(
            (merged_df["merge_status"] == "both").sum()
        ),
        "las_only_rows": int(
            (merged_df["merge_status"] == "left_only").sum()
        ),
        "scada_only_rows": int(
            (merged_df["merge_status"] == "right_only").sum()
        ),
    }

    for column in [
        "pressure_psi",
        "flow_m3_h",
        "temperature_c",
        "gamma_api",
        "density_g_cm3",
    ]:
        metrics[f"missing_{column}"] = int(
            engineered_df[column].isna().sum()
        )

    return pd.DataFrame(
        list(metrics.items()),
        columns=["metric", "value"],
    )


def main():
    """Run the pipeline and save the two output CSV files."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SCADA data...")
    scada_df = load_scada(SCADA_FILE)

    print("Loading LAS data...")
    las_df = load_las(LAS_FILE)

    print("Merging datasets...")
    merged_df = merge_inputs(las_df, scada_df)

    unmatched = merged_df[
        merged_df["merge_status"] != "both"
    ]

    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} unmatched rows found. "
            "Check well IDs and depth values."
        )

    print("Creating engineered features...")
    engineered_df = engineer_features(merged_df)

    quality_summary = create_quality_summary(
        las_df,
        scada_df,
        merged_df,
        engineered_df,
    )

    engineered_df.to_csv(
        OUTPUT_DIR / "engineered_features.csv",
        index=False,
    )

    quality_summary.to_csv(
        OUTPUT_DIR / "quality_summary.csv",
        index=False,
    )

    print("\nPipeline completed successfully!")
    print("Engineered rows:", len(engineered_df))
    print("\nQuality summary:")
    print(quality_summary.to_string(index=False))
    print("\nOutput files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
