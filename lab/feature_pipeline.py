
"""Week 11 Class 4 — Feature Engineering from LAS and SCADA."""

from pathlib import Path

import numpy as np
import pandas as pd
import lasio


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

SCADA_FILE = DATA_DIR / "scada_synthetic.csv"
LAS_FILE = DATA_DIR / "well_b_synthetic.las"


def find_column(df, candidates, description):
    """Find a column without depending on letter case."""

    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def load_scada(path: Path) -> pd.DataFrame:
    """Load SCADA data, validate columns and preserve missing values."""

    df = pd.read_csv(path)

    well_col = find_column(
        df, ["well_id"], "well ID"
    )

    depth_col = find_column(
        df, ["depth_m"], "depth"
    )

    pressure_col = find_column(
        df, ["pressure_psi"], "pressure"
    )

    temperature_col = find_column(
        df, ["temperature_c"], "temperature"
    )

    flow_col = find_column(
        df, ["flow_m3_h"], "flow"
    )

    mode_col = find_column(
        df, ["operating_mode"], "operating mode"
    )

    df = df.rename(columns={
        well_col: "well_id",
        depth_col: "depth_m",
        pressure_col: "pressure_psi",
        temperature_col: "temperature_c",
        flow_col: "flow_m3_h",
        mode_col: "operating_mode",
    })

    df["well_id"] = df["well_id"].astype("string").str.strip()

    numeric_columns = [
        "depth_m",
        "pressure_psi",
        "temperature_c",
        "flow_m3_h",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df["well_id"].isna().any() or df["well_id"].eq("").any():
        raise ValueError("SCADA contains missing well IDs.")

    if df["depth_m"].isna().any():
        raise ValueError("SCADA contains invalid or missing depths.")

    if df.duplicated(["well_id", "depth_m"]).any():
        raise ValueError("SCADA contains duplicate well/depth keys.")

    return df.sort_values(
        ["well_id", "depth_m"]
    ).reset_index(drop=True)


def load_las(path: Path) -> pd.DataFrame:
    """Read LAS file and return standardised well-log columns."""

    las = lasio.read(str(path))

    curve_names = [
        curve.mnemonic.strip()
        for curve in las.curves
    ]

    print("\nLAS curve names:")
    print(curve_names)

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
        df = df.rename(columns={
            df.columns[0]: depth_name
        })

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
        raise ValueError(
            "Could not read well ID from LAS WELL header."
        )

    df["well_id"] = str(well_id).strip()

    for column in [
        "depth_m",
        "gamma_api",
        "density_g_cm3",
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Replace the LAS NULL value with NaN.
    try:
        null_value = float(las.well.NULL.value)

        for column in ["gamma_api", "density_g_cm3"]:
            df[column] = df[column].replace(
                null_value, np.nan
            )

    except (AttributeError, TypeError, ValueError):
        pass

    if df["depth_m"].isna().any():
        raise ValueError("LAS contains invalid or missing depths.")

    if df.duplicated(["well_id", "depth_m"]).any():
        raise ValueError("LAS contains duplicate well/depth keys.")

    return df[
        ["well_id", "depth_m", "gamma_api", "density_g_cm3"]
    ].sort_values(
        ["well_id", "depth_m"]
    ).reset_index(drop=True)


def merge_inputs(
    las_df: pd.DataFrame,
    scada_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge LAS and SCADA with one-to-one validation."""

    keys = ["well_id", "depth_m"]

    if las_df.duplicated(keys).any():
        raise ValueError("Duplicate keys found in LAS data.")

    if scada_df.duplicated(keys).any():
        raise ValueError("Duplicate keys found in SCADA data.")

    merged = pd.merge(
        las_df,
        scada_df,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    print("\nMerge coverage:")
    print(merged["_merge"].value_counts())

    merged["merge_status"] = merged["_merge"].astype(str)

    merged = merged.drop(columns=["_merge"])

    return merged.sort_values(
        keys
    ).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the seven required engineered features."""

    df = df.copy()

    df = df.sort_values(
        ["well_id", "depth_m"]
    ).reset_index(drop=True)

    grouped = df.groupby("well_id", sort=False)

    # 1. Pressure missing flag.
    df["pressure_missing_flag"] = (
        df["pressure_psi"].isna().astype(int)
    )

    # 2. Flow missing flag.
    df["flow_missing_flag"] = (
        df["flow_m3_h"].isna().astype(int)
    )

    # 3. Pressure difference from previous depth.
    df["pressure_delta_psi"] = grouped[
        "pressure_psi"
    ].diff()

    # 4. Trailing three-sample population variance.
    df["pressure_var_3"] = grouped[
        "pressure_psi"
    ].transform(
        lambda values: values.rolling(
            window=3,
            min_periods=3
        ).var(ddof=0)
    )

    # 5. Temperature difference from previous depth.
    df["temperature_delta_c"] = grouped[
        "temperature_c"
    ].diff()

    # 6. Running sum of recorded flow rates.
    # This is NOT integrated total volume.
    df["cumulative_sampled_flow"] = grouped[
        "flow_m3_h"
    ].cumsum()

    # 7. Gamma divided by density.
    df["gamma_density_ratio"] = (
        df["gamma_api"] / df["density_g_cm3"].replace(0, np.nan)
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
    """Create row-count and missing-value quality report."""

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

    missing_columns = [
        "pressure_psi",
        "flow_m3_h",
        "temperature_c",
        "gamma_api",
        "density_g_cm3",
    ]

    for column in missing_columns:
        metrics[f"missing_{column}"] = int(
            engineered_df[column].isna().sum()
        )

    return pd.DataFrame(
        list(metrics.items()),
        columns=["metric", "value"],
    )


def main():
    """Run the complete feature engineering pipeline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading SCADA data...")
    scada_df = load_scada(SCADA_FILE)

    print("Loading LAS data...")
    las_df = load_las(LAS_FILE)

    print("Merging datasets...")
    merged_df = merge_inputs(las_df, scada_df)

    # Stop if the two sources do not align completely.
    unmatched = merged_df[
        merged_df["merge_status"] != "both"
    ]

    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} unmatched rows found. "
            "Check well IDs and depth values."
        )

    print("Engineering features...")
    engineered_df = engineer_features(merged_df)

    print("Creating quality summary...")
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

    print(
        f"Number of engineered rows: {len(engineered_df)}"
    )

    print("\nQuality summary:")
    print(quality_summary.to_string(index=False))

    print("\nFiles saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
