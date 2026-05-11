"""
Voltage calculator with automatic DirectSun/Umbra extraction and isotonic regression.

What this script does:
1. Takes input voltage and two comms timestamps:
      End time of last ground pass -> Start time of next ground pass
2. Reads Lighting.csv and calculates exact DirectSun and Umbra overlap inside that window.
3. Reads EPS_Capacity_Test_Processed_1.2V.xlsx, sheet V_Wh.
4. Applies isotonic regression to make the battery V vs Energy_Wh curve monotonic.
   In this battery table, Energy_Wh behaves like used/discharged energy:
      DirectSun charging -> Energy_Wh decreases -> voltage increases
      Umbra discharge    -> Energy_Wh increases -> voltage decreases
5. Builds lower/input/upper voltage profiles.
6. Saves plots and CSV outputs.

Required files in same folder:
- EPS_Capacity_Test_Processed_1.2V.xlsx
- Lighting.csv

Required packages:
- pandas
- numpy
- matplotlib
- openpyxl
- scikit-learn

Install if needed:
    pip install pandas numpy matplotlib openpyxl scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.isotonic import IsotonicRegression


# ============================================================
# CONFIG
# ============================================================
CAPACITY_EXCEL_PATH = "EPS_Capacity_Test_Processed_1.2V.xlsx"
LIGHTING_CSV_PATH = "Lighting.csv"

SHEET_NAME = "V_Wh"
VOLTAGE_COL = "TOTAL_BATTERY_VOLTAGE"
ENERGY_COL = "Energy_Wh"

VOLTAGE_BAND_WIDTH = 0.1

VOLTAGE_TIME_PLOT = "voltage_vs_time_isotonic.png"
VOLTAGE_WH_PLOT = "voltage_vs_wh_isotonic.png"
PROFILE_CSV = "voltage_time_profile_isotonic.csv"
ISOTONIC_TABLE_CSV = "isotonic_capacity_table.csv"


# ============================================================
# INPUT HELPERS
# ============================================================
def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Enter a valid number.")


def get_text_input(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty.")


# ============================================================
# TIME HELPERS
# ============================================================
def parse_utc_time(value):
    """Parse UTC timestamp like 2026-05-05T12:28:53.865Z."""
    return pd.to_datetime(value, utc=True, errors="raise")


def overlap_seconds(a_start, a_end, b_start, b_end):
    """Return overlap duration in seconds between interval A and interval B."""
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return max(0.0, (earliest_end - latest_start).total_seconds())


# ============================================================
# LIGHTING CALCULATION
# ============================================================
def calculate_lighting_between_passes(
    lighting_csv_path,
    pass_1_end_time_utc,
    pass_2_start_time_utc,
    start_col="Start Time (UTC)",
    end_col="End Time (UTC)",
    condition_col="Condition",
):
    """
    Intersects the pass gap with Lighting.csv and returns DirectSun/Umbra segments.
    """
    window_start = parse_utc_time(pass_1_end_time_utc)
    window_end = parse_utc_time(pass_2_start_time_utc)

    if window_end <= window_start:
        raise ValueError("Pass 2 start time must be after Pass 1 end time.")

    lighting_df = pd.read_csv(lighting_csv_path)
    lighting_df[start_col] = pd.to_datetime(lighting_df[start_col], utc=True, errors="raise")
    lighting_df[end_col] = pd.to_datetime(lighting_df[end_col], utc=True, errors="raise")
    lighting_df[condition_col] = lighting_df[condition_col].astype(str).str.upper().str.strip()

    rows = []
    totals = {"DIRECTSUN": 0.0, "UMBRA": 0.0}

    for _, row in lighting_df.iterrows():
        condition = row[condition_col]
        seconds = overlap_seconds(window_start, window_end, row[start_col], row[end_col])

        if seconds > 0:
            overlap_start = max(window_start, row[start_col])
            overlap_end = min(window_end, row[end_col])

            rows.append({
                "condition": condition,
                "lighting_start_utc": row[start_col],
                "lighting_end_utc": row[end_col],
                "overlap_start_utc": overlap_start,
                "overlap_end_utc": overlap_end,
                "overlap_seconds": seconds,
            })

            if condition in totals:
                totals[condition] += seconds

    details_df = pd.DataFrame(rows)
    if not details_df.empty:
        details_df = details_df.sort_values("overlap_start_utc").reset_index(drop=True)

    return {
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "gap_seconds": (window_end - window_start).total_seconds(),
        "directsun_seconds": totals["DIRECTSUN"],
        "umbra_seconds": totals["UMBRA"],
        "details": details_df,
    }


# ============================================================
# CAPACITY TABLE + ISOTONIC REGRESSION
# ============================================================
def load_capacity_table(
    file_path,
    sheet_name=SHEET_NAME,
    voltage_col=VOLTAGE_COL,
    energy_col=ENERGY_COL,
):
    """Read raw V vs Wh lookup table from Excel."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df[[voltage_col, energy_col]].dropna().copy()
    df[voltage_col] = df[voltage_col].astype(float)
    df[energy_col] = df[energy_col].astype(float)
    df = df.sort_values(energy_col).reset_index(drop=True)
    return df


def build_isotonic_capacity_table(raw_df):
    """
    Fit isotonic regression on Voltage vs Energy_Wh.

    The Excel Energy_Wh axis behaves like discharged/used energy.
    Therefore, as Energy_Wh increases, voltage should generally decrease.
    So we fit an isotonic curve with increasing=False.
    """
    # Average duplicate energy points so isotonic regression receives clean x values.
    grouped = (
        raw_df.groupby(ENERGY_COL, as_index=False)[VOLTAGE_COL]
        .mean()
        .sort_values(ENERGY_COL)
        .reset_index(drop=True)
    )

    x_energy = grouped[ENERGY_COL].to_numpy(dtype=float)
    y_voltage_raw = grouped[VOLTAGE_COL].to_numpy(dtype=float)

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    y_voltage_iso = iso.fit_transform(x_energy, y_voltage_raw)

    iso_df = pd.DataFrame({
        ENERGY_COL: x_energy,
        "raw_voltage_mean": y_voltage_raw,
        "isotonic_voltage": y_voltage_iso,
    })

    return iso, iso_df


def energy_to_voltage_iso(energy_wh, iso_model):
    """Convert Energy_Wh to voltage using the monotonic isotonic model."""
    arr = np.asarray([energy_wh], dtype=float)
    return float(iso_model.predict(arr)[0])


def voltage_to_energy_iso(voltage, iso_df):
    """
    Convert voltage to Energy_Wh using inverse interpolation of the isotonic curve.

    Since isotonic_voltage is decreasing with Energy_Wh, reverse/sort by voltage
    before interpolation.
    """
    v = iso_df["isotonic_voltage"].to_numpy(dtype=float)
    e = iso_df[ENERGY_COL].to_numpy(dtype=float)

    order = np.argsort(v)
    v_sorted = v[order]
    e_sorted = e[order]

    # Remove duplicate voltage values from flat isotonic sections.
    inv_df = pd.DataFrame({"v": v_sorted, "e": e_sorted})
    inv_df = inv_df.groupby("v", as_index=False)["e"].mean().sort_values("v")

    v_unique = inv_df["v"].to_numpy(dtype=float)
    e_unique = inv_df["e"].to_numpy(dtype=float)

    return float(np.interp(voltage, v_unique, e_unique))


# ============================================================
# VOLTAGE RANGE CALCULATION
# ============================================================
def calculate_voltage_range_isotonic(raw_df, iso_df, iso_model, input_voltage, net_wh):
    """
    Calculates final voltage range using the isotonic V-Wh curve.

    Sign convention:
        positive net_wh -> charging    -> used Energy_Wh decreases -> voltage increases
        negative net_wh -> discharging -> used Energy_Wh increases -> voltage decreases
    """
    v_low = input_voltage
    v_high = input_voltage + VOLTAGE_BAND_WIDTH

    band = raw_df[(raw_df[VOLTAGE_COL] >= v_low) & (raw_df[VOLTAGE_COL] < v_high)].copy()

    if band.empty:
        raise ValueError(f"No rows found for voltage range {v_low:.2f} V to {v_high:.2f} V")

    start_voltage_low = float(band[VOLTAGE_COL].min())
    start_voltage_high = float(band[VOLTAGE_COL].max())
    start_voltage_mid = float(input_voltage)

    # Use isotonic inverse for consistent starting energies.
    start_energy_low = voltage_to_energy_iso(start_voltage_low, iso_df)
    start_energy_high = voltage_to_energy_iso(start_voltage_high, iso_df)
    start_energy_mid = voltage_to_energy_iso(start_voltage_mid, iso_df)

    min_energy = float(iso_df[ENERGY_COL].min())
    max_energy = float(iso_df[ENERGY_COL].max())

    # Positive net_wh means charging, so used/discharged Energy_Wh decreases.
    final_energy_low = float(np.clip(start_energy_low - net_wh, min_energy, max_energy))
    final_energy_high = float(np.clip(start_energy_high - net_wh, min_energy, max_energy))
    final_energy_mid = float(np.clip(start_energy_mid - net_wh, min_energy, max_energy))

    final_voltage_low_raw = energy_to_voltage_iso(final_energy_low, iso_model)
    final_voltage_high_raw = energy_to_voltage_iso(final_energy_high, iso_model)
    final_voltage_mid = energy_to_voltage_iso(final_energy_mid, iso_model)

    final_voltage_low = float(min(final_voltage_low_raw, final_voltage_high_raw))
    final_voltage_high = float(max(final_voltage_low_raw, final_voltage_high_raw))

    return {
        "input_voltage_requested": input_voltage,
        "voltage_band_low": v_low,
        "voltage_band_high": v_high,
        "number_of_matching_rows": int(len(band)),
        "start_voltage_low": start_voltage_low,
        "start_voltage_high": start_voltage_high,
        "start_voltage_mid": start_voltage_mid,
        "start_energy_wh_low": float(min(start_energy_low, start_energy_high)),
        "start_energy_wh_high": float(max(start_energy_low, start_energy_high)),
        "start_energy_wh_mid": start_energy_mid,
        "net_wh": float(net_wh),
        "final_energy_wh_low": float(min(final_energy_low, final_energy_high)),
        "final_energy_wh_high": float(max(final_energy_low, final_energy_high)),
        "final_energy_wh_mid": final_energy_mid,
        "final_voltage_low": final_voltage_low,
        "final_voltage_high": final_voltage_high,
        "final_voltage_mid": final_voltage_mid,
    }


# ============================================================
# PROFILE BUILDING
# ============================================================
def build_voltage_time_profile_isotonic(
    details_df,
    start_voltage,
    iso_df,
    iso_model,
    charge_power_w,
    discharge_power_w,
):
    """
    Build one line/profile from a starting voltage through all lighting segments.
    """
    if details_df.empty:
        raise ValueError("No DIRECTSUN/UMBRA overlap rows found in the selected pass gap.")

    current_energy = voltage_to_energy_iso(start_voltage, iso_df)
    min_energy = float(iso_df[ENERGY_COL].min())
    max_energy = float(iso_df[ENERGY_COL].max())

    rows = []

    for _, row in details_df.iterrows():
        condition = row["condition"]
        t0 = row["overlap_start_utc"]
        t1 = row["overlap_end_utc"]
        dt_seconds = float(row["overlap_seconds"])

        energy_start = current_energy
        voltage_start = energy_to_voltage_iso(energy_start, iso_model)

        if condition == "DIRECTSUN":
            delta_wh_physical = (dt_seconds / 3600.0) * charge_power_w
            current_energy = current_energy - delta_wh_physical
            mode = "charging"
        elif condition == "UMBRA":
            delta_wh_physical = (dt_seconds / 3600.0) * discharge_power_w
            current_energy = current_energy + delta_wh_physical
            mode = "discharging"
        else:
            delta_wh_physical = 0.0
            mode = "unchanged"

        current_energy = float(np.clip(current_energy, min_energy, max_energy))
        energy_end = current_energy
        voltage_end = energy_to_voltage_iso(energy_end, iso_model)

        rows.append({
            "condition": condition,
            "mode": mode,
            "start_time_utc": t0,
            "end_time_utc": t1,
            "duration_seconds": dt_seconds,
            "energy_start_wh": energy_start,
            "energy_end_wh": energy_end,
            "delta_energy_axis_wh": energy_end - energy_start,
            "physical_energy_moved_wh": delta_wh_physical,
            "voltage_start_v": voltage_start,
            "voltage_end_v": voltage_end,
            "delta_voltage_v": voltage_end - voltage_start,
        })

    return pd.DataFrame(rows)


def build_three_profiles(details_df, result, iso_df, iso_model, charge_power_w, discharge_power_w):
    low_profile = build_voltage_time_profile_isotonic(
        details_df, result["start_voltage_low"], iso_df, iso_model, charge_power_w, discharge_power_w
    )
    input_profile = build_voltage_time_profile_isotonic(
        details_df, result["start_voltage_mid"], iso_df, iso_model, charge_power_w, discharge_power_w
    )
    high_profile = build_voltage_time_profile_isotonic(
        details_df, result["start_voltage_high"], iso_df, iso_model, charge_power_w, discharge_power_w
    )

    low_profile["profile"] = "lower_bound"
    input_profile["profile"] = "input_voltage"
    high_profile["profile"] = "upper_bound"

    return pd.concat([low_profile, input_profile, high_profile], ignore_index=True)


# ============================================================
# PLOTTING
# ============================================================
def add_lighting_shading(ax, details_df):
    for _, row in details_df.iterrows():
        if row["condition"] == "DIRECTSUN":
            ax.axvspan(row["overlap_start_utc"], row["overlap_end_utc"], color="orange", alpha=0.15)
        elif row["condition"] == "UMBRA":
            ax.axvspan(row["overlap_start_utc"], row["overlap_end_utc"], color="gray", alpha=0.20)


def clean_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())


def plot_voltage_vs_time(profile_df, details_df, output_path=VOLTAGE_TIME_PLOT):
    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    add_lighting_shading(ax, details_df)

    style_map = {
        "lower_bound": {"color": "blue", "linestyle": "-", "linewidth": 2, "label": "Lower bound"},
        "input_voltage": {"color": "black", "linestyle": "--", "linewidth": 3, "label": "Input voltage"},
        "upper_bound": {"color": "red", "linestyle": "-", "linewidth": 2, "label": "Upper bound"},
    }

    for profile_name, group in profile_df.groupby("profile"):
        style = style_map[profile_name]
        for _, row in group.iterrows():
            ax.plot(
                [row["start_time_utc"], row["end_time_utc"]],
                [row["voltage_start_v"], row["voltage_end_v"]],
                marker="o",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=style["label"],
            )

    ax.set_title("Voltage Change Between Ground Passes - Isotonic Corrected")
    ax.set_xlabel("UTC Time")
    ax.set_ylabel("Battery Voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)
    clean_legend(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_voltage_vs_wh(raw_df, iso_df, profile_df, output_path=VOLTAGE_WH_PLOT):
    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    ax.scatter(
        raw_df[ENERGY_COL],
        raw_df[VOLTAGE_COL],
        s=8,
        alpha=0.35,
        label="Raw Excel points",
    )
    ax.plot(
        iso_df[ENERGY_COL],
        iso_df["isotonic_voltage"],
        linewidth=2.5,
        label="Isotonic V-Wh curve",
    )

    style_map = {
        "lower_bound": {"color": "blue", "linestyle": "-", "linewidth": 2, "label": "Lower path"},
        "input_voltage": {"color": "black", "linestyle": "--", "linewidth": 3, "label": "Input path"},
        "upper_bound": {"color": "red", "linestyle": "-", "linewidth": 2, "label": "Upper path"},
    }

    for profile_name, group in profile_df.groupby("profile"):
        style = style_map[profile_name]
        for _, row in group.iterrows():
            ax.plot(
                [row["energy_start_wh"], row["energy_end_wh"]],
                [row["voltage_start_v"], row["voltage_end_v"]],
                marker="o",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=style["label"],
            )

    input_profile = profile_df[profile_df["profile"] == "input_voltage"].reset_index(drop=True)
    if not input_profile.empty:
        first = input_profile.iloc[0]
        last = input_profile.iloc[-1]
        ax.scatter(first["energy_start_wh"], first["voltage_start_v"], s=90, marker="X", label="Input start")
        ax.scatter(last["energy_end_wh"], last["voltage_end_v"], s=110, marker="*", label="Input final")

    ax.set_title("Voltage vs Energy_Wh Path - Isotonic Corrected")
    ax.set_xlabel("Energy_Wh from Excel table")
    ax.set_ylabel("Battery Voltage (V)")
    ax.grid(True, alpha=0.3)
    clean_legend(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n=== Voltage Calculator with DirectSun/Umbra + Isotonic Regression ===\n")

    input_voltage = get_float_input("Enter start voltage, example 27.7: ")

    print("\nEnter the comms gap timestamps from Possible passes.csv")
    pass_1_end_time = get_text_input("Enter END time of ground pass 1 UTC: ")
    pass_2_start_time = get_text_input("Enter START time of ground pass 2 UTC: ")

    charge_power_w = get_float_input("Enter charging power in W during DIRECTSUN, example 28: ")
    discharge_power_w = get_float_input("Enter discharging power in W during UMBRA, example 12: ")

    lighting_result = calculate_lighting_between_passes(
        lighting_csv_path=LIGHTING_CSV_PATH,
        pass_1_end_time_utc=pass_1_end_time,
        pass_2_start_time_utc=pass_2_start_time,
    )

    details_df = lighting_result["details"]
    if details_df.empty:
        raise ValueError("No DirectSun/Umbra overlap found between the selected pass timestamps.")

    directsun_seconds = lighting_result["directsun_seconds"]
    umbra_seconds = lighting_result["umbra_seconds"]

    charging_wh = (directsun_seconds / 3600.0) * charge_power_w
    discharging_wh = (umbra_seconds / 3600.0) * discharge_power_w
    net_wh = charging_wh - discharging_wh

    raw_capacity_df = load_capacity_table(CAPACITY_EXCEL_PATH)
    iso_model, iso_capacity_df = build_isotonic_capacity_table(raw_capacity_df)
    iso_capacity_df.to_csv(ISOTONIC_TABLE_CSV, index=False)

    result = calculate_voltage_range_isotonic(
        raw_df=raw_capacity_df,
        iso_df=iso_capacity_df,
        iso_model=iso_model,
        input_voltage=input_voltage,
        net_wh=net_wh,
    )

    profile_df = build_three_profiles(
        details_df=details_df,
        result=result,
        iso_df=iso_capacity_df,
        iso_model=iso_model,
        charge_power_w=charge_power_w,
        discharge_power_w=discharge_power_w,
    )

    profile_df.to_csv(PROFILE_CSV, index=False)

    plot_voltage_vs_time(profile_df, details_df, output_path=VOLTAGE_TIME_PLOT)
    plot_voltage_vs_wh(raw_capacity_df, iso_capacity_df, profile_df, output_path=VOLTAGE_WH_PLOT)

    print("\n--- Time Window from Comms Passes ---")
    print(f"Window start, pass 1 end : {lighting_result['window_start_utc']}")
    print(f"Window end, pass 2 start : {lighting_result['window_end_utc']}")
    print(f"Total gap time           : {lighting_result['gap_seconds']:.3f} s")

    print("\n--- Automatic Lighting Intersection ---")
    print(f"Charging time in DIRECTSUN : {directsun_seconds:.3f} s")
    print(f"Discharging time in UMBRA  : {umbra_seconds:.3f} s")
    print("\nDetailed overlap rows:")
    print(details_df.to_string(index=False))

    print("\n--- Energy Calculation ---")
    print(f"Charging Energy           : {charging_wh:.3f} Wh")
    print(f"Discharging Energy        : {discharging_wh:.3f} Wh")
    print(f"Net Energy                : {net_wh:.3f} Wh")

    if net_wh > 0:
        print("Net result                : Power Positive, voltage should increase")
    elif net_wh < 0:
        print("Net result                : Power Negative, voltage should decrease")
    else:
        print("Net result                : Balanced")

    print("\n--- Isotonic Voltage Result Range ---")
    print(f"Requested voltage         : {result['input_voltage_requested']:.3f} V")
    print(f"Voltage band used         : {result['voltage_band_low']:.3f} V to {result['voltage_band_high']:.3f} V")
    print(f"Matching raw rows         : {result['number_of_matching_rows']}")
    print(f"Start voltage range       : {result['start_voltage_low']:.3f} V to {result['start_voltage_high']:.3f} V")
    print(f"Start input voltage       : {result['start_voltage_mid']:.3f} V")
    print(f"Start Energy_Wh range     : {result['start_energy_wh_low']:.3f} Wh to {result['start_energy_wh_high']:.3f} Wh")
    print(f"Final Energy_Wh range     : {result['final_energy_wh_low']:.3f} Wh to {result['final_energy_wh_high']:.3f} Wh")
    print(f"Final voltage range       : {result['final_voltage_low']:.3f} V to {result['final_voltage_high']:.3f} V")
    print(f"Final input-line voltage  : {result['final_voltage_mid']:.3f} V")

    print("\n--- Files Created ---")
    print(f"1. {VOLTAGE_TIME_PLOT}")
    print("   Voltage vs time with lower/input/upper profiles and DirectSun/Umbra shading.")
    print(f"2. {VOLTAGE_WH_PLOT}")
    print("   Raw Excel V-Wh points, isotonic V-Wh curve, and operating paths.")
    print(f"3. {PROFILE_CSV}")
    print("   Exact voltage and Energy_Wh values for every DirectSun/Umbra segment.")
    print(f"4. {ISOTONIC_TABLE_CSV}")
    print("   Corrected monotonic V-Wh table created by isotonic regression.")
