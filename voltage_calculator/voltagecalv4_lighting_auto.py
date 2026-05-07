import pandas as pd
import numpy as np


# ============================================================
# Time helper functions
# ============================================================
def parse_utc_time(value):
    """Parse UTC timestamp like 2026-05-05T12:28:53.865Z."""
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts


def overlap_seconds(a_start, a_end, b_start, b_end):
    """Return overlap duration in seconds between interval A and interval B."""
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = (earliest_end - latest_start).total_seconds()
    return max(0.0, overlap)


def calculate_lighting_between_passes(
    lighting_csv_path,
    pass_1_end_time_utc,
    pass_2_start_time_utc,
    start_col="Start Time (UTC)",
    end_col="End Time (UTC)",
    condition_col="Condition",
):
    """
    Calculates total DIRECTSUN and UMBRA seconds between:
        End time of ground pass 1  ->  Start time of ground pass 2

    This automatically intersects the user-defined comms gap with all rows in Lighting.csv.
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
            rows.append({
                "condition": condition,
                "lighting_start_utc": row[start_col],
                "lighting_end_utc": row[end_col],
                "overlap_start_utc": max(window_start, row[start_col]),
                "overlap_end_utc": min(window_end, row[end_col]),
                "overlap_seconds": seconds,
            })

            if condition in totals:
                totals[condition] += seconds

    details_df = pd.DataFrame(rows)

    return {
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "gap_seconds": (window_end - window_start).total_seconds(),
        "directsun_seconds": totals["DIRECTSUN"],
        "umbra_seconds": totals["UMBRA"],
        "details": details_df,
    }


# ============================================================
# Existing voltage-energy helper functions
# ============================================================
def voltage_to_energy(voltage, voltages, energies):
    order = np.argsort(voltages)
    v_sorted = voltages[order]
    e_sorted = energies[order]
    return np.interp(voltage, v_sorted, e_sorted)


def energy_to_voltage(energy, voltages, energies):
    order = np.argsort(energies)
    e_sorted = energies[order]
    v_sorted = voltages[order]
    return np.interp(energy, e_sorted, v_sorted)


def calculate_voltage_range_from_excel(
    file_path,
    input_voltage,
    net_wh,
    sheet_name="V_Wh",
    voltage_col="TOTAL_BATTERY_VOLTAGE",
    energy_col="Energy_Wh",
    voltage_resolution=0.1,
):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df[[voltage_col, energy_col]].dropna()
    df = df.sort_values(energy_col).reset_index(drop=True)

    voltages = df[voltage_col].to_numpy(dtype=float)
    energies = df[energy_col].to_numpy(dtype=float)

    v_low = input_voltage
    v_high = input_voltage + voltage_resolution

    matching_rows = df[
        (df[voltage_col] >= v_low) &
        (df[voltage_col] < v_high)
    ].copy()

    if matching_rows.empty:
        raise ValueError(
            f"No rows found for voltage range {v_low:.2f} V to {v_high:.2f} V"
        )

    start_voltages = matching_rows[voltage_col].to_numpy(dtype=float)
    start_energies = matching_rows[energy_col].to_numpy(dtype=float)

    final_energies = start_energies + net_wh

    min_energy = energies.min()
    max_energy = energies.max()
    final_energies = np.clip(final_energies, min_energy, max_energy)

    final_voltages = np.array([
        energy_to_voltage(e, voltages, energies)
        for e in final_energies
    ])

    return {
        "input_voltage_requested": input_voltage,
        "voltage_band_low": v_low,
        "voltage_band_high": v_high,
        "number_of_matching_rows": len(matching_rows),
        "start_voltage_low": start_voltages.min(),
        "start_voltage_high": start_voltages.max(),
        "start_energy_wh_low": start_energies.min(),
        "start_energy_wh_high": start_energies.max(),
        "net_wh": net_wh,
        "final_energy_wh_low": final_energies.min(),
        "final_energy_wh_high": final_energies.max(),
        "final_voltage_low": final_voltages.min(),
        "final_voltage_high": final_voltages.max(),
    }


# ============================================================
# Safe input
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
# Main
# ============================================================
if __name__ == "__main__":

    capacity_excel_path = "EPS_Capacity_Test_Processed_1.2V.xlsx"
    lighting_csv_path = "Lighting.csv"

    print("\n=== Voltage Calculator with Automatic DirectSun/Umbra Time ===\n")

    input_voltage = get_float_input("Enter start voltage, example 29.7: ")

    print("\nEnter the comms gap timestamps from Possible passes.csv")
    pass_1_end_time = get_text_input("Enter END time of ground pass 1 UTC: ")
    pass_2_start_time = get_text_input("Enter START time of ground pass 2 UTC: ")

    charge_power_w = get_float_input("Enter charging power in W, example 28: ")
    discharge_power_w = get_float_input("Enter discharging power in W, example 12: ")

    lighting_result = calculate_lighting_between_passes(
        lighting_csv_path=lighting_csv_path,
        pass_1_end_time_utc=pass_1_end_time,
        pass_2_start_time_utc=pass_2_start_time,
    )

    x = lighting_result["directsun_seconds"]
    y = lighting_result["umbra_seconds"]

    A = (x / 3600.0) * charge_power_w
    B = (y / 3600.0) * discharge_power_w
    net_wh = A - B

    print("\n--- Time Window from Comms Passes ---")
    print(f"Window start, pass 1 end : {lighting_result['window_start_utc']}")
    print(f"Window end, pass 2 start : {lighting_result['window_end_utc']}")
    print(f"Total gap time           : {lighting_result['gap_seconds']:.3f} s")

    print("\n--- Automatic Lighting Intersection ---")
    print(f"Charging time in DIRECTSUN, x : {x:.3f} s")
    print(f"Discharging time in UMBRA, y  : {y:.3f} s")

    if not lighting_result["details"].empty:
        print("\nDetailed overlap rows:")
        print(lighting_result["details"].to_string(index=False))

    print("\n--- Energy Calculation ---")
    print(f"Charging Energy, A      : {A:.3f} Wh")
    print(f"Discharging Energy, B   : {B:.3f} Wh")
    print(f"Net Energy, A - B       : {net_wh:.3f} Wh")

    if net_wh > 0:
        print("Net Charging: Power Positive")
    elif net_wh < 0:
        print("Net Discharging: Power Negative")
    else:
        print("Balanced")

    try:
        result = calculate_voltage_range_from_excel(
            file_path=capacity_excel_path,
            input_voltage=input_voltage,
            net_wh=net_wh,
        )

        print("\n--- Result Range ---")
        print(f"Requested voltage      : {result['input_voltage_requested']:.2f} V")
        print(f"Voltage band used      : {result['voltage_band_low']:.2f} V to {result['voltage_band_high']:.2f} V")
        print(f"Matching rows          : {result['number_of_matching_rows']}")
        print(f"Start voltage range    : {result['start_voltage_low']:.3f} V to {result['start_voltage_high']:.3f} V")
        print(f"Start Energy_Wh range  : {result['start_energy_wh_low']:.3f} Wh to {result['start_energy_wh_high']:.3f} Wh")
        print(f"Final Energy_Wh range  : {result['final_energy_wh_low']:.3f} Wh to {result['final_energy_wh_high']:.3f} Wh")
        print(f"Final voltage range    : {result['final_voltage_low']:.3f} V to {result['final_voltage_high']:.3f} V")

    except Exception as e:
        print(f"\nVoltage lookup error: {e}")
