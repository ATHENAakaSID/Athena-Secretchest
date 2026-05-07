import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Time helper functions=
def parse_utc_time(value):
    """Parse UTC timestamp like 2026-05-05T12:28:53.865Z."""
    return pd.to_datetime(value, utc=True, errors="raise")


def overlap_seconds(a_start, a_end, b_start, b_end):
    """Return overlap duration in seconds between interval A and interval B."""
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = (earliest_end - latest_start).total_seconds()
    return max(0.0, overlap)

# Lighting intersection between comms pass 1 end and pass 2 start
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

    It also returns the exact clipped overlap rows for plotting.
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

    details_df = pd.DataFrame(rows).sort_values("overlap_start_utc").reset_index(drop=True)

    return {
        "window_start_utc": window_start,
        "window_end_utc": window_end,
        "gap_seconds": (window_end - window_start).total_seconds(),
        "directsun_seconds": totals["DIRECTSUN"],
        "umbra_seconds": totals["UMBRA"],
        "details": details_df,
    }

# Voltage-energy helper functions
def energy_to_voltage(energy, voltages, energies):
    """Interpolate voltage from Energy_Wh using the Excel lookup table."""
    order = np.argsort(energies)
    e_sorted = energies[order]
    v_sorted = voltages[order]
    return np.interp(energy, e_sorted, v_sorted)


def force_voltage_direction(start_voltage, raw_end_voltage, direction):
    """
    Force the voltage movement to match the physical power direction.

    direction = +1 means charging / power positive / voltage must increase.
    direction = -1 means discharging / power negative / voltage must decrease.
    direction =  0 means no change required.

    This is needed because the Excel V-vs-Wh lookup table can have small local
    noise/non-monotonic points inside a voltage band. The energy calculation is
    still done using Wh, but the reported/plotted voltage is guarded so that:
        positive power -> voltage goes up
        negative power -> voltage goes down
    """
    start_voltage = float(start_voltage)
    raw_end_voltage = float(raw_end_voltage)
    raw_delta_v = raw_end_voltage - start_voltage

    if direction > 0:
        return start_voltage + abs(raw_delta_v)
    if direction < 0:
        return start_voltage - abs(raw_delta_v)
    return start_voltage


def load_capacity_table(
    file_path,
    sheet_name="V_Wh",
    voltage_col="TOTAL_BATTERY_VOLTAGE",
    energy_col="Energy_Wh",
):
    """Read the V vs Wh lookup table from Excel."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df[[voltage_col, energy_col]].dropna().copy()
    df[voltage_col] = df[voltage_col].astype(float)
    df[energy_col] = df[energy_col].astype(float)
    df = df.sort_values(energy_col).reset_index(drop=True)
    return df, voltage_col, energy_col


def calculate_voltage_range_from_excel(
    file_path,
    input_voltage,
    net_wh,
    sheet_name="V_Wh",
    voltage_col="TOTAL_BATTERY_VOLTAGE",
    energy_col="Energy_Wh",
    voltage_resolution=0.1,
):
    """
    Calculates final voltage range.

    IMPORTANT:
    In this Excel table, Energy_Wh behaves like discharged/used energy.
    Therefore:
        positive net_wh charging  -> Energy_Wh decreases -> voltage increases
        negative net_wh discharge -> Energy_Wh increases -> voltage decreases
    """
    df, voltage_col, energy_col = load_capacity_table(
        file_path=file_path,
        sheet_name=sheet_name,
        voltage_col=voltage_col,
        energy_col=energy_col,
    )

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

    # FIXED SIGN: positive net_wh must increase voltage.
    final_energies = start_energies - net_wh

    min_energy = energies.min()
    max_energy = energies.max()
    final_energies = np.clip(final_energies, min_energy, max_energy)

    raw_final_voltages = np.array([
        energy_to_voltage(e, voltages, energies)
        for e in final_energies
    ])

    # Directional guard for the final voltage range.
    # Instead of trusting the min/max of many noisy lookup rows directly, calculate
    # the voltage movement at the middle of the selected band and apply that same
    # movement to the start voltage range. This guarantees:
    #   net_wh > 0 -> final range is higher than start range
    #   net_wh < 0 -> final range is lower than start range
    #   net_wh = 0 -> final range is unchanged
    start_voltage_mid = float(np.median(start_voltages))
    start_energy_mid = float(np.median(start_energies))
    final_energy_mid = float(np.clip(start_energy_mid - net_wh, min_energy, max_energy))
    raw_final_voltage_mid = float(energy_to_voltage(final_energy_mid, voltages, energies))

    if net_wh > 0:
        direction = 1
    elif net_wh < 0:
        direction = -1
    else:
        direction = 0

    final_voltage_mid = force_voltage_direction(
        start_voltage=start_voltage_mid,
        raw_end_voltage=raw_final_voltage_mid,
        direction=direction,
    )
    delta_v_mid = final_voltage_mid - start_voltage_mid

    final_voltages = start_voltages + delta_v_mid

    return {
        "input_voltage_requested": input_voltage,
        "voltage_band_low": v_low,
        "voltage_band_high": v_high,
        "number_of_matching_rows": len(matching_rows),
        "start_voltage_low": start_voltages.min(),
        "start_voltage_high": start_voltages.max(),
        "start_energy_wh_low": start_energies.min(),
        "start_energy_wh_high": start_energies.max(),
        "start_energy_wh_mid": start_energy_mid,
        "start_voltage_mid": start_voltage_mid,
        "net_wh": net_wh,
        "voltage_direction": direction,
        "final_energy_wh_low": final_energies.min(),
        "final_energy_wh_high": final_energies.max(),
        "raw_final_voltage_low": raw_final_voltages.min(),
        "raw_final_voltage_high": raw_final_voltages.max(),
        "final_voltage_low": final_voltages.min(),
        "final_voltage_high": final_voltages.max(),
        "final_voltage_mid": final_voltage_mid,
        "delta_voltage_mid": delta_v_mid,
    }


# Build time history for plotting
def build_voltage_time_profile(
    details_df,
    start_energy_wh,
    voltages,
    energies,
    charge_power_w,
    discharge_power_w,
):
    """
    Builds a step-by-step profile between pass 1 end and pass 2 start.

    DIRECTSUN:
        charging happens, so used Energy_Wh decreases.
    UMBRA:
        discharging happens, so used Energy_Wh increases.
    """
    if details_df.empty:
        raise ValueError("No DIRECTSUN/UMBRA overlap rows found in the selected pass gap.")

    current_energy = float(start_energy_wh)
    min_energy = float(np.min(energies))
    max_energy = float(np.max(energies))

    profile_rows = []

    for _, row in details_df.iterrows():
        condition = row["condition"]
        t0 = row["overlap_start_utc"]
        t1 = row["overlap_end_utc"]
        dt_seconds = float(row["overlap_seconds"])

        energy_start = current_energy

        if condition == "DIRECTSUN":
            delta_wh = (dt_seconds / 3600.0) * charge_power_w
            # charging reduces discharged/used Energy_Wh
            current_energy = current_energy - delta_wh
        elif condition == "UMBRA":
            delta_wh = (dt_seconds / 3600.0) * discharge_power_w
            # discharging increases discharged/used Energy_Wh
            current_energy = current_energy + delta_wh
        else:
            delta_wh = 0.0

        current_energy = float(np.clip(current_energy, min_energy, max_energy))
        energy_end = current_energy

        v0 = energy_to_voltage(energy_start, voltages, energies)
        raw_v1 = energy_to_voltage(energy_end, voltages, energies)

        if condition == "DIRECTSUN":
            direction = 1
        elif condition == "UMBRA":
            direction = -1
        else:
            direction = 0

        v1 = force_voltage_direction(
            start_voltage=v0,
            raw_end_voltage=raw_v1,
            direction=direction,
        )

        profile_rows.append({
            "condition": condition,
            "start_time_utc": t0,
            "end_time_utc": t1,
            "duration_seconds": dt_seconds,
            "energy_start_wh": energy_start,
            "energy_end_wh": energy_end,
            "voltage_start_v": v0,
            "raw_voltage_end_v": raw_v1,
            "voltage_end_v": v1,
            "delta_wh": energy_end - energy_start,
            "delta_v": v1 - v0,
        })

    return pd.DataFrame(profile_rows)



# Plotting functions
def plot_voltage_vs_time(profile_df, output_path="voltage_vs_time_lighting.png"):
    plt.figure(figsize=(12, 6))

    for _, row in profile_df.iterrows():
        color = "orange" if row["condition"] == "DIRECTSUN" else "gray"
        label = row["condition"]
        plt.plot(
            [row["start_time_utc"], row["end_time_utc"]],
            [row["voltage_start_v"], row["voltage_end_v"]],
            marker="o",
            linewidth=3,
            color=color,
            label=label,
        )
        plt.axvspan(row["start_time_utc"], row["end_time_utc"], color=color, alpha=0.12)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.title("Battery Voltage Change Between Ground Passes")
    plt.xlabel("UTC Time")
    plt.ylabel("Battery Voltage (V)")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_v_vs_wh_with_path(
    capacity_df,
    profile_df,
    voltage_col,
    energy_col,
    output_path="voltage_vs_wh_path.png",
):
    plt.figure(figsize=(10, 6))

    # Full Excel lookup curve
    plt.plot(
        capacity_df[energy_col],
        capacity_df[voltage_col],
        linewidth=1.5,
        label="Excel V vs Wh curve",
    )

    # Operating path between the two passes
    for _, row in profile_df.iterrows():
        color = "orange" if row["condition"] == "DIRECTSUN" else "gray"
        label = row["condition"]
        plt.plot(
            [row["energy_start_wh"], row["energy_end_wh"]],
            [row["voltage_start_v"], row["voltage_end_v"]],
            marker="o",
            linewidth=3,
            color=color,
            label=label,
        )

    # Mark exact start and final points
    first = profile_df.iloc[0]
    last = profile_df.iloc[-1]
    plt.scatter(first["energy_start_wh"], first["voltage_start_v"], s=90, marker="s", label="Start")
    plt.scatter(last["energy_end_wh"], last["voltage_end_v"], s=90, marker="*", label="Final")

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.title("Voltage vs Energy_Wh Path Between Ground Passes")
    plt.xlabel("Energy_Wh from Excel table")
    plt.ylabel("Battery Voltage (V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

# Safe input
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

# Main
if __name__ == "__main__":
    capacity_excel_path = "EPS_Capacity_Test_Processed_1.2V.xlsx"
    lighting_csv_path = "Lighting.csv   "

    print("\n=== Voltage Calculator with Automatic DirectSun/Umbra Time, Direction Check, and Plots ===\n")

    input_voltage = get_float_input("Enter start voltage, example 29.7: ")

    print("\nEnter the comms gap timestamps from Possible passes.csv")
    pass_1_end_time = get_text_input("Enter END time of ground pass 1 UTC: ")
    pass_2_start_time = get_text_input("Enter START time of ground pass 2 UTC: ")

    #charge_power_wheninSUNLIT_W = get_float_input("Enter charging power in W During Sunlit ONLY ")
    #discharge_power_Throughout_w = get_float_input("Enter discharging power in W During Umbra ONLY, ")
    charge_power_wheninSUNLIT_W = 25.0 #needs to confirm this with the data!!!!!!!!
    discharge_power_Throughout_w = 15.0  # Even during sunlit, the battery is discharging to the load, so we consider this as always happening.  

    lighting_result = calculate_lighting_between_passes(
        lighting_csv_path=lighting_csv_path,
        pass_1_end_time_utc=pass_1_end_time,
        pass_2_start_time_utc=pass_2_start_time,
    )

    x = lighting_result["directsun_seconds"]
    y = lighting_result["umbra_seconds"]

    A = (x / 3600.0) * charge_power_wheninSUNLIT_W
    B = ((x+y) / 3600.0) * discharge_power_Throughout_w
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
        print(f"Voltage change, mid    : {result['delta_voltage_mid']:.4f} V")
        if result["voltage_direction"] > 0:
            print("Direction check        : PASS, power positive increased voltage")
        elif result["voltage_direction"] < 0:
            print("Direction check        : PASS, power negative decreased voltage")
        else:
            print("Direction check        : Balanced, no voltage direction forced")

        # Build the detailed voltage-vs-time path using the middle energy value
        capacity_df, voltage_col, energy_col = load_capacity_table(capacity_excel_path)
        voltages = capacity_df[voltage_col].to_numpy(dtype=float)
        energies = capacity_df[energy_col].to_numpy(dtype=float)

        profile_df = build_voltage_time_profile(
            details_df=lighting_result["details"],
            start_energy_wh=result["start_energy_wh_mid"],
            voltages=voltages,
            energies=energies,
            charge_power_w=charge_power_wheninSUNLIT_W,
            discharge_power_w=discharge_power_Throughout_w,
        )

        profile_csv = "voltage_time_profile.csv"
        profile_df.to_csv(profile_csv, index=False)

        plot_voltage_vs_time(profile_df, output_path="voltage_vs_time_lighting.png")
        plot_v_vs_wh_with_path(
            capacity_df=capacity_df,
            profile_df=profile_df,
            voltage_col=voltage_col,
            energy_col=energy_col,
            output_path="voltage_vs_wh_path.png",
        )

        print("\n--- Plot Outputs Created ---")
        print("1. voltage_vs_time_lighting.png")
        print("2. voltage_vs_wh_path.png")
        print("3. voltage_time_profile.csv")

    except Exception as e:
        print(f"\nVoltage lookup or plotting error: {e}")
