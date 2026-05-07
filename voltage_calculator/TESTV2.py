import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
'''from sklearn.isotonic import IsotonicRegression'''


# =========================
# TIME HELPERS
# =========================
def parse_utc_time(value):
    return pd.to_datetime(value, utc=True)


def overlap_seconds(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return max(0.0, (earliest_end - latest_start).total_seconds())


# =========================
# LIGHTING CALCULATION
# =========================
def calculate_lighting_between_passes(lighting_csv_path, pass_1_end, pass_2_start):
    window_start = parse_utc_time(pass_1_end)
    window_end = parse_utc_time(pass_2_start)

    df = pd.read_csv(lighting_csv_path)
    df["Start Time (UTC)"] = pd.to_datetime(df["Start Time (UTC)"], utc=True)
    df["End Time (UTC)"] = pd.to_datetime(df["End Time (UTC)"], utc=True)
    df["Condition"] = df["Condition"].str.upper()

    rows = []
    totals = {"DIRECTSUN": 0.0, "UMBRA": 0.0}

    for _, r in df.iterrows():
        sec = overlap_seconds(window_start, window_end,
                              r["Start Time (UTC)"], r["End Time (UTC)"])
        if sec > 0:
            rows.append({
                "condition": r["Condition"],
                "overlap_start_utc": max(window_start, r["Start Time (UTC)"]),
                "overlap_end_utc": min(window_end, r["End Time (UTC)"]),
                "overlap_seconds": sec
            })
            if r["Condition"] in totals:
                totals[r["Condition"]] += sec

    return {
        "details": pd.DataFrame(rows),
        "directsun_seconds": totals["DIRECTSUN"],
        "umbra_seconds": totals["UMBRA"]
    }


# =========================
# LOOKUP FUNCTIONS
# =========================
def load_capacity_table(file_path):
    df = pd.read_excel(file_path, sheet_name="V_Wh")
    df = df[["TOTAL_BATTERY_VOLTAGE", "Energy_Wh"]].dropna()
    df = df.sort_values("Energy_Wh")
    return df


def energy_to_voltage(e, voltages, energies):
    return np.interp(e, energies, voltages)


def voltage_to_energy(v, voltages, energies):
    order = np.argsort(voltages)
    return np.interp(v, voltages[order], energies[order])


# =========================
# PROFILE FOR SINGLE VOLTAGE
# =========================
def build_profile(details_df, start_voltage,
                  voltages, energies,
                  charge_power, discharge_power):

    energy = voltage_to_energy(start_voltage, voltages, energies)

    rows = []

    for _, r in details_df.iterrows():
        t0, t1 = r["overlap_start_utc"], r["overlap_end_utc"]
        dt = r["overlap_seconds"]

        e0 = energy

        if r["condition"] == "DIRECTSUN":
            energy -= (dt / 3600.0) * charge_power
        else:
            energy += (dt / 3600.0) * discharge_power

        v0 = energy_to_voltage(e0, voltages, energies)
        v1 = energy_to_voltage(energy, voltages, energies)

        rows.append((t0, t1, v0, v1))

    return rows


# =========================
# PLOTTING (3 LINES)
# =========================
def plot_three_lines(details_df, result,
                     voltages, energies,
                     charge_power, discharge_power):

    v_low = result["start_voltage_low"]
    v_high = result["start_voltage_high"]
    v_input = result["input_voltage"]

    low = build_profile(details_df, v_low, voltages, energies, charge_power, discharge_power)
    high = build_profile(details_df, v_high, voltages, energies, charge_power, discharge_power)
    mid = build_profile(details_df, v_input, voltages, energies, charge_power, discharge_power)

    plt.figure(figsize=(12,6))
    ax = plt.gca()

    # BACKGROUND SHADING
    for _, row in details_df.iterrows():
        if row["condition"] == "DIRECTSUN":
            ax.axvspan(row["overlap_start_utc"], row["overlap_end_utc"],
                       color="orange", alpha=0.15)
        else:
            ax.axvspan(row["overlap_start_utc"], row["overlap_end_utc"],
                       color="gray", alpha=0.2)
            
    # 🔵 LOWER LINE
    # =========================
    for t0, t1, v0, v1 in low:
        ax.plot([t0, t1], [v0, v1], color="blue", linewidth=2, label="Lower bound")

    # =========================
    # 🔴 UPPER LINE
    # =========================
    for t0, t1, v0, v1 in high:
        ax.plot([t0, t1], [v0, v1], color="red", linewidth=2, label="Upper bound")

    # =========================
    # ⚫ INPUT LINE
    # =========================
    for t0, t1, v0, v1 in mid:
        ax.plot([t0, t1], [v0, v1],
                color="black", linestyle="--", linewidth=3, label="Input")
        
    # Fix duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(),
              dict(zip(labels, handles)).keys())

    ax.set_title("Battery Voltage Band with Sunlit/Umbra")
    ax.set_xlabel("UTC Time")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    return mid

    '''for t0, t1, v0, v1 in low:
        plt.plot([t0, t1], [v0, v1], color="blue", label="Lower")

    for t0, t1, v0, v1 in high:
        plt.plot([t0, t1], [v0, v1], color="red", label="Upper")

    for t0, t1, v0, v1 in mid:
        plt.plot([t0, t1], [v0, v1], color="black", linestyle="--", label="Input")

    plt.legend()
    plt.title("Voltage Band (3 Lines)")
    plt.grid()
    plt.xticks(rotation=30)
    plt.show()

    return mid'''


# =========================
# ISOTONIC SMOOTHING
# =========================
'''def plot_isotonic(profile):

    times = []
    volts = []

    for t0, t1, v0, v1 in profile:
        times += [t0.timestamp(), t1.timestamp()]
        volts += [v0, v1]

    ir = IsotonicRegression(increasing=False)
    smooth_v = ir.fit_transform(times, volts)

    times = pd.to_datetime(times, unit="s")

    plt.figure(figsize=(12,6))
    plt.plot(times, smooth_v, linewidth=3)
    plt.title("Isotonic Smoothed Voltage")
    plt.grid()
    plt.show()'''


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    capacity_excel = "EPS_Capacity_Test_Processed_1.2V.xlsx"
    lighting_csv = "Lighting.csv"

    input_voltage = float(input("Enter voltage (e.g. 27.7): "))
    pass1 = input("Pass1 end UTC: ")
    pass2 = input("Pass2 start UTC: ")

    charge_power = 25.0
    discharge_power = 15.0

    lighting = calculate_lighting_between_passes(lighting_csv, pass1, pass2)

    x = lighting["directsun_seconds"]
    y = lighting["umbra_seconds"]

    net_wh = (x/3600)*charge_power - ((x+y)/3600)*discharge_power

    df = load_capacity_table(capacity_excel)
    voltages = df["TOTAL_BATTERY_VOLTAGE"].values
    energies = df["Energy_Wh"].values

    # Band extraction
    band = df[(df["TOTAL_BATTERY_VOLTAGE"] >= input_voltage) &
              (df["TOTAL_BATTERY_VOLTAGE"] < input_voltage+0.1)]

    result = {
        "input_voltage": input_voltage,
        "start_voltage_low": band["TOTAL_BATTERY_VOLTAGE"].min(),
        "start_voltage_high": band["TOTAL_BATTERY_VOLTAGE"].max()
    }

    print("\nStart band:",
          result["start_voltage_low"],
          "to",
          result["start_voltage_high"])

    mid_profile = plot_three_lines(
        lighting["details"],
        result,
        voltages,
        energies,
        charge_power,
        discharge_power
    )

    # OPTIONAL isotonic
    #plot_isotonic(mid_profile)