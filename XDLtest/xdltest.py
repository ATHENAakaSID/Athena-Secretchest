import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================

INPUT_FILE = "EPS_Capacity_Test.txt"

# Set this to the actual logging interval if you know it.
# Example: 1.0 means one reading every second.
ASSUMED_SAMPLE_PERIOD_SEC = 1.0

OUTPUT_EXCEL = "EPS_Capacity_Test_Processed.xlsx"
OUTPUT_CSV = "EPS_Capacity_Test_Processed.csv"


def extract_float(block: str, label: str):
    """
    Extract a float value for a given label from one decoded block.
    Returns None if not found.
    """
    pattern = rf"{re.escape(label)}\s*\|\s*([-+]?\d+(?:\.\d+)?)"
    match = re.search(pattern, block)
    return float(match.group(1)) if match else None


def parse_log_file(file_path: str) -> pd.DataFrame:
    """
    Parse repeated BMS_Get_Batt_V_and_I_Readings blocks from the text log.
    Only keeps blocks with CRC Match.
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

    blocks = re.split(
        r"=+\s*\n\s*Command:\s*BMS_Get_Batt_V_and_I_Readings\s*\n=+",
        text
    )

    records = []

    for block in blocks:
        if "Total Battery Voltage Reading" not in block:
            continue

        if "CRC Match" not in block:
            continue

        total_batt_v = extract_float(block, "Total Battery Voltage Reading")
        batt_current = extract_float(block, "Battery Current Reading")
        dsw1_current = extract_float(block, "Discharge Switch 1 Current Reading")
        dsw2_current = extract_float(block, "Discharge Switch 2 Current Reading")

        if total_batt_v is None or batt_current is None:
            continue

        records.append({
            "TOTAL_BATTERY_VOLTAGE": total_batt_v,
            "BATTERY_CURRENT": batt_current,
            "DISCHARGE_SWITCH_1_CURRENT": dsw1_current,
            "DISCHARGE_SWITCH_2_CURRENT": dsw2_current,
            "CRC_OK": True,
        })

    if not records:
        raise ValueError("No valid blocks with CRC Match were found in the log.")

    df = pd.DataFrame(records)
    df["SAMPLE_INDEX"] = range(len(df))

    return df


def build_time_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since no real timestamp is available, build elapsed time from sample index.
    """
    df = df.copy()

    df["dt_sec"] = ASSUMED_SAMPLE_PERIOD_SEC
    df.loc[0, "dt_sec"] = 0.0

    df["ELAPSED_TIME_SEC"] = df["SAMPLE_INDEX"] * ASSUMED_SAMPLE_PERIOD_SEC
    df["ELAPSED_TIME_MIN"] = df["ELAPSED_TIME_SEC"] / 60.0

    return df


def compute_capacity_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute capacity and energy from current, voltage, and assumed dt.
    """
    df = df.copy()

    # Use pack current magnitude so discharge accumulates positively
    df["CURRENT_ABS_A"] = df["BATTERY_CURRENT"].abs()

    # Capacity
    df["d_mAh"] = df["CURRENT_ABS_A"] * df["dt_sec"] / 3600.0 * 1000.0
    df["Capacity_mAh"] = df["d_mAh"].cumsum()
    df["Capacity_Ah"] = df["Capacity_mAh"] / 1000.0

    # Power and Energy
    df["Power_W"] = df["TOTAL_BATTERY_VOLTAGE"] * df["CURRENT_ABS_A"]
    df["d_Wh"] = df["Power_W"] * df["dt_sec"] / 3600.0
    df["Energy_Wh"] = df["d_Wh"].cumsum()

    return df


def save_outputs(df: pd.DataFrame):
    df.to_csv(OUTPUT_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Capacity_Test_Processed", index=False)


def plot_voltage_vs_mAh(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Capacity_mAh"], df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
    plt.scatter(df["Capacity_mAh"], df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
    plt.xlabel("Capacity (mAh)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs mAh")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("txt_voltage_vs_mAh.png", dpi=300)
    plt.close()


def plot_voltage_vs_Ah(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Capacity_Ah"], df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
    plt.scatter(df["Capacity_Ah"], df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
    plt.xlabel("Capacity (Ah)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs Ah")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("txt_voltage_vs_Ah.png", dpi=300)
    plt.close()


def plot_voltage_vs_elapsed_time(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["ELAPSED_TIME_MIN"], df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
    plt.scatter(df["ELAPSED_TIME_MIN"], df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs Elapsed Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("txt_voltage_vs_elapsed_time.png", dpi=300)
    plt.close()


def plot_current_vs_elapsed_time(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["ELAPSED_TIME_MIN"], df["BATTERY_CURRENT"], label="Battery Current", linewidth=2)
    plt.scatter(df["ELAPSED_TIME_MIN"], df["BATTERY_CURRENT"], color="red", s=12, zorder=3)
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel("Battery Current (A)")
    plt.title("Battery Current vs Elapsed Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("txt_current_vs_elapsed_time.png", dpi=300)
    plt.close()


def plot_capacity_vs_elapsed_time(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["ELAPSED_TIME_MIN"], df["Capacity_Ah"], label="Capacity", linewidth=2)
    plt.scatter(df["ELAPSED_TIME_MIN"], df["Capacity_Ah"], color="red", s=12, zorder=3)
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel("Capacity (Ah)")
    plt.title("Capacity vs Elapsed Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("txt_capacity_vs_elapsed_time.png", dpi=300)
    plt.close()


def plot_voltage_current_vs_elapsed_time(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    line1 = ax1.plot(
        df["ELAPSED_TIME_MIN"],
        df["TOTAL_BATTERY_VOLTAGE"],
        label="Voltage",
        linewidth=2
    )
    ax1.scatter(
        df["ELAPSED_TIME_MIN"],
        df["TOTAL_BATTERY_VOLTAGE"],
        color="red",
        s=12,
        zorder=3
    )
    ax1.set_xlabel("Elapsed Time (min)")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        df["ELAPSED_TIME_MIN"],
        df["BATTERY_CURRENT"],
        label="Current",
        linewidth=2,
        linestyle="--"
    )
    ax2.scatter(
        df["ELAPSED_TIME_MIN"],
        df["BATTERY_CURRENT"],
        color="red",
        s=12,
        zorder=3
    )
    ax2.set_ylabel("Battery Current (A)")

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Voltage and Current vs Elapsed Time")
    plt.tight_layout()
    plt.savefig("txt_voltage_current_vs_elapsed_time.png", dpi=300)
    plt.close()


def plot_voltage_vs_Wh(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Energy_Wh"], df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
    plt.scatter(df["Energy_Wh"], df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
    plt.xlabel("Energy (Wh)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs Wh")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("txt_voltage_vs_Wh.png", dpi=300)
    plt.close()


def plot_switch_currents_vs_elapsed_time(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["ELAPSED_TIME_MIN"], df["DISCHARGE_SWITCH_1_CURRENT"], label="Discharge Switch 1", linewidth=2)
    plt.plot(df["ELAPSED_TIME_MIN"], df["DISCHARGE_SWITCH_2_CURRENT"], label="Discharge Switch 2", linewidth=2)
    plt.scatter(df["ELAPSED_TIME_MIN"], df["DISCHARGE_SWITCH_1_CURRENT"], color="red", s=10, zorder=3)
    plt.scatter(df["ELAPSED_TIME_MIN"], df["DISCHARGE_SWITCH_2_CURRENT"], color="red", s=10, zorder=3)
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel("Switch Current (A)")
    plt.title("Discharge Switch Currents vs Elapsed Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("txt_switch_currents_vs_elapsed_time.png", dpi=300)
    plt.close()


def main():
    df = parse_log_file(INPUT_FILE)
    df = build_time_base(df)
    df = compute_capacity_energy(df)

    save_outputs(df)

    plot_voltage_vs_mAh(df)
    plot_voltage_vs_Ah(df)
    plot_voltage_vs_elapsed_time(df)
    plot_current_vs_elapsed_time(df)
    plot_capacity_vs_elapsed_time(df)
    plot_voltage_current_vs_elapsed_time(df)
    plot_voltage_vs_Wh(df)
    plot_switch_currents_vs_elapsed_time(df)

    print("Done.")
    print(f"Parsed valid records : {len(df)}")
    print(f"Assumed sample period: {ASSUMED_SAMPLE_PERIOD_SEC} sec")
    print(f"Processed Excel      : {OUTPUT_EXCEL}")
    print(f"Processed CSV        : {OUTPUT_CSV}")
    print("Created plots:")
    print("1. txt_voltage_vs_mAh.png")
    print("2. txt_voltage_vs_Ah.png")
    print("3. txt_voltage_vs_elapsed_time.png")
    print("4. txt_current_vs_elapsed_time.png")
    print("5. txt_capacity_vs_elapsed_time.png")
    print("6. txt_voltage_current_vs_elapsed_time.png")
    print("7. txt_voltage_vs_Wh.png")
    print("8. txt_switch_currents_vs_elapsed_time.png")
    print()
    print(f"Final cumulative capacity : {df['Capacity_Ah'].iloc[-1]:.3f} Ah")
    print(f"Final cumulative energy   : {df['Energy_Wh'].iloc[-1]:.3f} Wh")


if __name__ == "__main__":
    main()