import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FILE = "EPS Charge Discharge Test.xlsx"


def load_sheet(sheet_name, header_mode="normal"):
    if header_mode == "normal":
        df = pd.read_excel(FILE, sheet_name=sheet_name)
    elif header_mode == "no_header":
        df = pd.read_excel(FILE, sheet_name=sheet_name, header=None)
        df.columns = ["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "CHARGE_CURRENT"]
    else:
        raise ValueError(f"Invalid header_mode: {header_mode}")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "CHARGE_CURRENT"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{sheet_name}: {col} column not found")

    df = df[["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "CHARGE_CURRENT"]].copy()

    df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    df["TOTAL_BATTERY_VOLTAGE"] = pd.to_numeric(df["TOTAL_BATTERY_VOLTAGE"], errors="coerce")
    df["CHARGE_CURRENT"] = pd.to_numeric(df["CHARGE_CURRENT"], errors="coerce")

    df = df.dropna(subset=["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "CHARGE_CURRENT"])
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    # Real datetime column for plotting
    df["TIMESTAMP_UTC_DT"] = pd.to_datetime(df["TIMESTAMP"], unit="s", utc=True).dt.tz_localize(None)

    # Text column for Excel export
    df["TIMESTAMP_UTC"] = df["TIMESTAMP_UTC_DT"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Time difference in seconds
    df["dt_sec"] = df["TIMESTAMP"].diff().fillna(0)
    df.loc[df["dt_sec"] < 0, "dt_sec"] = 0

    # Use absolute current so charging capacity and energy stay positive
    df["CURRENT_ABS_A"] = df["CHARGE_CURRENT"].abs()

    # Capacity calculation
    df["d_mAh"] = df["CURRENT_ABS_A"] * df["dt_sec"] / 3600.0 * 1000.0
    df["Capacity_mAh"] = df["d_mAh"].cumsum()
    df["Capacity_Ah"] = df["Capacity_mAh"] / 1000.0

    # Power and energy calculation
    df["Power_W"] = df["TOTAL_BATTERY_VOLTAGE"] * df["CURRENT_ABS_A"]
    df["d_Wh"] = df["Power_W"] * df["dt_sec"] / 3600.0
    df["Energy_Wh"] = df["d_Wh"].cumsum()

    return df


chg_df = load_sheet("Charging", header_mode="normal")

# Save processed charging data
with pd.ExcelWriter("EPS-Charge-Processed.xlsx", engine="openpyxl") as writer:
    chg_df.to_excel(writer, sheet_name="Charging_Processed", index=False)

# Plot 1: Voltage vs mAh
plt.figure(figsize=(10, 6))
plt.plot(chg_df["Capacity_mAh"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
plt.scatter(chg_df["Capacity_mAh"], chg_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
plt.xlabel("Capacity (mAh)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs mAh")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("charging_voltage_vs_mAh.png", dpi=300)
plt.close()

# Plot 2: Voltage vs Ah
plt.figure(figsize=(10, 6))
plt.plot(chg_df["Capacity_Ah"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
plt.scatter(chg_df["Capacity_Ah"], chg_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
plt.xlabel("Capacity (Ah)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Ah")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("charging_voltage_vs_Ah.png", dpi=300)
plt.close()

# Plot 3: Voltage vs Time (UTC)
plt.figure(figsize=(10, 6))
plt.plot(chg_df["TIMESTAMP_UTC_DT"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
plt.scatter(chg_df["TIMESTAMP_UTC_DT"], chg_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Voltage (V)")
plt.title("Charging Voltage vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charging_voltage_vs_Time.png", dpi=300)
plt.close()

# Plot 4: Current vs Time (UTC)
plt.figure(figsize=(10, 6))
plt.plot(chg_df["TIMESTAMP_UTC_DT"], chg_df["CHARGE_CURRENT"], label="Charging", linewidth=2)
plt.scatter(chg_df["TIMESTAMP_UTC_DT"], chg_df["CHARGE_CURRENT"], color="red", s=12, zorder=3)
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Charge Current (A)")
plt.title("Charging Current vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charging_Current_vs_Time.png", dpi=300)
plt.close()

# Plot 5: Capacity vs Time (UTC)
plt.figure(figsize=(10, 6))
plt.plot(chg_df["TIMESTAMP_UTC_DT"], chg_df["Capacity_Ah"], label="Charging", linewidth=2)
plt.scatter(chg_df["TIMESTAMP_UTC_DT"], chg_df["Capacity_Ah"], color="red", s=12, zorder=3)
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Capacity (Ah)")
plt.title("Charging Capacity vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charging_Capacity_vs_Time.png", dpi=300)
plt.close()

# Plot 6: Voltage and Current vs Time (UTC)
fig, ax1 = plt.subplots(figsize=(12, 6))

line1 = ax1.plot(
    chg_df["TIMESTAMP_UTC_DT"],
    chg_df["TOTAL_BATTERY_VOLTAGE"],
    label="Voltage",
    linewidth=2
)
ax1.scatter(
    chg_df["TIMESTAMP_UTC_DT"],
    chg_df["TOTAL_BATTERY_VOLTAGE"],
    color="red",
    s=12,
    zorder=3
)
ax1.set_xlabel("Timestamp (UTC)")
ax1.set_ylabel("Voltage (V)")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))

ax2 = ax1.twinx()
line2 = ax2.plot(
    chg_df["TIMESTAMP_UTC_DT"],
    chg_df["CHARGE_CURRENT"],
    label="Current",
    color= "black",
    linewidth=2,
)
ax2.scatter(
    chg_df["TIMESTAMP_UTC_DT"],
    chg_df["CHARGE_CURRENT"],
    color="orange",
    s=12,
    zorder=3
)
ax2.set_ylabel("Charge Current (A)")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")
plt.title("Charging Voltage and Current vs Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Charging_Voltage_Current_vs_Time.png", dpi=300)
plt.close()

# Plot 7: Voltage vs Wh
plt.figure(figsize=(10, 6))
plt.plot(chg_df["Energy_Wh"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
plt.scatter(chg_df["Energy_Wh"], chg_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
plt.xlabel("Energy (Wh)")
plt.ylabel("Voltage (V)")
plt.title("Charging Voltage vs Wh")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("charging_voltage_vs_Wh.png", dpi=300)
plt.close()

print("Done.")
print("Created:")
print("1. EPS-Charge-Processed.xlsx")
print("2. charging_voltage_vs_mAh.png")
print("3. charging_voltage_vs_Ah.png")
print("4. charging_voltage_vs_Time.png")
print("5. charging_Current_vs_Time.png")
print("6. charging_Capacity_vs_Time.png")
print("7. Charging_Voltage_Current_vs_Time.png")
print("8. charging_voltage_vs_Wh.png")
print()
print(f"Measured cumulative charging energy from dataset: {chg_df['Energy_Wh'].iloc[-1]:.3f} Wh")