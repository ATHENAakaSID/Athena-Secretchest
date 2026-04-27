import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FILE = "EPS Charge Discharge Test.xlsx"


def load_sheet(sheet_name, header_mode="normal"):
    if header_mode == "normal":
        df = pd.read_excel(FILE, sheet_name=sheet_name)
    elif header_mode == "no_header":
        df = pd.read_excel(FILE, sheet_name=sheet_name, header=None)
        df.columns = ["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "DISCHARGE_CURRENT"]
    else:
        raise ValueError(f"Invalid header_mode: {header_mode}")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "DISCHARGE_CURRENT"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{sheet_name}: {col} column not found")

    df = df[["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "DISCHARGE_CURRENT"]].copy()

    df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    df["TOTAL_BATTERY_VOLTAGE"] = pd.to_numeric(df["TOTAL_BATTERY_VOLTAGE"], errors="coerce")
    df["DISCHARGE_CURRENT"] = pd.to_numeric(df["DISCHARGE_CURRENT"], errors="coerce")

    df = df.dropna(subset=["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "DISCHARGE_CURRENT"])
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    # Real datetime column for plotting
    df["TIMESTAMP_UTC_DT"] = pd.to_datetime(df["TIMESTAMP"], unit="s", utc=True).dt.tz_localize(None)

    # Text column for Excel export
    df["TIMESTAMP_UTC"] = df["TIMESTAMP_UTC_DT"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    df["dt_sec"] = df["TIMESTAMP"].diff().fillna(0)
    df.loc[df["dt_sec"] < 0, "dt_sec"] = 0

    # Capacity calculation
    df["d_mAh"] = df["DISCHARGE_CURRENT"] * df["dt_sec"] / 3600.0 * 1000.0
    df["Capacity_mAh"] = df["d_mAh"].cumsum()
    df["Capacity_Ah"] = df["Capacity_mAh"] / 1000.0

    return df


dis_df = load_sheet("Discharging", header_mode="normal")

# Save processed data to Excel
with pd.ExcelWriter("EPS-Charge-Discharge-Processed.xlsx", engine="openpyxl") as writer:
    dis_df.to_excel(writer, sheet_name="Discharging_Processed", index=False)

# Plot 1: Voltage vs mAh
plt.figure(figsize=(10, 6))
plt.plot(dis_df["Capacity_mAh"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
plt.scatter(dis_df["Capacity_mAh"], dis_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
plt.xlabel("Capacity (mAh)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs mAh")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("voltage_vs_mAh.png", dpi=300)
plt.close()

# Plot 2: Voltage vs Ah
plt.figure(figsize=(10, 6))
plt.plot(dis_df["Capacity_Ah"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
plt.scatter(dis_df["Capacity_Ah"], dis_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
plt.xlabel("Capacity (Ah)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Ah")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("voltage_vs_Ah.png", dpi=300)
plt.close()

# Plot 3: Voltage vs Time (UTC)
plt.figure(figsize=(10, 6))
plt.plot(dis_df["TIMESTAMP_UTC_DT"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
plt.scatter(dis_df["TIMESTAMP_UTC_DT"], dis_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("voltage_vs_Time.png", dpi=300)
plt.close()

# Plot 4: Current vs Time (UTC)
plt.figure(figsize=(10, 6))
plt.plot(dis_df["TIMESTAMP_UTC_DT"], dis_df["DISCHARGE_CURRENT"], label="Discharging", linewidth=2)
plt.scatter(dis_df["TIMESTAMP_UTC_DT"], dis_df["DISCHARGE_CURRENT"], color="red", s=12, zorder=3)
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Discharge Current (A)")
plt.title("Current vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Current_vs_Time.png", dpi=300)
plt.close()

# Plot 5: Capacity vs Time (UTC)
plt.figure(figsize=(10, 6))
plt.plot(dis_df["TIMESTAMP_UTC_DT"], dis_df["Capacity_Ah"], label="Discharging", linewidth=2)
plt.scatter(dis_df["TIMESTAMP_UTC_DT"], dis_df["Capacity_Ah"], color="red", s=12, zorder=3)
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Capacity (Ah)")
plt.title("Capacity vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Capacity_vs_Time.png", dpi=300)
plt.close()

# Plot 6: Voltage and Current vs Time (UTC) - dual y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Voltage on left axis
line1 = ax1.plot(
    dis_df["TIMESTAMP_UTC_DT"],
    dis_df["TOTAL_BATTERY_VOLTAGE"],
    label="Voltage",
    linewidth=2
)
scatter1 = ax1.scatter(
    dis_df["TIMESTAMP_UTC_DT"],
    dis_df["TOTAL_BATTERY_VOLTAGE"],
    color="red",
    s=12,
    zorder=3
)
ax1.set_xlabel("Timestamp (UTC)")
ax1.set_ylabel("Voltage (V)")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))

# Current on right axis
ax2 = ax1.twinx()
line2 = ax2.plot(
    dis_df["TIMESTAMP_UTC_DT"],
    dis_df["DISCHARGE_CURRENT"],
    label="Current",
    color= "black",
    linewidth=2,
    linestyle="--"
)
scatter2 = ax2.scatter(
    dis_df["TIMESTAMP_UTC_DT"],
    dis_df["DISCHARGE_CURRENT"],
    color="pink",
    s=12,
    zorder=3
)
ax2.set_ylabel("Discharge Current (A)")

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.title("Voltage and Current vs Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Voltage_Current_vs_Time.png", dpi=300)
plt.close()

print("Done.")
print("Created:")
print("1. EPS-Charge-Discharge-Processed.xlsx")
print("2. voltage_vs_mAh.png")
print("3. voltage_vs_Ah.png")
print("4. voltage_vs_Time.png")
print("5. Current_vs_Time.png")
print("6. Capacity_vs_Time.png")
print("7. Voltage_Current_vs_Time.png")