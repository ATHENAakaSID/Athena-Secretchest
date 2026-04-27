# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# FILE = "EPS Charge Discharge Test.xlsx"

# def load_sheet(sheet_name, header_mode="normal"):
#     if header_mode == "normal":
#         df = pd.read_excel(FILE, sheet_name=sheet_name)
#     else:
#         df = pd.read_excel(FILE, sheet_name=sheet_name, header=None)
#         df.columns = ["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "TOTAL_BATTERY_CURRENT"]

#     df = df.iloc[:, :len(df.columns)].copy()
#     df.columns = [str(c).strip() for c in df.columns]

#     if "TIMESTAMP" not in df.columns:
#         raise ValueError(f"{sheet_name}: TIMESTAMP column not found")

#     if "TOTAL_BATTERY_VOLTAGE" not in df.columns:
#         raise ValueError(f"{sheet_name}: TOTAL_BATTERY_VOLTAGE column not found")

#     if "TOTAL_BATTERY_CURRENT" not in df.columns:
#         raise ValueError(f"{sheet_name}: TOTAL_BATTERY_CURRENT column not found")

#     df = df[["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "TOTAL_BATTERY_CURRENT"]].copy()

#     df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
#     df["TOTAL_BATTERY_VOLTAGE"] = pd.to_numeric(df["TOTAL_BATTERY_VOLTAGE"], errors="coerce")
#     df["TOTAL_BATTERY_CURRENT"] = pd.to_numeric(df["TOTAL_BATTERY_CURRENT"], errors="coerce")

#     df = df.dropna(subset=["TIMESTAMP", "TOTAL_BATTERY_VOLTAGE", "TOTAL_BATTERY_CURRENT"])
#     df = df.sort_values("TIMESTAMP").reset_index(drop=True)

#     df["TIMESTAMP_UTC"] = pd.to_datetime(df["TIMESTAMP"], unit="s", utc=True)

#     df["dt_sec"] = df["TIMESTAMP"].diff().fillna(0)
#     df.loc[df["dt_sec"] < 0, "dt_sec"] = 0

#     # If current is in A, mAh increment = A * sec / 3600 * 1000
#     df["d_mAh"] = df["TOTAL_BATTERY_CURRENT"] * df["dt_sec"] / 3600.0 * 1000.0
#     df["Capacity_mAh"] = df["d_mAh"].cumsum()
#     df["Capacity_Ah"] = df["Capacity_mAh"] / 1000.0

#     return df

# dis_df = load_sheet("Discharging", header_mode="normal")
# chg_df = load_sheet("Charging", header_mode="no_header")

# # Save processed sheets with UTC and capacity columns
# with pd.ExcelWriter("EPS-Charge-Discharge-Processed.xlsx", engine="openpyxl") as writer:
#     dis_df.to_excel(writer, sheet_name="Discharging_Processed", index=False)
#     chg_df.to_excel(writer, sheet_name="Charging_Processed", index=False)

# # Plot 1: Voltage vs mAh
# plt.figure(figsize=(10, 6))
# plt.plot(dis_df["Capacity_mAh"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
# plt.plot(chg_df["Capacity_mAh"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
# plt.xlabel("Capacity (mAh)")
# plt.ylabel("Voltage (V)")
# plt.title("Voltage vs mAh")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("voltage_vs_mAh.png", dpi=300)

# # Plot 2: Voltage vs Ah
# plt.figure(figsize=(10, 6))
# plt.plot(dis_df["Capacity_Ah"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
# plt.plot(chg_df["Capacity_Ah"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
# plt.xlabel("Capacity (Ah)")
# plt.ylabel("Voltage (V)")
# plt.title("Voltage vs Ah")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("voltage_vs_Ah.png", dpi=300)

# print("Done.")
# print("Created:")
# print("1. EPS-Charge-Discharge-Processed.xlsx")
# print("2. voltage_vs_mAh.png")
# print("3. voltage_vs_Ah.png")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    # Create UTC time as STRING so Excel can save it safely
    df["TIMESTAMP_UTC"] = (
        pd.to_datetime(df["TIMESTAMP"], unit="s", utc=True)
        .dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    df["dt_sec"] = df["TIMESTAMP"].diff().fillna(0)
    df.loc[df["dt_sec"] < 0, "dt_sec"] = 0

    # If current is in A, mAh increment = A * sec / 3600 * 1000
    df["d_mAh"] = df["DISCHARGE_CURRENT"] * df["dt_sec"] / 3600.0 * 1000.0
    df["Capacity_mAh"] = df["d_mAh"].cumsum()
    df["Capacity_Ah"] = df["Capacity_mAh"] / 1000.0

    return df


dis_df = load_sheet("Discharging", header_mode="normal")
# chg_df = load_sheet("Charging", header_mode="no_header")

# Save processed sheets
with pd.ExcelWriter("EPS-Charge-Discharge-Processed.xlsx", engine="openpyxl") as writer:
    dis_df.to_excel(writer, sheet_name="Discharging_Processed", index=False)
    # chg_df.to_excel(writer, sheet_name="Charging_Processed", index=False)

# Plot 1: Voltage vs mAh
# plt.figure(figsize=(10, 6))
# plt.plot(dis_df["Capacity_mAh"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
# plt.plot(x, y, color="black", linewidth=1)
# plt.scatter(x, y, color="red", s=10)
# # plt.plot(chg_df["Capacity_mAh"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
# plt.xlabel("Capacity (mAh)")
# plt.ylabel("Voltage (V)")
# plt.title("Voltage vs mAh")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("voltage_vs_mAh.png", dpi=300)
# plt.close()

plt.figure(figsize=(10, 6))
plt.plot(dis_df["Capacity_mAh"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
# plt.plot(chg_df["Capacity_mAh"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)

# Red dots for all plotted data points
plt.scatter(dis_df["Capacity_mAh"], dis_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
# plt.scatter(chg_df["Capacity_mAh"], chg_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)

plt.xlabel("Capacity (mAh)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs mAh")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("voltage_vs_mAh.png", dpi=300)
plt.close()


# Plot 2: Voltage vs Ah
# plt.figure(figsize=(10, 6))
# plt.plot(dis_df["Capacity_Ah"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
# # plt.plot(chg_df["Capacity_Ah"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)
# plt.xlabel("Capacity (Ah)")
# plt.ylabel("Voltage (V)")
# plt.title("Voltage vs Ah")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("voltage_vs_Ah.png", dpi=300)
# plt.close()

# Plot 2: Voltage vs Ah
plt.figure(figsize=(10, 6))
plt.plot(dis_df["Capacity_Ah"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
# /plt.plot(chg_df["Capacity_Ah"], chg_df["TOTAL_BATTERY_VOLTAGE"], label="Charging", linewidth=2)

# Red dots for all plotted data points
plt.scatter(dis_df["Capacity_Ah"], dis_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)
# plt.scatter(chg_df["Capacity_Ah"], chg_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)

plt.xlabel("Capacity (Ah)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Ah")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("voltage_vs_Ah.png", dpi=300)
plt.close()

# plot 3: Voltage vs Time(UTC)
plt.figure(figsize=(10, 6))
plt.plot(dis_df["TIMESTAMP_UTC"], dis_df["TOTAL_BATTERY_VOLTAGE"], label="Discharging", linewidth=2)
plt.scatter(dis_df["TIMESTAMP_UTC"], dis_df["TOTAL_BATTERY_VOLTAGE"], color="red", s=12, zorder=3)

plt.xlabel("TIMESTAMP (UTC)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("voltage_vs_Time.png", dpi=300)
plt.close()

# plot 4: A vs Time(UTC)
plt.figure(figsize=(10, 6))
plt.plot(dis_df["TIMESTAMP_UTC"], dis_df["DISCHARGE_CURRENT"], label="Discharging", linewidth=2)
plt.scatter(dis_df["TIMESTAMP_UTC"], dis_df["DISCHARGE_CURRENT"], color="red", s=12, zorder=3)

plt.xlabel("TIMESTAMP (UTC)")
plt.ylabel("DISCHARGE CURRENT (A)")
plt.title("Current vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Current_vs_Time.png", dpi=300)
plt.close()

# plt.figure(figsize=(10, 6))
# plot 5: Ah vs Time(UTC)
plt.figure(figsize=(10, 6))
plt.plot(dis_df["TIMESTAMP_UTC"], dis_df["Capacity_Ah"], label="Discharging", linewidth=2)
plt.scatter(dis_df["TIMESTAMP_UTC"], dis_df["Capacity_Ah"], color="red", s=12, zorder=3)

plt.xlabel("TIMESTAMP (UTC)")
plt.ylabel("Capacity (Ah)")
plt.title("Capacity vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Capacity_vs_Time.png", dpi=300)
plt.close()

# plt.figure(figsize=(10, 6))


print("Done.")
print("Created:")
print("1. EPS-Charge-Discharge-Processed.xlsx")
print("2. voltage_vs_mAh.png")
print("3. voltage_vs_Ah.png")
print("4. voltage_vs_Time.png")
print("5. Current_vs_Time.png")
print("6. Ah vs Time(UTC)")