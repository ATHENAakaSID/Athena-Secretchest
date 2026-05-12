import pandas as pd
import numpy as np

from sklearn.isotonic import IsotonicRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# USER SETTINGS
# ============================================================
CAPACITY_EXCEL_PATH = "EPS_Capacity_Test_Processed_1.2V.xlsx"
LIGHTING_CSV_PATH = "Lighting.csv"

SHEET_NAME = "V_Wh"
VOLTAGE_COL = "TOTAL_BATTERY_VOLTAGE"
ENERGY_COL = "Energy_Wh"
AH_COL = "Discharged_Ah"

VOLTAGE_RESOLUTION = 0.1

# Your Excel Energy_Wh behaves like used/discharged energy.
# Therefore higher Energy_Wh / Ah means lower battery voltage.
ENERGY_AXIS_IS_DISCHARGED_ENERGY = True


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
# LIGHTING INTERSECTION BETWEEN PASS 1 END AND PASS 2 START
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
    Finds exact DIRECTSUN and UMBRA overlaps between:
        End time of ground pass 1 -> Start time of ground pass 2

    Time is only used to calculate the amount of energy gained/lost.
    The main battery curve is time-independent: Voltage vs discharged Ah.
    """
    window_start = parse_utc_time(pass_1_end_time_utc)
    window_end = parse_utc_time(pass_2_start_time_utc)

    if window_end <= window_start:
        raise ValueError("Pass 2 start time must be after Pass 1 end time.")

    df = pd.read_csv(lighting_csv_path)
    df[start_col] = pd.to_datetime(df[start_col], utc=True, errors="raise")
    df[end_col] = pd.to_datetime(df[end_col], utc=True, errors="raise")
    df[condition_col] = df[condition_col].astype(str).str.upper().str.strip()

    rows = []
    totals = {"DIRECTSUN": 0.0, "UMBRA": 0.0}

    for _, row in df.iterrows():
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
# CAPACITY TABLE: CONVERT Wh AXIS TO Ah AXIS
# ============================================================
def load_capacity_table(
    file_path,
    sheet_name=SHEET_NAME,
    voltage_col=VOLTAGE_COL,
    energy_col=ENERGY_COL,
):
    """Read the voltage-vs-Energy_Wh lookup table from Excel."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df[[voltage_col, energy_col]].dropna().copy()
    df[voltage_col] = df[voltage_col].astype(float)
    df[energy_col] = df[energy_col].astype(float)
    df = df.sort_values(energy_col).reset_index(drop=True)
    return df


def add_discharged_ah_axis(capacity_df, voltage_col=VOLTAGE_COL, energy_col=ENERGY_COL):
    """
    Convert cumulative discharged Energy_Wh to cumulative discharged Ah.

    Because the Excel file has Wh, not Ah, this derives Ah by integration:
        dAh = dWh / Vavg

    The resulting Ah axis is independent of time.
    It represents battery state/capacity removed, not elapsed duration.
    """
    df = capacity_df[[energy_col, voltage_col]].dropna().copy()
    df = df.sort_values(energy_col).reset_index(drop=True)

    energy = df[energy_col].to_numpy(dtype=float)
    voltage = df[voltage_col].to_numpy(dtype=float)

    if len(df) < 2:
        raise ValueError("Capacity table must contain at least two rows.")

    delta_wh = np.diff(energy)
    avg_voltage = (voltage[:-1] + voltage[1:]) / 2.0

    # Avoid division by zero or negative voltage problems.
    if np.any(avg_voltage <= 0):
        raise ValueError("Voltage values must be positive to derive Ah from Wh.")

    delta_ah = delta_wh / avg_voltage
    discharged_ah = np.concatenate([[0.0], np.cumsum(delta_ah)])

    df[AH_COL] = discharged_ah
    return df


# ============================================================
# ISOTONIC V vs Ah MODEL
# ============================================================
class IsotonicVoltageAhModel:
    """
    Time-independent monotonic battery model: Voltage = f(discharged Ah).

    The raw Excel table may have local noise, so direct interpolation can sometimes
    produce physically wrong direction. Isotonic regression removes local reversals.

    For this table:
        discharged Ah increases as the battery is discharged,
        so voltage must decrease as discharged Ah increases.
    Therefore isotonic regression uses increasing=False.
    """

    def __init__(self, capacity_ah_df, voltage_col=VOLTAGE_COL, ah_col=AH_COL, energy_col=ENERGY_COL):
        self.voltage_col = voltage_col
        self.ah_col = ah_col
        self.energy_col = energy_col

        clean = capacity_ah_df[[ah_col, voltage_col, energy_col]].dropna().copy()
        clean = clean.sort_values(ah_col).reset_index(drop=True)

        self.raw_ah = clean[ah_col].to_numpy(dtype=float)
        self.raw_voltage = clean[voltage_col].to_numpy(dtype=float)
        self.raw_energy = clean[energy_col].to_numpy(dtype=float)

        self.iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self.smooth_voltage = self.iso.fit_transform(self.raw_ah, self.raw_voltage)

        self.ah_min = float(np.min(self.raw_ah))
        self.ah_max = float(np.max(self.raw_ah))
        self.energy_min = float(np.min(self.raw_energy))
        self.energy_max = float(np.max(self.raw_energy))
        self.voltage_min = float(np.min(self.smooth_voltage))
        self.voltage_max = float(np.max(self.smooth_voltage))

        # Inverse Ah(V). Isotonic output can contain repeated voltages, so group.
        inverse_df = pd.DataFrame({
            "smooth_voltage": self.smooth_voltage,
            "ah": self.raw_ah,
        })
        inverse_df = inverse_df.groupby("smooth_voltage", as_index=False)["ah"].mean()
        inverse_df = inverse_df.sort_values("smooth_voltage").reset_index(drop=True)

        self.inverse_voltage = inverse_df["smooth_voltage"].to_numpy(dtype=float)
        self.inverse_ah = inverse_df["ah"].to_numpy(dtype=float)

    def ah_to_voltage(self, discharged_ah):
        discharged_ah = np.clip(discharged_ah, self.ah_min, self.ah_max)
        return float(self.iso.predict([discharged_ah])[0])

    def voltage_to_ah(self, voltage_v):
        voltage_v = np.clip(voltage_v, self.voltage_min, self.voltage_max)
        return float(np.interp(voltage_v, self.inverse_voltage, self.inverse_ah))

    def energy_to_ah(self, energy_wh):
        energy_wh = np.clip(energy_wh, self.energy_min, self.energy_max)
        return float(np.interp(energy_wh, self.raw_energy, self.raw_ah))

    def ah_to_energy(self, discharged_ah):
        discharged_ah = np.clip(discharged_ah, self.ah_min, self.ah_max)
        return float(np.interp(discharged_ah, self.raw_ah, self.raw_energy))

    def apply_energy_change(self, current_ah, condition, dt_seconds, charge_power_w, discharge_power_w):
        """
        Apply segment energy change while plotting in Ah.

        Power and time naturally give Wh:
            segment_wh = W * seconds / 3600

        To update the Ah position on the time-independent V-Ah curve:
            1. Convert current Ah to equivalent Wh-axis position.
            2. Apply charging/discharging Wh.
            3. Convert the new Wh-axis position back to Ah.

        DIRECTSUN charging:
            discharged Energy_Wh decreases, so discharged Ah decreases and voltage rises.
        UMBRA discharging:
            discharged Energy_Wh increases, so discharged Ah increases and voltage falls.
        """
        condition = str(condition).upper().strip()
        current_energy = self.ah_to_energy(current_ah)

        if condition == "DIRECTSUN":
            segment_wh = (dt_seconds / 3600.0) * charge_power_w
            new_energy = current_energy - segment_wh
        elif condition == "UMBRA":
            segment_wh = (dt_seconds / 3600.0) * discharge_power_w
            new_energy = current_energy + segment_wh
        else:
            segment_wh = 0.0
            new_energy = current_energy

        new_energy = float(np.clip(new_energy, self.energy_min, self.energy_max))
        new_ah = self.energy_to_ah(new_energy)
        return new_ah, new_energy, segment_wh


# ============================================================
# RESULT CALCULATION
# ============================================================
def calculate_voltage_result(capacity_df, model, input_voltage):
    """Calculate starting voltage band and corresponding starting Ah values."""
    band = capacity_df[
        (capacity_df[VOLTAGE_COL] >= input_voltage)
        & (capacity_df[VOLTAGE_COL] < input_voltage + VOLTAGE_RESOLUTION)
    ].copy()

    if band.empty:
        raise ValueError(
            f"No rows found in voltage band {input_voltage:.2f} V "
            f"to {input_voltage + VOLTAGE_RESOLUTION:.2f} V."
        )

    start_voltage_low = float(band[VOLTAGE_COL].min())
    start_voltage_high = float(band[VOLTAGE_COL].max())
    start_voltage_mid = float(input_voltage)

    return {
        "input_voltage": input_voltage,
        "voltage_band_low": input_voltage,
        "voltage_band_high": input_voltage + VOLTAGE_RESOLUTION,
        "number_of_matching_rows": int(len(band)),
        "start_voltage_low": start_voltage_low,
        "start_voltage_high": start_voltage_high,
        "start_voltage_mid": start_voltage_mid,
        "start_ah_low": model.voltage_to_ah(start_voltage_low),
        "start_ah_high": model.voltage_to_ah(start_voltage_high),
        "start_ah_mid": model.voltage_to_ah(start_voltage_mid),
    }


# ============================================================
# BUILD LOW / HIGH / INPUT PATHS ON V vs Ah CURVE
# ============================================================
def build_ah_profile(details_df, start_voltage, model, charge_power_w, discharge_power_w, label):
    """Build segment-by-segment profile on the time-independent V-Ah curve."""
    if details_df.empty:
        raise ValueError("No DIRECTSUN/UMBRA overlaps found inside the selected pass gap.")

    current_ah = model.voltage_to_ah(start_voltage)
    current_energy = model.ah_to_energy(current_ah)
    rows = []

    for segment_index, r in details_df.iterrows():
        condition = str(r["condition"]).upper().strip()
        dt = float(r["overlap_seconds"])

        ah0 = current_ah
        e0 = current_energy
        v0 = model.ah_to_voltage(ah0)

        current_ah, current_energy, segment_wh = model.apply_energy_change(
            current_ah=current_ah,
            condition=condition,
            dt_seconds=dt,
            charge_power_w=charge_power_w,
            discharge_power_w=discharge_power_w,
        )

        ah1 = current_ah
        e1 = current_energy
        v1 = model.ah_to_voltage(ah1)

        rows.append({
            "profile": label,
            "segment_index": int(segment_index + 1),
            "condition": condition,
            "duration_seconds": dt,
            "overlap_start_utc": r["overlap_start_utc"],
            "overlap_end_utc": r["overlap_end_utc"],
            "ah_start": ah0,
            "ah_end": ah1,
            "energy_start_wh": e0,
            "energy_end_wh": e1,
            "voltage_start_v": v0,
            "voltage_end_v": v1,
            "segment_wh": segment_wh,
            "signed_ah_delta": ah1 - ah0,
            "delta_voltage_v": v1 - v0,
        })

    return pd.DataFrame(rows)


def combine_ah_profiles(details_df, result, model, charge_power_w, discharge_power_w):
    low_df = build_ah_profile(details_df, result["start_voltage_low"], model, charge_power_w, discharge_power_w, "Lower bound")
    high_df = build_ah_profile(details_df, result["start_voltage_high"], model, charge_power_w, discharge_power_w, "Upper bound")
    input_df = build_ah_profile(details_df, result["input_voltage"], model, charge_power_w, discharge_power_w, "Input voltage")
    return pd.concat([low_df, high_df, input_df], ignore_index=True)


# ============================================================
# INTERACTIVE PLOT HELPERS
# ============================================================
def profile_ah_arrays(profile_df):
    xs, ys, text = [], [], []
    for _, row in profile_df.iterrows():
        hover_0 = (
            f"{row['profile']}<br>"
            f"Segment {row['segment_index']} - {row['condition']} start<br>"
            f"Ah: {row['ah_start']:.6f}<br>"
            f"Voltage: {row['voltage_start_v']:.4f} V<br>"
            f"Energy_Wh: {row['energy_start_wh']:.4f}"
        )
        hover_1 = (
            f"{row['profile']}<br>"
            f"Segment {row['segment_index']} - {row['condition']} end<br>"
            f"Ah: {row['ah_end']:.6f}<br>"
            f"Voltage: {row['voltage_end_v']:.4f} V<br>"
            f"Energy_Wh: {row['energy_end_wh']:.4f}"
        )
        xs.extend([row["ah_start"], row["ah_end"], None])
        ys.extend([row["voltage_start_v"], row["voltage_end_v"], None])
        text.extend([hover_0, hover_1, None])
    return xs, ys, text


def create_interactive_v_vs_ah_plot(capacity_ah_df, model, profile_all_df, output_html):
    """Main time-independent interactive V vs Ah plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=capacity_ah_df[AH_COL],
        y=capacity_ah_df[VOLTAGE_COL],
        mode="markers",
        name="Raw Excel-derived V vs Ah points",
        marker=dict(size=4, color="rgba(120,120,120,0.45)"),
        hovertemplate="Discharged Ah: %{x:.6f}<br>Voltage: %{y:.4f} V<extra>Raw table</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=model.raw_ah,
        y=model.smooth_voltage,
        mode="lines",
        name="Isotonic V vs Ah curve",
        line=dict(color="black", width=3),
        hovertemplate="Discharged Ah: %{x:.6f}<br>Isotonic Voltage: %{y:.4f} V<extra>Isotonic</extra>",
    ))

    styles = {
        "Lower bound": dict(color="blue", width=3, dash="solid"),
        "Upper bound": dict(color="red", width=3, dash="solid"),
        "Input voltage": dict(color="black", width=4, dash="dash"),
    }

    for profile_name in ["Lower bound", "Upper bound", "Input voltage"]:
        sub = profile_all_df[profile_all_df["profile"] == profile_name]
        x, y, text = profile_ah_arrays(sub)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=f"{profile_name} operating path",
            line=styles[profile_name],
            text=text,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Time-Independent Battery Curve: Voltage vs Discharged Ah",
        xaxis_title="Discharged Ah derived from Energy_Wh",
        yaxis_title="Battery Voltage (V)",
        hovermode="closest",
        template="plotly_white",
        legend_title="Curve / path",
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    return fig


def create_interactive_ah_segment_plot(profile_all_df, output_html):
    """
    Optional non-time plot that shows segment order vs Ah/Voltage.
    X-axis is segment number, not clock time.
    This helps identify which Sunlit/Umbra segment caused each Ah movement.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    styles = {
        "Lower bound": dict(color="blue", width=2, dash="solid"),
        "Upper bound": dict(color="red", width=2, dash="solid"),
        "Input voltage": dict(color="black", width=4, dash="dash"),
    }

    for profile_name in ["Lower bound", "Upper bound", "Input voltage"]:
        sub = profile_all_df[profile_all_df["profile"] == profile_name].copy()
        x = [0]
        ah = [sub.iloc[0]["ah_start"]]
        v = [sub.iloc[0]["voltage_start_v"]]
        conditions = ["Start"]

        for _, row in sub.iterrows():
            x.append(int(row["segment_index"]))
            ah.append(row["ah_end"])
            v.append(row["voltage_end_v"])
            conditions.append(row["condition"])

        fig.add_trace(go.Scatter(
            x=x,
            y=v,
            mode="lines+markers",
            name=f"{profile_name} voltage",
            line=styles[profile_name],
            text=conditions,
            hovertemplate="Segment: %{x}<br>Condition: %{text}<br>Voltage: %{y:.4f} V<extra></extra>",
        ), secondary_y=False)

        if profile_name == "Input voltage":
            fig.add_trace(go.Scatter(
                x=x,
                y=ah,
                mode="lines+markers",
                name="Input discharged Ah",
                line=dict(color="green", width=3, dash="dot"),
                text=conditions,
                hovertemplate="Segment: %{x}<br>Condition: %{text}<br>Discharged Ah: %{y:.6f}<extra></extra>",
            ), secondary_y=True)

    fig.update_layout(
        title="Segment-Ordered Voltage and Ah Movement, Not Clock Time",
        xaxis_title="Lighting segment number",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Battery Voltage (V)", secondary_y=False)
    fig.update_yaxes(title_text="Discharged Ah", secondary_y=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    return fig


# ============================================================
# SAFE INPUT
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
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n=== Voltage Calculator: Time-Independent Isotonic V vs Ah ===\n")

    input_voltage = get_float_input("Enter start voltage, example 27.7: ")

    print("\nEnter the comms gap timestamps from Possible passes.csv")
    pass_1_end_time = get_text_input("Enter END time of ground pass 1 UTC: ")
    pass_2_start_time = get_text_input("Enter START time of ground pass 2 UTC: ")

    charge_power_w = get_float_input("Enter charging power in W during DIRECTSUN, example 28: ")
    discharge_power_w = get_float_input("Enter discharging power in W during UMBRA, example 12: ")

    lighting = calculate_lighting_between_passes(
        lighting_csv_path=LIGHTING_CSV_PATH,
        pass_1_end_time_utc=pass_1_end_time,
        pass_2_start_time_utc=pass_2_start_time,
    )

    x = lighting["directsun_seconds"]
    y = lighting["umbra_seconds"]

    charging_energy_wh = (x / 3600.0) * charge_power_w
    discharging_energy_wh = (y / 3600.0) * discharge_power_w
    net_wh = charging_energy_wh - discharging_energy_wh

    capacity_df = load_capacity_table(CAPACITY_EXCEL_PATH)
    capacity_ah_df = add_discharged_ah_axis(capacity_df)
    model = IsotonicVoltageAhModel(capacity_ah_df)

    result = calculate_voltage_result(
        capacity_df=capacity_df,
        model=model,
        input_voltage=input_voltage,
    )

    details_df = lighting["details"]
    profile_all_df = combine_ah_profiles(
        details_df=details_df,
        result=result,
        model=model,
        charge_power_w=charge_power_w,
        discharge_power_w=discharge_power_w,
    )

    # Final values from the input-voltage path
    input_profile = profile_all_df[profile_all_df["profile"] == "Input voltage"]
    final_input_voltage = float(input_profile.iloc[-1]["voltage_end_v"])
    final_input_ah = float(input_profile.iloc[-1]["ah_end"])

    low_profile = profile_all_df[profile_all_df["profile"] == "Lower bound"]
    high_profile = profile_all_df[profile_all_df["profile"] == "Upper bound"]
    final_voltage_candidates = [
        float(low_profile.iloc[-1]["voltage_end_v"]),
        float(high_profile.iloc[-1]["voltage_end_v"]),
    ]

    final_voltage_low = min(final_voltage_candidates)
    final_voltage_high = max(final_voltage_candidates)

    # Save outputs
    capacity_ah_csv = "capacity_table_with_derived_ah.csv"
    profile_csv = "voltage_ah_profile_isotonic.csv"
    v_ah_html = "interactive_v_vs_ah_isotonic.html"
    segment_html = "interactive_segment_order_v_ah.html"

    capacity_ah_df.to_csv(capacity_ah_csv, index=False)
    profile_all_df.to_csv(profile_csv, index=False)
    create_interactive_v_vs_ah_plot(capacity_ah_df, model, profile_all_df, v_ah_html)
    create_interactive_ah_segment_plot(profile_all_df, segment_html)

    print("\n--- Time Window from Comms Passes ---")
    print(f"Window start, pass 1 end : {lighting['window_start_utc']}")
    print(f"Window end, pass 2 start : {lighting['window_end_utc']}")
    print(f"Total gap time           : {lighting['gap_seconds']:.3f} s")

    print("\n--- Automatic Lighting Intersection ---")
    print(f"Charging time in DIRECTSUN, x : {x:.3f} s")
    print(f"Discharging time in UMBRA, y  : {y:.3f} s")

    if not details_df.empty:
        print("\nDetailed overlap rows:")
        print(details_df.to_string(index=False))

    print("\n--- Energy Calculation ---")
    print(f"Charging Energy         : {charging_energy_wh:.3f} Wh")
    print(f"Discharging Energy      : {discharging_energy_wh:.3f} Wh")
    print(f"Net Energy              : {net_wh:.3f} Wh")

    if net_wh > 0:
        print("Net Charging: Power Positive, discharged Ah decreases, voltage increases.")
    elif net_wh < 0:
        print("Net Discharging: Power Negative, discharged Ah increases, voltage decreases.")
    else:
        print("Balanced: final voltage should remain approximately the same.")

    print("\n--- Time-Independent Isotonic V vs Ah Result ---")
    print(f"Requested voltage      : {result['input_voltage']:.3f} V")
    print(f"Voltage band used      : {result['voltage_band_low']:.3f} V to {result['voltage_band_high']:.3f} V")
    print(f"Matching rows          : {result['number_of_matching_rows']}")
    print(f"Start voltage range    : {result['start_voltage_low']:.3f} V to {result['start_voltage_high']:.3f} V")
    print(f"Final voltage range    : {final_voltage_low:.3f} V to {final_voltage_high:.3f} V")
    print(f"Input final voltage    : {final_input_voltage:.3f} V")
    print(f"Input final Ah         : {final_input_ah:.6f} Ah discharged")

    print("\n--- Interactive Outputs Created ---")
    print(f"1. {v_ah_html}")
    print("   Main time-independent interactive Voltage vs discharged Ah curve.")
    print(f"2. {segment_html}")
    print("   Segment-ordered voltage and Ah movement, not clock-time based.")
    print(f"3. {profile_csv}")
    print("   Segment-by-segment Ah, Wh, and voltage values for Lower/Input/Upper paths.")
    print(f"4. {capacity_ah_csv}")
    print("   Original capacity table plus derived discharged Ah axis.")
