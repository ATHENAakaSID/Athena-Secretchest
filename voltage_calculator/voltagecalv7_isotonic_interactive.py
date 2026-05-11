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

VOLTAGE_RESOLUTION = 0.1

# In your battery Excel table, Energy_Wh behaves like USED / DISCHARGED energy.
# Therefore:
#   DIRECTSUN charging  -> Energy_Wh decreases -> voltage increases
#   UMBRA discharging   -> Energy_Wh increases -> voltage decreases
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
# CAPACITY TABLE + ISOTONIC REGRESSION
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


class IsotonicVoltageModel:
    """
    Smooth monotonic battery model.

    The raw Excel table may have local noise, so direct interpolation can sometimes
    create a physically wrong direction. Isotonic regression removes those local
    reversals and forces voltage to be monotonic with Energy_Wh.

    For your table:
        Energy_Wh increases as battery is discharged, so V decreases as Energy_Wh increases.
        Therefore the isotonic fit uses increasing=False.
    """

    def __init__(self, capacity_df, voltage_col=VOLTAGE_COL, energy_col=ENERGY_COL):
        self.voltage_col = voltage_col
        self.energy_col = energy_col

        clean = capacity_df[[energy_col, voltage_col]].dropna().copy()
        clean = clean.sort_values(energy_col).reset_index(drop=True)

        self.raw_energy = clean[energy_col].to_numpy(dtype=float)
        self.raw_voltage = clean[voltage_col].to_numpy(dtype=float)

        # Isotonic V(E). Because Energy_Wh is discharged energy, V should decrease with Energy_Wh.
        self.iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self.smooth_voltage = self.iso.fit_transform(self.raw_energy, self.raw_voltage)

        self.energy_min = float(np.min(self.raw_energy))
        self.energy_max = float(np.max(self.raw_energy))
        self.voltage_min = float(np.min(self.smooth_voltage))
        self.voltage_max = float(np.max(self.smooth_voltage))

        # Build inverse E(V). Isotonic output can contain repeated voltage values,
        # so group by voltage and use the mean energy for a stable inverse lookup.
        inverse_df = pd.DataFrame({
            "smooth_voltage": self.smooth_voltage,
            "energy": self.raw_energy,
        })
        inverse_df = inverse_df.groupby("smooth_voltage", as_index=False)["energy"].mean()
        inverse_df = inverse_df.sort_values("smooth_voltage").reset_index(drop=True)

        self.inverse_voltage = inverse_df["smooth_voltage"].to_numpy(dtype=float)
        self.inverse_energy = inverse_df["energy"].to_numpy(dtype=float)

    def energy_to_voltage(self, energy_wh):
        energy_wh = np.clip(energy_wh, self.energy_min, self.energy_max)
        return float(self.iso.predict([energy_wh])[0])

    def voltage_to_energy(self, voltage_v):
        voltage_v = np.clip(voltage_v, self.voltage_min, self.voltage_max)
        return float(np.interp(voltage_v, self.inverse_voltage, self.inverse_energy))

    def apply_energy_change(self, current_energy_wh, condition, dt_seconds, charge_power_w, discharge_power_w):
        """
        Apply segment energy change.

        DIRECTSUN:
            Charging happens. Since Energy_Wh is discharged energy, subtract Wh.
        UMBRA:
            Discharging happens. Since Energy_Wh is discharged energy, add Wh.
        """
        if condition == "DIRECTSUN":
            delta_wh = (dt_seconds / 3600.0) * charge_power_w
            new_energy = current_energy_wh - delta_wh
        elif condition == "UMBRA":
            delta_wh = (dt_seconds / 3600.0) * discharge_power_w
            new_energy = current_energy_wh + delta_wh
        else:
            delta_wh = 0.0
            new_energy = current_energy_wh

        new_energy = float(np.clip(new_energy, self.energy_min, self.energy_max))
        return new_energy, delta_wh


# ============================================================
# VOLTAGE RANGE CALCULATION
# ============================================================
def calculate_voltage_result(capacity_df, model, input_voltage, net_wh):
    """Calculate start and final voltage range using isotonic model."""
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

    # Use the monotonic isotonic inverse model for starting energies.
    e_low = model.voltage_to_energy(start_voltage_low)
    e_high = model.voltage_to_energy(start_voltage_high)
    e_mid = model.voltage_to_energy(start_voltage_mid)

    # Since Energy_Wh is discharged energy:
    # positive net_wh -> charging -> Energy_Wh decreases -> voltage increases
    # negative net_wh -> discharging -> Energy_Wh increases -> voltage decreases
    final_e_low = np.clip(e_low - net_wh, model.energy_min, model.energy_max)
    final_e_high = np.clip(e_high - net_wh, model.energy_min, model.energy_max)
    final_e_mid = np.clip(e_mid - net_wh, model.energy_min, model.energy_max)

    final_v_low_path = model.energy_to_voltage(final_e_low)
    final_v_high_path = model.energy_to_voltage(final_e_high)
    final_v_mid = model.energy_to_voltage(final_e_mid)

    final_voltage_low = min(final_v_low_path, final_v_high_path)
    final_voltage_high = max(final_v_low_path, final_v_high_path)

    return {
        "input_voltage": input_voltage,
        "voltage_band_low": input_voltage,
        "voltage_band_high": input_voltage + VOLTAGE_RESOLUTION,
        "number_of_matching_rows": int(len(band)),
        "start_voltage_low": start_voltage_low,
        "start_voltage_high": start_voltage_high,
        "start_voltage_mid": start_voltage_mid,
        "start_energy_low_wh": e_low,
        "start_energy_high_wh": e_high,
        "start_energy_mid_wh": e_mid,
        "net_wh": float(net_wh),
        "final_energy_low_path_wh": float(final_e_low),
        "final_energy_high_path_wh": float(final_e_high),
        "final_energy_mid_wh": float(final_e_mid),
        "final_voltage_low": float(final_voltage_low),
        "final_voltage_high": float(final_voltage_high),
        "final_voltage_mid": float(final_v_mid),
    }


# ============================================================
# BUILD PROFILES FOR LOW / HIGH / INPUT VOLTAGE
# ============================================================
def build_profile(details_df, start_voltage, model, charge_power_w, discharge_power_w, label):
    """Build segment-by-segment voltage and Energy_Wh profile for one starting voltage."""
    if details_df.empty:
        raise ValueError("No DIRECTSUN/UMBRA overlaps found inside the selected pass gap.")

    current_energy = model.voltage_to_energy(start_voltage)
    rows = []

    for _, r in details_df.iterrows():
        condition = str(r["condition"]).upper().strip()
        t0 = r["overlap_start_utc"]
        t1 = r["overlap_end_utc"]
        dt = float(r["overlap_seconds"])

        e0 = current_energy
        v0 = model.energy_to_voltage(e0)

        current_energy, segment_delta_wh = model.apply_energy_change(
            current_energy_wh=current_energy,
            condition=condition,
            dt_seconds=dt,
            charge_power_w=charge_power_w,
            discharge_power_w=discharge_power_w,
        )

        e1 = current_energy
        v1 = model.energy_to_voltage(e1)

        rows.append({
            "profile": label,
            "condition": condition,
            "start_time_utc": t0,
            "end_time_utc": t1,
            "duration_seconds": dt,
            "energy_start_wh": e0,
            "energy_end_wh": e1,
            "voltage_start_v": v0,
            "voltage_end_v": v1,
            "segment_energy_wh": segment_delta_wh,
            "signed_energy_axis_delta_wh": e1 - e0,
            "delta_voltage_v": v1 - v0,
        })

    return pd.DataFrame(rows)


def combine_profiles(details_df, result, model, charge_power_w, discharge_power_w):
    low_df = build_profile(
        details_df, result["start_voltage_low"], model,
        charge_power_w, discharge_power_w, "Lower bound"
    )
    high_df = build_profile(
        details_df, result["start_voltage_high"], model,
        charge_power_w, discharge_power_w, "Upper bound"
    )
    input_df = build_profile(
        details_df, result["input_voltage"], model,
        charge_power_w, discharge_power_w, "Input voltage"
    )
    return pd.concat([low_df, high_df, input_df], ignore_index=True)


# ============================================================
# INTERACTIVE PLOT HELPERS
# ============================================================
def profile_line_arrays(profile_df):
    """Convert segment rows into continuous x/y arrays with gaps between segments."""
    xs, ys = [], []
    for _, row in profile_df.iterrows():
        xs.extend([row["start_time_utc"], row["end_time_utc"], None])
        ys.extend([row["voltage_start_v"], row["voltage_end_v"], None])
    return xs, ys


def profile_wh_arrays(profile_df):
    xs, ys = [], []
    for _, row in profile_df.iterrows():
        xs.extend([row["energy_start_wh"], row["energy_end_wh"], None])
        ys.extend([row["voltage_start_v"], row["voltage_end_v"], None])
    return xs, ys


def add_lighting_rectangles(fig, details_df, yref="paper"):
    """Add DirectSun/Umbra background shading to time plots."""
    for _, row in details_df.iterrows():
        condition = str(row["condition"]).upper().strip()
        fill = "rgba(255, 165, 0, 0.16)" if condition == "DIRECTSUN" else "rgba(120, 120, 120, 0.20)"
        fig.add_vrect(
            x0=row["overlap_start_utc"],
            x1=row["overlap_end_utc"],
            fillcolor=fill,
            opacity=1.0,
            line_width=0,
            layer="below",
            annotation_text=condition,
            annotation_position="top left",
        )


def create_interactive_voltage_time_plot(profile_all_df, details_df, output_html):
    fig = go.Figure()
    add_lighting_rectangles(fig, details_df)

    styles = {
        "Lower bound": dict(color="blue", width=2, dash="solid"),
        "Upper bound": dict(color="red", width=2, dash="solid"),
        "Input voltage": dict(color="black", width=4, dash="dash"),
    }

    for profile_name in ["Lower bound", "Upper bound", "Input voltage"]:
        sub = profile_all_df[profile_all_df["profile"] == profile_name]
        x, y = profile_line_arrays(sub)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=profile_name,
            line=styles[profile_name],
            hovertemplate=(
                "Time: %{x}<br>"
                "Voltage: %{y:.4f} V"
                "<extra>" + profile_name + "</extra>"
            ),
        ))

    fig.update_layout(
        title="Interactive Battery Voltage vs Time with DirectSun/Umbra",
        xaxis_title="UTC Time",
        yaxis_title="Battery Voltage (V)",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Voltage profile",
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    return fig


def create_interactive_v_vs_wh_plot(capacity_df, model, profile_all_df, output_html):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=model.raw_energy,
        y=model.raw_voltage,
        mode="markers",
        name="Raw Excel points",
        marker=dict(size=4, color="rgba(120,120,120,0.45)"),
        hovertemplate="Energy_Wh: %{x:.4f}<br>Voltage: %{y:.4f} V<extra>Raw Excel</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=model.raw_energy,
        y=model.smooth_voltage,
        mode="lines",
        name="Isotonic V vs Wh curve",
        line=dict(color="black", width=3),
        hovertemplate="Energy_Wh: %{x:.4f}<br>Isotonic Voltage: %{y:.4f} V<extra>Isotonic</extra>",
    ))

    styles = {
        "Lower bound": dict(color="blue", width=3, dash="solid"),
        "Upper bound": dict(color="red", width=3, dash="solid"),
        "Input voltage": dict(color="black", width=4, dash="dash"),
    }

    for profile_name in ["Lower bound", "Upper bound", "Input voltage"]:
        sub = profile_all_df[profile_all_df["profile"] == profile_name]
        x, y = profile_wh_arrays(sub)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=f"{profile_name} path",
            line=styles[profile_name],
            hovertemplate=(
                "Energy_Wh: %{x:.4f}<br>"
                "Voltage: %{y:.4f} V"
                "<extra>" + profile_name + "</extra>"
            ),
        ))

    fig.update_layout(
        title="Interactive V vs Energy_Wh with Isotonic Curve and Operating Path",
        xaxis_title="Energy_Wh from Excel table",
        yaxis_title="Battery Voltage (V)",
        hovermode="closest",
        template="plotly_white",
        legend_title="Curve / path",
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    return fig


def create_interactive_isotonic_time_plot(profile_all_df, details_df, output_html):
    """Dedicated isotonic-smoothed time plot for the input-voltage path."""
    input_profile = profile_all_df[profile_all_df["profile"] == "Input voltage"].copy()
    fig = go.Figure()
    add_lighting_rectangles(fig, details_df)

    x, y = profile_line_arrays(input_profile)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name="Input voltage isotonic path",
        line=dict(color="black", width=4),
        hovertemplate="Time: %{x}<br>Voltage: %{y:.4f} V<extra>Input path</extra>",
    ))

    fig.update_layout(
        title="Interactive Isotonic Voltage Path for Input Voltage",
        xaxis_title="UTC Time",
        yaxis_title="Battery Voltage (V)",
        hovermode="x unified",
        template="plotly_white",
    )
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
    print("\n=== Voltage Calculator: Isotonic Regression + Interactive Curves ===\n")

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
    model = IsotonicVoltageModel(capacity_df)

    result = calculate_voltage_result(
        capacity_df=capacity_df,
        model=model,
        input_voltage=input_voltage,
        net_wh=net_wh,
    )

    details_df = lighting["details"]
    profile_all_df = combine_profiles(
        details_df=details_df,
        result=result,
        model=model,
        charge_power_w=charge_power_w,
        discharge_power_w=discharge_power_w,
    )

    profile_csv = "voltage_time_profile_isotonic.csv"
    profile_all_df.to_csv(profile_csv, index=False)

    voltage_time_html = "interactive_voltage_vs_time.html"
    v_wh_html = "interactive_v_vs_wh_isotonic.html"
    isotonic_time_html = "interactive_input_isotonic_voltage_path.html"

    create_interactive_voltage_time_plot(profile_all_df, details_df, voltage_time_html)
    create_interactive_v_vs_wh_plot(capacity_df, model, profile_all_df, v_wh_html)
    create_interactive_isotonic_time_plot(profile_all_df, details_df, isotonic_time_html)

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
        print("Net Charging: Power Positive, final voltage must increase.")
    elif net_wh < 0:
        print("Net Discharging: Power Negative, final voltage must decrease.")
    else:
        print("Balanced: final voltage should remain approximately the same.")

    print("\n--- Isotonic Voltage Result ---")
    print(f"Requested voltage      : {result['input_voltage']:.3f} V")
    print(f"Voltage band used      : {result['voltage_band_low']:.3f} V to {result['voltage_band_high']:.3f} V")
    print(f"Matching rows          : {result['number_of_matching_rows']}")
    print(f"Start voltage range    : {result['start_voltage_low']:.3f} V to {result['start_voltage_high']:.3f} V")
    print(f"Final voltage range    : {result['final_voltage_low']:.3f} V to {result['final_voltage_high']:.3f} V")
    print(f"Input final voltage    : {result['final_voltage_mid']:.3f} V")

    print("\n--- Interactive Outputs Created ---")
    print(f"1. {voltage_time_html}")
    print("   Interactive voltage vs time with DirectSun/Umbra shaded regions.")
    print(f"2. {v_wh_html}")
    print("   Interactive raw Excel points + isotonic V vs Wh curve + operating path.")
    print(f"3. {isotonic_time_html}")
    print("   Interactive input-voltage isotonic path vs time.")
    print(f"4. {profile_csv}")
    print("   Segment-by-segment voltage and Energy_Wh values for Lower/Input/Upper paths.")
