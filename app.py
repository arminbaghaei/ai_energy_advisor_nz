import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Energy Advisor (NZ Homes)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
CAT_MAPS = {
    "insulation_level": {"poor":0, "moderate":1, "good":2},
    "heating_system": {"resistive_heaters":0, "heat_pump":1, "gas_heater":2, "wood_burner":3},
    "window_glazing": {"single":0, "double":1, "triple":2},
    "air_tightness": {"leaky":0, "typical":1, "tight":2},
    "hot_water_system": {"electric_cylinder":0, "gas_instant":1, "heat_pump_water_heater":2}
}

INPUT_COLUMNS = [
    "climate_zone","floor_area_m2","building_age_years","insulation_level","heating_system",
    "window_glazing","air_tightness","occupancy","has_mechanical_ventilation",
    "hot_water_system","solar_pv_kw","electricity_price_nzd_per_kwh"
]

TARGET = "annual_energy_kwh"

# NZ Building Code H1 climate zones (friendly labels -> numeric code)
CLIMATE_ZONE_OPTIONS = [
    ("Zone 1 — Northland & Auckland", 1),
    ("Zone 2 — Waikato, Bay of Plenty, Gisborne, Taranaki", 2),
    ("Zone 3 — Lower North Island (Hawke’s Bay, Manawatū, Wairarapa, Wellington coastal)", 3),
    ("Zone 4 — Northern South Island (Nelson, Tasman, Marlborough, West Coast north)", 4),
    ("Zone 5 — Canterbury plains & coastal Otago", 5),
    ("Zone 6 — Central Plateau, Southern Alps, Queenstown-Lakes, Southland", 6),
]

def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in CAT_MAPS.items():
        out[col] = out[col].map(mapping)
    out["has_mechanical_ventilation"] = out["has_mechanical_ventilation"].astype(int)
    return out

@st.cache_data
def load_sample():
    # expects data/sample_nz_housing_energy.csv in repo
    return pd.read_csv("data/sample_nz_housing_energy.csv")

def train_model(df: pd.DataFrame):
    X = encode_df(df[INPUT_COLUMNS])
    y = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return model, {
        "r2": float(r2_score(y_te, preds)),
        "mae": float(mean_absolute_error(y_te, preds))
    }

def predict_one(model, features: dict) -> float:
    row = pd.DataFrame([features])[INPUT_COLUMNS]
    row_enc = encode_df(row)
    return float(model.predict(row_enc)[0])

def monthly_breakdown(annual_kwh: float, climate_zone: int) -> pd.DataFrame:
    shares = {
        1: np.array([0.06,0.06,0.07,0.07,0.08,0.09,0.10,0.10,0.09,0.09,0.09,0.10]),
        2: np.array([0.06,0.06,0.07,0.07,0.08,0.09,0.11,0.11,0.09,0.09,0.09,0.08]),
        3: np.array([0.06,0.06,0.07,0.07,0.08,0.10,0.12,0.12,0.09,0.09,0.09,0.05]),
        4: np.array([0.05,0.05,0.06,0.07,0.08,0.11,0.13,0.13,0.10,0.09,0.08,0.05]),
        5: np.array([0.05,0.05,0.06,0.07,0.08,0.12,0.14,0.14,0.10,0.08,0.07,0.04]),
        6: np.array([0.04,0.04,0.06,0.07,0.09,0.13,0.15,0.15,0.10,0.08,0.06,0.03]),
    }[climate_zone]
    shares = shares / shares.sum()
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return pd.DataFrame({"month": months, "kWh": (shares * annual_kwh).round(1)})

def measure_catalog():
    return [
        {"id":"upgrade_to_heat_pump","label":"Install a high-efficiency heat pump (space heating)","capex_range":"$3,500–$6,000","apply": lambda f: {**f, "heating_system":"heat_pump"}},
        {"id":"insulation_upgrade","label":"Upgrade insulation to 'good'","capex_range":"$2,000–$5,000","apply": lambda f: {**f, "insulation_level":"good"}},
        {"id":"draught_stopping","label":"Draught-stopping & air tightening","capex_range":"$200–$600","apply": lambda f: {**f, "air_tightness":"tight"}},
        {"id":"double_glazing","label":"Upgrade windows (single→double)","capex_range":"$8,000–$20,000","apply": lambda f: {**f, "window_glazing":"double" if f.get("window_glazing")=="single" else "triple"}},
        {"id":"pv_2kw","label":"Add 2 kW rooftop solar PV","capex_range":"$5,000–$7,000","apply": lambda f: {**f, "solar_pv_kw": f.get("solar_pv_kw",0)+2.0}},
        {"id":"hpwh","label":"Heat pump water heater","capex_range":"$3,500–$5,500","apply": lambda f: {**f, "hot_water_system":"heat_pump_water_heater"}},
        {"id":"smart_thermostat","label":"Smart thermostat & schedules (~5%)","capex_range":"$200–$400","apply": lambda f: {**f}},
    ]

def evaluate_measures(model, base_features: dict, base_kwh: float, e_price: float):
    recs = []
    for m in measure_catalog():
        new_features = m["apply"](base_features)
        new_kwh = predict_one(model, new_features)
        if m["id"] == "smart_thermostat":
            new_kwh *= 0.95
        annual_saving_kwh = max(0.0, base_kwh - new_kwh)
        annual_saving_nzd = annual_saving_kwh * e_price
        recs.append({
            "measure": m["label"],
            "capex_range": m["capex_range"],
            "annual_saving_kwh": round(annual_saving_kwh,1),
            "annual_saving_nzd": round(annual_saving_nzd,2),
        })
    recs.sort(key=lambda r: r["annual_saving_nzd"], reverse=True)
    return pd.DataFrame(recs)

# -----------------------------
# Load data / model
# -----------------------------
sample_df = load_sample()
model, scores = train_model(sample_df)

# =========================================
# Intro screen (shows once, then "Next →")
# =========================================
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.title("🏠 AI Energy Advisor for NZ Homes")
    st.subheader("What this tool does")
    st.markdown(
        """
- **Predicts** annual household electricity use and bill from key home details (NZ H1 climate zone, size, age, insulation, windows, airtightness, heating, hot water, PV).
- **Recommends** upgrades (e.g., heat pump, insulation, glazing, draught-stopping, HPWH, PV) ranked by **kWh** and **$ saved**.
- **Explains seasonality** with an estimated monthly breakdown.
- **Batch mode**: upload a CSV of homes to get predictions for all.
        """
    )
    with st.expander("How to use in 30 seconds"):
        st.markdown(
            """
1. Open the **sidebar** and choose your **NZ H1 climate zone** (by region name), floor area, and current systems.  
2. Go to **Predictor** to see energy & bill estimates.  
3. Open **Recommendations** to see savings and capex ranges.  
4. Use **Batch & Export** if you have a CSV of multiple homes.
            """
        )
    st.caption("Disclaimer: Prototype estimates only — always get quotes and professional advice for investment decisions.")
    if st.button("Next → Show the full features", type="primary"):
        st.session_state.started = True
        st.rerun()
    st.stop()

# -----------------------------
# Full App UI (after Next)
# -----------------------------
# -----------------------------
# Intro gate (shows once)
# -----------------------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.title("🏠 AI Energy Advisor for NZ Homes")
    st.subheader("What this tool does")
    st.markdown(
        """
- **Predicts** your home’s annual electricity use and bill using NZ-specific inputs (H1 climate zone, size, age, insulation, glazing, airtightness, heating, hot water, PV).
- **Recommends** retrofit options (heat pump, insulation, glazing, draught-stopping, HPWH, PV) ranked by **kWh** and **$** savings.
- **Explains seasonality** via an estimated monthly breakdown.
- **Batch mode** lets you upload a CSV of homes for bulk predictions.
        """
    )
    with st.expander("How to use (30-second guide)"):
        st.markdown(
            """
1. In the **sidebar**, pick your **NZ H1 climate zone** (by region name), floor area, and current systems.  
2. Open **Predictor** to see energy & bill estimates.  
3. Open **Recommendations** to view savings and typical capex ranges.  
4. Use **Batch & Export** if you have a CSV for many homes.
            """
        )
    st.caption("Disclaimer: Prototype estimates only — get quotes and professional advice for decisions.")
    if st.button("Next → Show the main features", type="primary"):
        st.session_state.started = True
        st.rerun()
    st.stop()  # don’t render the rest until Next is pressed

# -----------------------------
# UI (main app after Next)
# -----------------------------
st.title("🏠 AI Energy Advisor for NZ Homes")
st.caption("ML prediction + personalized retrofit advice (prototype)")


    # Friendly climate zone selector with tooltip; returns numeric 1–6
    selected_zone = st.selectbox(
        "NZBC H1 climate zone",
        options=CLIMATE_ZONE_OPTIONS,
        index=1,  # default Zone 2
        format_func=lambda opt: opt[0],
        help="Defined by the NZ Building Code (H1). Higher zone numbers = colder climates."
    )
    cz = selected_zone[1]  # numeric value for the model

    floor = st.number_input("Floor area (m²)", 40, 400, 140, 1)
    age = st.number_input("Building age (years)", 0, 140, 35, 1)
    ins = st.selectbox("Insulation level", ["poor","moderate","good"], index=1)
    glz = st.selectbox("Window glazing", ["single","double","triple"], index=1)
    tight = st.selectbox("Airtightness", ["leaky","typical","tight"], index=1)
    heat = st.selectbox("Main heating system", ["resistive_heaters","heat_pump","gas_heater","wood_burner"], index=0)
    occ = st.slider("Occupancy (people)", 1, 10, 3)
    mech = st.checkbox("Mechanical ventilation / HRV", value=False)
    hw = st.selectbox("Hot water system", ["electric_cylinder","gas_instant","heat_pump_water_heater"], index=0)
    pv = st.number_input("Rooftop solar PV (kW)", 0.0, 20.0, 0.0, 0.1)
    price = st.number_input("Electricity price (NZD/kWh)", 0.15, 0.60, 0.32, 0.01)

    features = {
        "climate_zone": int(cz),
        "floor_area_m2": float(floor),
        "building_age_years": int(age),
        "insulation_level": ins,
        "heating_system": heat,
        "window_glazing": glz,
        "air_tightness": tight,
        "occupancy": int(occ),
        "has_mechanical_ventilation": bool(mech),
        "hot_water_system": hw,
        "solar_pv_kw": float(pv),
        "electricity_price_nzd_per_kwh": float(price),
    }

    st.caption("Hint: 1=Auckland/Northland · 2=Upper NI · 3=Lower NI · 4=Top of SI · 5=Canterbury/coastal Otago · 6=Central Plateau/Southern Alps/Southland")
    if st.button("Back to intro"):
        st.session_state.started = False
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor","💡 Recommendations","💬 Advisor Chat","📦 Batch & Export"])

with tab1:
    st.subheader("Predicted annual electricity use")
    pred_kwh = predict_one(model, features)
    pred_cost = pred_kwh * features["electricity_price_nzd_per_kwh"]
    colA, colB, colC = st.columns(3)
    colA.metric("Annual energy", f"{pred_kwh:,.0f} kWh")
    colB.metric("Estimated annual bill", f"${pred_cost:,.0f}")
    colC.metric("Model MAE (test)", f"{scores['mae']:,.0f} kWh")

    st.divider()
    st.markdown("**Monthly breakdown (approx.)**")
    mb = monthly_breakdown(pred_kwh, features["climate_zone"])
    fig, ax = plt.subplots()
    ax.bar(mb["month"], mb["kWh"])
    ax.set_xlabel("Month"); ax.set_ylabel("kWh"); ax.set_title("Monthly Energy Use (Estimated)")
    st.pyplot(fig)

with tab2:
    st.subheader("Top retrofit & behavior recommendations")
    recs_df = evaluate_measures(model, features, pred_kwh, features["electricity_price_nzd_per_kwh"])
    st.dataframe(recs_df, use_container_width=True)
    st.caption("Payback depends on quotes and usage. Use this to shortlist measures.")

with tab3:
    st.subheader("Ask the advisor")
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role":"assistant","content": "Kia ora! Tell me your goals—lower bills, reduce damp/mould risk, or increase comfort?"}
        ]
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    user_msg = st.chat_input("Type your question")
    if user_msg:
        st.session_state.chat.append({"role":"user","content": user_msg})
        # build quick tips from current top 3 recommendations
        recs_df = evaluate_measures(model, features, pred_kwh, features["electricity_price_nzd_per_kwh"])
        tips = []
        top = recs_df.head(3).to_dict(orient="records")
        for t in top:
            tips.append(f"- {t['measure']} (~{t['annual_saving_kwh']:,.0f} kWh / ${t['annual_saving_nzd']:,.0f} per year)")
        if features["insulation_level"] in ["poor","moderate"]:
            tips.append("- Improve insulation levels; it reduces heating demand and improves comfort.")
        if features["window_glazing"] == "single":
            tips.append("- Upgrade from single to double glazing; combine with heavy curtains.")
        if features["heating_system"] != "heat_pump":
            tips.append("- Consider a modern heat pump for efficient heating and dehumidification.")
        if features["air_tightness"] == "leaky":
            tips.append("- Draught-proof doors & windows; maintain fresh air with trickle vents or HRV.")
        if features["solar_pv_kw"] < 1.0:
            tips.append("- Add rooftop PV if the roof has good sun exposure.")
        resp = [
            f"Based on your home (~{pred_kwh:,.0f} kWh/yr, ~${pred_cost:,.0f}/yr), try:",
            *tips,
            "Want me to sort by lowest likely capex?"
        ]
        st.session_state.chat.append({"role":"assistant","content": "\n".join(resp)})
        st.experimental_rerun()

with tab4:
    st.subheader("Batch predict & export")
    st.markdown("Upload a CSV with the columns shown in the data dictionary.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            user_df = pd.read_csv(up)
            missing = [c for c in INPUT_COLUMNS if c not in user_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                preds = []
                for _, r in user_df.iterrows():
                    feats = {k: r[k] for k in INPUT_COLUMNS}
                    k = predict_one(model, feats)
                    cost = k * float(r["electricity_price_nzd_per_kwh"])
                    preds.append({"annual_energy_kwh_pred": k, "annual_bill_nzd_pred": cost})
                pred_df = pd.concat([user_df.reset_index(drop=True), pd.DataFrame(preds)], axis=1)
                st.dataframe(pred_df.head(50), use_container_width=True)
                csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

st.divider()
st.caption("Disclaimer: Prototype estimates only. For investment decisions, seek professional advice and quotes.")
