# 🏠 Your Home Energy Advisor

ML-based household energy **prediction** + **personalized retrofit advice** for New Zealand homes.

## 🚀 Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud
1. Push this folder to GitHub
2. Streamlit Cloud → New app → select repo → app file: `app.py` → Deploy

## 📦 Data
`data/sample_nz_housing_energy.csv` is a small synthetic NZ dataset used to fit a baseline model.
Use the **Batch & Export** tab to upload your own CSV for many homes.

## 🔧 Settings / Secrets (optional)
In Streamlit Cloud, set **Settings → Secrets** (e.g. emissions factor):
```
MFE_ELECTRICITY_EF_KG_PER_KWH = 0.12
```

## 📝 License
MIT
