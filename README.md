# AMR Early Warning System
Forecasts antibiotic resistance (E. coli vs ciprofloxacin demo) and raises Green/Amber/Red alerts.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Deploy (free)
1. Push these 4 files to a public GitHub repo.
2. Go to share.streamlit.io, connect the repo, set main file to app.py, click Deploy.

## Your CSV format
date,tested,resistant   (one row per month, e.g. 2024-01-01,120,38)
