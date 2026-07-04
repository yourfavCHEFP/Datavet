# Streamlit Deployment Guide

## Local Run

1. Double-click `start.bat`.
2. Open `http://localhost:8501`.

## Streamlit Community Cloud Deployment

1. Push this project to a GitHub repository.
2. In Streamlit Cloud, create a new app from that repository.
3. Set app entrypoint to `streamlit_app.py`.
4. Add Python dependencies from `requirements.txt` automatically.
5. Add optional secrets in Streamlit Cloud settings:
   - `kaggle.username`
   - `kaggle.key`
   - `gemini_proxy` (optional baseline pinning API)
   - `kimi_proxy` (optional baseline pinning API)
6. Deploy.

## One-Click Automation Pipeline

1. Enable GitHub Actions on your repository.
2. Keep `.github/workflows/ci-refresh.yml` committed.
3. Add GitHub repository secrets (optional, for Kaggle-enhanced refresh):
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
   - `DATAVET_GEMINI_PROXY` (optional)
   - `DATAVET_KIMI_PROXY` (optional)
4. Use **Actions > CI and Scheduled Refresh > Run workflow** to trigger checks and refresh manually.
5. Scheduled refresh runs every 6 hours and commits `.datavet` snapshots automatically.

## CI Checks Included

- Dependency install on Python 3.12
- Syntax compile checks for `streamlit_app.py`
- Syntax compile checks for `scripts/run_refresh_jobs.py`

## Notes

- Scheduled refresh jobs are persisted in `.datavet/refresh_jobs.json`.
- Refresh snapshots are saved in `.datavet/refresh_snapshots/`.
- For persistent cloud state, set `DATAVET_DATABASE_URL` for PostgreSQL-backed key/value persistence.
- For snapshot archival, set `DATAVET_S3_BUCKET` and optional `DATAVET_S3_PREFIX`.
- Ranking controls and Gemini/Kimi baseline pinning are available in the app under "Ranking Controls and External AI Baseline".
