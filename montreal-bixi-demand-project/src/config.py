from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

BIXI_ARCHIVES = {
    2025: "https://cdn.bixi.com/wp-content/uploads/2026/02/DonneesOuvertes2025_010203040506070809101112.zip",
    2024: "https://cdn.bixi.com/wp-content/uploads/2025/02/DonneesOuvertes2024_010203040506070809101112.zip",
}

for path in (RAW, INTERIM, PROCESSED, MODELS, REPORTS / "figures"):
    path.mkdir(parents=True, exist_ok=True)
