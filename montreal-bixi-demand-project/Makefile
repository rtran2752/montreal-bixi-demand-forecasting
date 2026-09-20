.PHONY: install demo data train dashboard test lint all

install:
	python -m pip install -r requirements-dev.txt

demo:
	python -m src.data.make_demo
	python -m src.models.train --data data/processed/hourly_features.parquet

data:
	python -m src.data.download --year $${BIXI_YEAR:-2025}
	python -m src.data.build_dataset --year $${BIXI_YEAR:-2025}

train:
	python -m src.models.train

dashboard:
	streamlit run app.py

test:
	pytest -q

lint:
	ruff check .

all: data train test

