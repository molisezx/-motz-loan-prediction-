.PHONY: help install pipeline dashboard api test lint report slides clean

help:
	@echo "Targets:"
	@echo "  install    Install Python dependencies"
	@echo "  pipeline   Reproduce data, model, report and slides"
	@echo "  dashboard  Launch the Streamlit dashboard"
	@echo "  api        Launch the FastAPI service"
	@echo "  test       Run the test suite"
	@echo "  lint       Run ruff"
	@echo "  report     Regenerate the Word report"
	@echo "  slides     Regenerate the PowerPoint deck"
	@echo "  clean      Remove generated artefacts"

install:
	python -m pip install -r requirements.txt

pipeline:
	python run_pipeline.py

dashboard:
	streamlit run dashboard/app.py

api:
	uvicorn src.api:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check src dashboard

report:
	python src/build_report.py

slides:
	python src/build_presentation.py

clean:
	rm -rf reports/figures data/processed models/*.pkl
