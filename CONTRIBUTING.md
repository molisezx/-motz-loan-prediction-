# Contributing

Thanks for your interest in the Motz Loan Prediction project.

## Development setup

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest ruff
python run_pipeline.py     # reproduce data, model, report, slides
```

## Workflow

1. Create a feature branch: `git checkout -b feature/my-change`.
2. Make your change; keep modules small and documented.
3. Run the checks before committing:

   ```powershell
   ruff check src dashboard
   pytest -q
   python run_pipeline.py
   ```

4. Open a pull request describing the change and the motivation.

## Code style

- Python 3.10+, 120-column lines, ruff-clean.
- No comments unless they add information the code cannot express.
- Every pipeline artefact must be reproducible from a fixed random seed.

## Reporting issues

Please include the command you ran, the full traceback, and the output of
`python --version` and `pip freeze`.
