# Phoenix AI E2E validation helper
$ErrorActionPreference = "Stop"

Write-Host "[1/4] Checking Python environment..."
python --version

Write-Host "[2/4] Installing evaluation dependencies..."
python -m pip install -r evaluation/requirements-evaluation.txt

Write-Host "[3/4] Running evaluator preflight..."
python evaluation/scripts/run_live_evaluation.py --preflight

Write-Host "[4/4] Running live E2E acceptance suite..."
python evaluation/scripts/run_live_evaluation.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "E2E suite reported failures. Inspect evaluation/results/latest.json and latest.md." -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host "E2E suite passed." -ForegroundColor Green
