# Quick setup script for the backend using uv (Windows PowerShell)

Write-Host "🚀 Setting up Earnings Summarizer Backend with uv..." -ForegroundColor Green

# Check if uv is installed
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing uv..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
}

Write-Host "✨ Creating virtual environment..." -ForegroundColor Cyan
uv venv

Write-Host "📚 Installing dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

Write-Host "🎭 Installing Playwright browsers..." -ForegroundColor Cyan
playwright install chromium

Write-Host "⚙️  Setting up environment..." -ForegroundColor Cyan
if (!(Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "📝 Please edit .env and add your OPENAI_API_KEY" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the backend server:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  python run.py" -ForegroundColor Gray
Write-Host ""

