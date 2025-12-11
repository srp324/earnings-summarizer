#!/bin/bash
# Quick setup script for the backend using uv

set -e

echo "🚀 Setting up Earnings Summarizer Backend with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✨ Creating virtual environment..."
uv venv

echo "📚 Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo "🎭 Installing Playwright browsers..."
playwright install chromium

echo "⚙️  Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Please edit .env and add your OPENAI_API_KEY"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the backend server:"
echo "  source .venv/bin/activate"
echo "  python run.py"
echo ""

