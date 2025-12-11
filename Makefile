.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend docker-up docker-down clean

help:
	@echo "🚀 Earnings Summarizer - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install all dependencies (backend + frontend)"
	@echo "  make install-backend  - Install backend with uv"
	@echo "  make install-frontend - Install frontend with npm"
	@echo ""
	@echo "Development:"
	@echo "  make dev             - Run both backend and frontend"
	@echo "  make dev-backend     - Run backend only"
	@echo "  make dev-frontend    - Run frontend only"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up       - Start all services with Docker Compose"
	@echo "  make docker-down     - Stop all services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           - Remove virtual environments and caches"

install: install-backend install-frontend
	@echo "✅ All dependencies installed!"

install-backend:
	@echo "📦 Installing backend dependencies with uv..."
	@cd backend && uv venv && uv pip install -r requirements.txt
	@cd backend && . .venv/bin/activate && playwright install chromium
	@echo "✅ Backend dependencies installed!"

install-frontend:
	@echo "📦 Installing frontend dependencies..."
	@cd frontend && npm install
	@echo "✅ Frontend dependencies installed!"

dev:
	@echo "🚀 Starting development servers..."
	@make -j2 dev-backend dev-frontend

dev-backend:
	@echo "🐍 Starting backend server..."
	@cd backend && . .venv/bin/activate && python run.py

dev-frontend:
	@echo "⚛️  Starting frontend server..."
	@cd frontend && npm run dev

docker-up:
	@echo "🐳 Starting Docker containers..."
	@docker-compose up --build

docker-down:
	@echo "🛑 Stopping Docker containers..."
	@docker-compose down

clean:
	@echo "🧹 Cleaning up..."
	@rm -rf backend/.venv backend/__pycache__ backend/**/__pycache__
	@rm -rf frontend/node_modules frontend/dist
	@rm -rf backend/.pytest_cache backend/.ruff_cache
	@echo "✅ Cleanup complete!"

