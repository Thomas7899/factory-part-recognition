# Factory Part Recognition - Development Tasks
# Verwende mit: make <target>

.PHONY: help install train evaluate serve test clean docker-build docker-run

# Farben für Output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
NC     := \033[0m

help: ## Zeigt diese Hilfe an
	@echo "$(GREEN)Factory Part Recognition - Make Targets$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Installiert alle Dependencies
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installiert$(NC)"

train: ## Startet das Training mit verbesserter Pipeline
	python src/train_improved.py
	@echo "$(GREEN)✅ Training abgeschlossen$(NC)"

evaluate: ## Führt Evaluation auf Test-Set durch
	python src/evaluate.py
	@echo "$(GREEN)✅ Evaluation abgeschlossen$(NC)"

serve: ## Startet den FastAPI Server
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

serve-prod: ## Startet Server im Produktionsmodus
	uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

test: ## Führt Unit Tests aus
	python -m pytest tests/ -v

lint: ## Code Linting mit ruff
	ruff check src/ main.py
	ruff format --check src/ main.py

format: ## Code Formatierung
	ruff format src/ main.py

clean: ## Räumt temporäre Dateien auf
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache
	rm -rf logs/run_*
	@echo "$(GREEN)✅ Cleanup abgeschlossen$(NC)"

docker-build: ## Baut Docker Image
	docker build -t factory-part-recognition:latest .
	@echo "$(GREEN)✅ Docker Image gebaut$(NC)"

docker-run: ## Startet Docker Container
	docker-compose up -d
	@echo "$(GREEN)✅ Container gestartet: http://localhost:8000$(NC)"

docker-stop: ## Stoppt Docker Container
	docker-compose down

docker-logs: ## Zeigt Container Logs
	docker-compose logs -f

# Kombinierte Targets
setup: install ## Vollständiges Setup
	mkdir -p static logs models plots docs
	@echo "$(GREEN)✅ Projekt Setup abgeschlossen$(NC)"

all: train evaluate ## Training + Evaluation
	@echo "$(GREEN)✅ Pipeline abgeschlossen$(NC)"
