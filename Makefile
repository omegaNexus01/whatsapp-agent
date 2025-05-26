ifeq (,$(wildcard .env))
$(error .env file is missing. Please create one based on .env.example)
endif

include .env

CHECK_DIRS := .

ava-build:
	docker compose build

ava-run:
	docker compose up --build -d

ava-stop:
	docker compose stop

ava-delete:
	@if [ -d "long_term_memory" ]; then rm -rf long_term_memory; fi
	@if [ -d "short_term_memory" ]; then rm -rf short_term_memory; fi
	@if [ -d "generated_images" ]; then rm -rf generated_images; fi
	docker compose down

format-fix:
	uv run ruff format $(CHECK_DIRS) 
	uv run ruff check --select I --fix $(CHECK_DIRS)

lint-fix:
	uv run ruff check --fix $(CHECK_DIRS)

format-check:
	uv run ruff format --check $(CHECK_DIRS) 
	uv run ruff check -e $(CHECK_DIRS)
	uv run ruff check --select I -e $(CHECK_DIRS)

lint-check:
	uv run ruff check $(CHECK_DIRS)


# Crear el entorno virtual con uv
setup-venv:
	uv venv .venv
	@echo "Entorno virtual creado. Actívalo manualmente con:"
	@echo "  En Windows: .venv\\Scripts\\activate"
	@echo "  En Linux/Mac: source .venv/bin/activate"

# Instalar dependencias usando uv (después de activar el entorno virtual)
install-deps:
	uv pip install -e .
	uv sync

# Ejecutar la aplicación en modo desarrollo (requiere entorno virtual activado)
run-dev:
	@echo "Asegúrate de haber activado el entorno virtual"
	fastapi run src/ai_companion/interfaces/whatsapp/webhook_endpoint.py --port 8000 --host 0.0.0.0 --reload

# Windows: Configuración completa en un solo comando
run-local-win:
	uv venv .venv && \
	.venv\Scripts\activate && \
	uv sync && \
	uv pip install -e . && \
	fastapi run ai_companion/interfaces/whatsapp/webhook_endpoint.py --port 8000 --host 0.0.0.0 --reload