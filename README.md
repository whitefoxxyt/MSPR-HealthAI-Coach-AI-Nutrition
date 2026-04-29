# MSPR-HealthAI-Coach-AI-Nutrition

[![CI](https://github.com/whitefoxxyt/MSPR-HealthAI-Coach-AI-Nutrition/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/whitefoxxyt/MSPR-HealthAI-Coach-AI-Nutrition/actions/workflows/ci.yml)

Micro-service d'analyse nutritionnelle par IA, partie de la plateforme HealthAI Coach (MSPR2).

## Stack

- **FastAPI** : API REST
- **HuggingFace Transformers** : classification d'aliments depuis photo
- **Ollama + Gemma3:4b** : génération de plans repas personnalisés (JSON)
- **PostgreSQL** : stockage des analyses et plans générés

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /analyze-meal` | Analyse photo → macros + déséquilibres |
| `POST /generate-meal-plan` | Génération plan repas personnalisé |
| `GET /meal-plans/{user_id}` | Historique des plans générés |
| `GET /meal-analyses/{user_id}` | Historique des analyses |
| `GET /health` | Healthcheck |

## Démarrage

```bash
# À venir
```

## Tests

Le harnais utilise pytest + testcontainers (PostgreSQL ephemere) + respx (mocks HTTP).
Structure :

```
tests/
├── conftest.py     # 5 fixtures partagees (pg_container, db_session, mock_ollama, mock_classifier, valid_jwt)
├── unit/           # tests purs, sans IO (rapides)
├── integration/    # tests contre PG ephemere via testcontainers
└── slow/           # smoke tests dependances externes (Ollama, HuggingFace), exclus du CI
```

### Lancer la suite

Conteneur (recommandé, isole testcontainers du host) :

```bash
docker compose --profile test build tests
docker compose --profile test run --rm tests          # unit + integration
```

Local (Python 3.12 + Docker pour testcontainers) :

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt -r requirements-dev.txt
MIGRATIONS_DIR=../MSPR-DB/migrations pytest                # unit + integration
MIGRATIONS_DIR=../MSPR-DB/migrations pytest -m slow        # tests dependances externes
```

### Couverture

Cible : > 80% global. L'enforcement `--cov-fail-under=80` sera active une fois
les phases 4-6 livrees ; le harnais actuel est un echafaudage.
Rapport HTML genere dans `htmlcov/` apres chaque run.
