# MSPR-HealthAI-Coach-AI-Nutrition

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
