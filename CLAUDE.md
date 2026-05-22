# CLAUDE.md — MSPR-AI-Nutrition

Instructions ciblées pour Claude Code sur ce repo. Le `CLAUDE.md` racine de `MSPR/` reste la reference globale (architecture des 8 repos, reseau Docker, flux d'auth). Ce fichier complete sans dupliquer.

---

## Role du service

Microservice FastAPI d'analyse nutritionnelle par IA, expose sur le port `8001`. Deux flux :

1. **Classification d'aliments depuis photo** (HuggingFace `nateraw/food`, modele Food-101) : macros + recommandations + serving_sizes PNNS.
2. **Generation de plans repas** : architecture LLM multi-provider livree par PRD #71. Mistral Small managed en primaire, Ollama Gemma3:4b en fallback / mode offline. Validator DeCRIM-light + retry + fallback statique. Selecteur utilisateur via `PATCH /me/preferences`.

Persistance dans la BDD partagee `healthai` (PostgreSQL 17, MSPR-DB). Pas de BDD propre.

---

## Avancement (au 2026-05-22)

PRD #45 (squelette + analyse + plans) et PRD #71 (multi-provider Mistral + Ollama + selecteur user) cloturees. Tous les livrables PDF MSPR2 (#5 API IA, #6 OpenAPI, #8 modele de donnees, #9 tests / couverture, #1 metriques) sont en place.

| Item | Statut | Reference |
|------|--------|-----------|
| Squelette + healthcheck + `/analyze-meal` + `/generate-meal-plan` | OK | PRD #45 (PR #62, #67, #68, #70) |
| Pipeline DeCRIM-light + few-shot | OK | Slices 6 / 9 (PR #64, #69) |
| Migrations V8 -> V13 (BDD live a jour) | OK | `MSPR-DB/migrations/V8` ... `V13` |
| Tests pytest + couverture lignes 95.96 % / branches 84.58 % | OK | Slice 10 (PR #70) |
| Documentation OpenAPI nettoyee | OK | Slice 10 (PR #70) |
| Eval classifier Food-101 (top-1 0.888, top-5 0.972) | OK | `docs/metrics.md` |
| Eval LLM Mistral N=20 + N=100 (json_validity 1.0, p50 ~4 s) | OK | `docs/metrics.md` |
| Selecteur multi-provider + endpoint `/me/preferences` (V13) | OK | PRD #71 Slice 3 (commit `fc21fa0`) |
| FallbackChain inter-providers Mistral <-> Ollama | OK | PRD #71 Slice 2 (PR #73) |
| Doc pivot Mistral + narration soutenance | OK | PRD #71 Slice 6 (issue #77, ce commit) |

---

## Etat post-pivot Mistral : ce qu'il faut connaitre

Les 3 pieges historiques (FK `users`, filtre `user_id IS NULL`, `spring_client.get_user_me`) sont resolus. Le repo est dans un etat coherent. Quelques points qui restent contre-intuitifs :

### 1. Gemma3:4b ne tient pas le prompt few-shot sur cette infra CPU

Le prompt avec 3 exemples few-shot pese ~2245 tokens. Ollama / Gemma3:4b en mode CPU pur (`torch==2.7.0+cpu`) renvoie 500 / aborted apres 180 s sans emettre le premier token. C'est ce constat qui a declenche le pivot vers Mistral. Ne pas relancer une eval LLM Gemma sur cette infra : prevoir un host GPU (cf. `GPU_EVAL_PLAYBOOK.md` et `docker-compose.gpu-eval.yml`). Pour le code, Gemma reste branche dans `LLMProvider` et adresse les chemins fallback + offline.

### 2. `meal_plans.llm_backend_used` n'est pas le primaire demande

La `FallbackChain` peut basculer du primaire (Mistral) vers le secondaire (Ollama) en cas d'echec reseau / quota. Dans ce cas, la ligne persistee porte le backend reellement utilise et un `compliance_warning` explicitant la bascule. Le helper `_latest_plan_id` dans `meal_plan_orchestrator.py` ne filtre donc pas par backend, contrairement a `_lookup_cached_plan` qui filtre par `(user_id, inputs_hash, llm_backend_used)` pour eviter un cache hit cross-backend apres un switch user.

### 3. Cle Mistral via `MISTRAL_API_KEY`

Renomme depuis l'historique `CLE_MISTRAL` (convention anglaise comme le reste du repo). Lue depuis `.env`, jamais commitee. En son absence, le `MistralProvider` est instancie en mode degrade : tout appel leve `ValueError` avant le HTTP. La factory bascule alors automatiquement sur Ollama si la preference user / le defaut env vaut `mistral`.

### 4. `PLAN.md` et `TODO_DEMAIN.md` restent untracked

Fichiers de travail d'Arthur, retravailles ponctuellement entre sessions. Ne pas les commiter, ne pas les reecrire de fond. Voir aussi la regle globale : ne pas reformuler les textes d'Arthur, prefere supprimer ou pointer le passage problematique.

---

## Stack & conventions

- **Python 3.12**, FastAPI 0.115, SQLAlchemy 2.0 (style `DeclarativeBase`), Pydantic v2 (`pydantic-settings`).
- **Inference CPU uniquement** : `torch==2.7.0+cpu` via l'index PyTorch CPU. Ne pas ajouter de deps GPU. Eval GPU = host externe via `GPU_EVAL_PLAYBOOK.md`.
- **HuggingFace** : modele `nateraw/food` charge en lazy singleton (`_classifier` global dans `food_classifier.py`). Choix valide par benchmark (`docs/model_benchmark.md`, `docs/benchmark_results.json`).
- **LLM multi-provider** : `MistralProvider` (primaire, `mistral-small-latest`, `response_format=json_schema strict:true`) + `OllamaProvider` (fallback, `gemma3:4b`). Selection via `get_preferences(user_id, db).effective_llm`. Factory `get_provider(name)` dans `app/services/llm_provider.py`.
- **Reseaux Docker** : `mspr_data_network` (external, cree par le compose racine) + `internal` (Ollama).
- **Imports** : `from __future__ import annotations` en tete des fichiers Python.
- **Strings** : guillemets doubles partout (style transformers/FastAPI).

---

## Commandes courantes

```bash
# Demarrage standalone
docker compose up -d --build

# Avec le reste de la stack (preferable)
cd /home/arthur/Projects/MSPR && docker compose up -d --build

# Healthcheck
curl http://localhost:8001/health

# OpenAPI
open http://localhost:8001/docs

# Benchmark des modeles HuggingFace (necessite container demarre)
./scripts/run_benchmark.sh
```

Pas de `pytest` configure pour l'instant (phase 6).

---

## Workflow git

- **Branches** : `master` (defaut), PR systematique pour chaque phase / feature.
- **Numerotation** : les commits referencent les numeros d'issue/PR (`(#N)`). Continuer cette convention.
- **Co-author Claude** : interdit (regle globale Arthur). Aucune mention de Claude Code dans les commits, PR ou issues sauf demande explicite.
- **Push** : pas de SSH agent dans cet env, preparer les commandes pour Arthur plutot que tenter `git push`.

---

## Style de redaction (specifique Arthur)

- **Pas de tirets cadratins** (`—`) ni demi-cadratins (`–`) dans le code, les commentaires, les commit messages, ou la doc. Utiliser `:` ou `,` ou `-` ASCII.
- **Ne pas reecrire** les textes d'Arthur (commentaires, README, docs) : prefere supprimer ou pointer le passage problematique.
- Commentaires en francais quand ils existent. Style sobre, pas de docstring multi-paragraphes.

---

## References documentaires

- `docs/metrics.md` : chiffres classifier Food-101 + LLM Mistral N=20 / N=100 + comparaison naive vs pipeline.
- `docs/data_model.md` : tables AI-Nutrition + decisions architecturales (pas de FK users, hashes cache, selection multi-provider V13).
- `docs/pivot_mistral.md` : justification du pivot Mistral et architecture multi-provider (narration soutenance).
- `GPU_EVAL_PLAYBOOK.md` + `docker-compose.gpu-eval.yml` : relance eval LLM Gemma sur host GPU externe.
- `TODO_DEMAIN.md` : etat / decisions en attente entre sessions (untracked).
