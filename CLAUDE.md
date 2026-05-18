# CLAUDE.md — MSPR-AI-Nutrition

Instructions ciblées pour Claude Code sur ce repo. Le `CLAUDE.md` racine de `MSPR/` reste la reference globale (architecture des 8 repos, reseau Docker, flux d'auth). Ce fichier complete sans dupliquer.

---

## Role du service

Microservice FastAPI d'analyse nutritionnelle par IA, expose sur le port `8001`. Deux flux :

1. **Classification d'aliments depuis photo** (HuggingFace `nateraw/food`, modele Food-101) → macros + recommandations.
2. **Generation de plans repas** (Ollama / `gemma3:4b`) — *non implemente, phase 4 du `PLAN.md`*.

Persistance dans la BDD partagee `healthai` (PostgreSQL 17, MSPR-DB). Pas de BDD propre.

---

## Avancement (au 2026-04-28)

Le `PLAN.md` (untracked) decrit 6 phases. Etat reel :

| Phase | Statut | Commits |
|-------|--------|---------|
| 1. Squelette + healthcheck | OK | #1, #2, #3 |
| 2. Modeles SQLAlchemy + schemas Pydantic | OK cote code, **PAS cote BDD** (voir piege ci-dessous) | #5 |
| 3. `POST /analyze-meal` (HuggingFace) | OK, 2 rounds de bugfix | #6, #7, #8 |
| 4. `POST /generate-meal-plan` (Ollama) | A faire | — |
| 5. `GET /meal-plans/{user_id}`, `GET /meal-analyses/{user_id}` | A faire | — |
| 6. Tests + doc | A faire | — |

---

## Pieges connus — a verifier avant tout changement

Ces ecarts entre le code et l'infra reelle sont importants. Claude doit les connaitre pour ne pas proposer de fix qui aggraveraient la situation.

### 1. La table `users` n'existe plus (MSPR-DB V7)

`app/db/models.py` declare `ForeignKey("users.id", ondelete="CASCADE")` sur `MealAnalysis.user_id`, `MealPlan.user_id`, `NutritionGoal.user_id`. La table `users` a ete droppee dans `MSPR-DB/migrations/V7__drop_users_table.sql`. Ces FK pointent dans le vide ; toute migration creee a partir de ces modeles echouera.

**A faire avant la phase 4** : ecrire les migrations `meal_analyses`, `meal_plans`, `nutrition_goals` dans `MSPR-DB` **sans** la FK vers `users`. Le `user_id` reste un `BIGINT` mais devient un identifiant opaque venant du JWT (service AUTH).

### 2. `nutrition_lookup.py` filtre sur une colonne supprimee

La requete contient `WHERE user_id IS NULL` mais la colonne `user_id` de `nutrition_entries` a ete droppee (V7 toujours). Le filtre n'echoue pas en runtime parce que SQLAlchemy ne valide pas le SQL en `text()`, mais la requete renverra une erreur PostgreSQL des qu'elle sera executee contre la BDD a jour. Retirer `user_id IS NULL AND` des deux requetes du fichier.

### 3. `spring_client.get_user_me` appelle un endpoint supprime

`GET /api/users/me` n'existe plus sur la branche `Sonar` de MSPR-API (`UserController` purge, auth deleguee a MSPR-AUTH). Le router `meal_analysis` casse en prod des qu'il essaie de recuperer le profil. Deux options :
- recuperer `user_id` directement depuis le JWT (decoder localement avec `BETTER_AUTH_SECRET`),
- ou appeler MSPR-AUTH (`GET /api/session`) plutot que MSPR-API.

Pour l'instant les objectifs nutritionnels (`NutritionGoal`) viennent de la BDD locale, pas du profil Spring — donc l'appel Spring ne sert qu'a recuperer `user_id`. Migrer vers du JWT decode local est plus simple.

### 4. `PLAN.md` n'est pas commite

Untracked depuis le debut. A commiter quand la prochaine phase est attaquee, pas avant (Arthur le retravaille ponctuellement).

---

## Stack & conventions

- **Python 3.12**, FastAPI 0.115, SQLAlchemy 2.0 (style `DeclarativeBase`), Pydantic v2 (`pydantic-settings`).
- **Inference CPU uniquement** : `torch==2.7.0+cpu` via l'index PyTorch CPU. Ne pas ajouter de deps GPU.
- **HuggingFace** : modele `nateraw/food` charge en lazy singleton (`_classifier` global dans `food_classifier.py`). Choix valide par benchmark (`docs/model_benchmark.md`, `docs/benchmark_results.json`).
- **Ollama** : container separe dans le `docker-compose.yml` du repo, reseau `internal`. Pull automatique de `gemma3:4b` au demarrage.
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

## Avant la phase 4 (Ollama / generate-meal-plan)

Checklist a derouler dans cet ordre :

1. Resoudre les 3 pieges ci-dessus (sinon la phase 4 hereitera des memes problemes).
2. Ecrire la migration MSPR-DB `V8__ai_nutrition_tables.sql` pour `meal_analyses`, `meal_plans`, `nutrition_goals` (sans FK vers `users`).
3. Decider du flux JWT → `user_id` (decode local recommande).
4. Ajouter `services/meal_plan_generator.py` : prompt structure → Ollama HTTP `/api/generate` avec `format: "json"` → validation Pydantic du JSON.
5. Persister dans `meal_plans`, retourner le plan.
