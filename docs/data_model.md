# Modele de donnees relationnel : adaptations AI-Nutrition

Document destine au livrable PDF MSPR2 numero 8 ("Modele de donnees relationnel documente expliquant les adaptations realisees sur le modele existant").

Le service `MSPR-AI-Nutrition` partage la base PostgreSQL `healthai` avec le reste de la plateforme MSPR (cf. `MSPR-DB`). Trois nouvelles tables et deux enrichissements ont ete ajoutes pour porter les pipelines IA. Les migrations sont versionnees dans `MSPR-DB/migrations/`.

## Migrations concernees

| Version | Fichier | Apport |
|---------|---------|--------|
| V8 | `V8__ai_nutrition_tables.sql` | Creation des 3 tables AI-Nutrition (`meal_analyses`, `meal_plans`, `nutrition_goals`). |
| V9 | `V9__ai_nutrition_enrichment.sql` | Ajout `nutrition_goals.health_goal` (CHECK), `meal_plans.inputs_hash`, `meal_analyses.recommendations`. |
| V10 | `V10__meal_analyses_recommendations_hash.sql` | Ajout `meal_analyses.recommendations_hash` + index. |
| V11 | `V11__ai_nutrition_v11.sql` | Ajout `meal_analyses.imbalances`, `meal_analyses.serving_sizes`, `meal_analyses.meal_type`, `meal_plans.compliance_status` (NOT NULL DEFAULT `'full'`), `meal_plans.compliance_warnings`. |

## Tables ajoutees

### `meal_analyses`

Resultat d'une analyse de photo de repas (endpoint `POST /api/v1/analyze-meal`).

| Colonne | Type | Role |
|---------|------|------|
| `id` | `BIGSERIAL PK` | Identifiant. |
| `user_id` | `BIGINT NOT NULL` | Identifiant opaque issu du JWT (cf. ci-dessous). |
| `photo_url` | `VARCHAR(500)` | Reserve, non utilise actuellement (la photo n'est pas stockee). |
| `detected_foods` | `JSONB` | Top-3 aliments avec score : `[{label, confidence, nutrition}]`. |
| `macros` | `JSONB` | Macros calculees du repas : `{calories, protein_g, carbs_g, fat_g, fiber_g}`. |
| `confidence_scores` | `JSONB` | Map `label -> score` (HuggingFace). |
| `recommendations` | `JSONB` | Liste de phrases de recommandations (LLM ou matrice). |
| `recommendations_hash` | `VARCHAR(64)` | SHA256 hex du tuple `(top_label, health_goal, imbalances triees)`. NULL en mode fallback. |
| `imbalances` | `JSONB` | (V11) Tags structures `{nutrient, status, delta_pct, target_value, actual_value, unit}`. NULL ou liste vide quand le profil utilisateur est incomplet (TDEE non calculable). |
| `serving_sizes` | `JSONB` | (V11) 3 portions (`small` / `medium` / `large`) par item detecte avec macros recalculees. Reperes PNNS via le mapping Food-101 -> PNNS. |
| `meal_type` | `TEXT` | (V11) `breakfast` / `lunch` / `dinner` / `snack`. NULL = fallback TDEE/4 cote app. |
| `created_at` | `TIMESTAMP` | Date de l'analyse, sert egalement de TTL (30 jours pour le cache). |

Indexes :
- `idx_meal_analyses_user_id (user_id)` : historique par utilisateur.
- `idx_meal_analyses_created_at (created_at DESC)` : tri pour la pagination.
- `idx_meal_analyses_recommendations_hash (recommendations_hash)` : lookup cache LLM.

### `meal_plans`

Plans repas generes par LLM (endpoint `POST /api/v1/generate-meal-plan`).

| Colonne | Type | Role |
|---------|------|------|
| `id` | `BIGSERIAL PK` | Identifiant. |
| `user_id` | `BIGINT NOT NULL` | Identifiant opaque issu du JWT. |
| `plan` | `JSONB` | Plan complet : `{fallback, days: [{day, meals: [{name, macros, ingredients, est_budget_eur, prep_time_min}]}]}`. |
| `objective` | `VARCHAR(100)` | Objectif sante effectif (`weight_loss`, `muscle_gain`, `balance`, `sport_performance`). |
| `constraints` | `JSONB` | Contraintes canonicalisees (regime, allergies triees, budget, duree, calories cible). |
| `inputs_hash` | `VARCHAR(64)` | SHA256 hex de `constraints` + user_id, cle de cache (TTL 7 jours). |
| `compliance_status` | `TEXT NOT NULL DEFAULT 'full'` | (V11) Sortie de la boucle DeCRIM-light : `full`, `partial_budget` ou `static_fallback`. Default `full` pour la retro-compatibilite des lignes pre-V11. |
| `compliance_warnings` | `TEXT[]` | (V11) Strings explicitant les relachements de contraintes (ex. budget depasse). NULL ou tableau vide quand `compliance_status = 'full'`. |
| `generated_at` | `TIMESTAMP` | Date de generation. |

Indexes :
- `idx_meal_plans_user_id (user_id)` : historique par utilisateur.
- `idx_meal_plans_generated_at (generated_at DESC)` : tri pour la pagination.
- `idx_meal_plans_inputs_hash (inputs_hash)` : lookup cache LLM.

### `nutrition_goals`

Profil nutritionnel de l'utilisateur (endpoints `GET` / `PUT /api/v1/nutrition-goals/me`).

| Colonne | Type | Role |
|---------|------|------|
| `user_id` | `BIGINT PK` | Une ligne par utilisateur (PK = user_id, pas d'ID synthetique). |
| `calories_target` | `INTEGER` | Apport quotidien cible. |
| `protein_g` / `carbs_g` / `fat_g` | `DECIMAL(8, 2)` | Macros cibles. |
| `allergies` | `TEXT[]` | Liste declarative. |
| `diet_type` | `VARCHAR(50)` | `omnivore`, `vegetarien`, `vegan`, `sans_gluten` (validation cote app via Pydantic). |
| `health_goal` | `VARCHAR(30)` | CHECK in `('weight_loss', 'muscle_gain', 'balance', 'sport_performance')`. NULL = `balance` par defaut. |

## Choix d'architecture

### Pas de cle etrangere vers `users`

La table `users` heritee du schema initial (V1) a ete supprimee en V7. Les datasets sources de l'ETL sont anonymises et n'ont pas d'identifiant utilisateur commun. La verite des comptes vit desormais dans le service `MSPR-AUTH` (PostgreSQL dedie sur le port 5433, schema better-auth).

`user_id` reste donc un `BIGINT` opaque dans les 3 tables AI-Nutrition. Il est extrait du JWT decode localement par AI-Nutrition (HS256 partage avec MSPR-AUTH via `BETTER_AUTH_SECRET`). L'integrite est garantie par le secret partage : un JWT mal signe est rejete avant toute requete BDD.

Aucune contrainte d'integrite referentielle n'est posee. C'est un compromis assume :
- avantage : les services AI-Nutrition et AUTH peuvent evoluer independamment, pas de couplage transversal sur la BDD.
- limite : un user supprime cote AUTH laisse des lignes orphelines en BDD AI-Nutrition. Acceptable puisque l'API filtre toujours par `user_id` issu du JWT (un utilisateur supprime ne peut plus s'authentifier).

### `nutrition_goals.user_id` comme cle primaire

Une seule ligne par utilisateur. La PK directement sur `user_id` evite un ID synthetique inutile et permet l'upsert via `INSERT ... ON CONFLICT (user_id) DO UPDATE`.

### Hashes de cache (`inputs_hash`, `recommendations_hash`)

Deux caches applicatifs sont implementes en BDD (pas de Redis ni autre store).

- **`meal_plans.inputs_hash`** (V9) : SHA256 hex (64 caracteres) du JSON canonicalise des inputs (`PlanInputs` triee : objective, duration_days, diet_type, allergies triees, budget_per_day, calories_target, user_id). TTL 7 jours. Index dedie pour le lookup.
- **`meal_analyses.recommendations_hash`** (V10) : SHA256 hex du tuple `(top_food_label, health_goal, imbalances triees)`. Cache global (sans user_id) car la recommandation depend uniquement du contexte nutritionnel, pas de l'utilisateur. TTL 30 jours, applique cote SELECT (`created_at > NOW() - INTERVAL`).

NULL = recommandation issue du fallback matrice. On ne cache pas le fallback : un appel ulterieur retentera Ollama.

Race condition assumee : deux requetes concurrentes avec le meme hash peuvent toutes deux declencher un appel LLM avant que la premiere n'ecrive. La seconde ecrasera juste l'entree avec une valeur equivalente. Volume actuel trop faible pour justifier un `INSERT ON CONFLICT`.

### Champ `health_goal` enrichi en V9

Initialement `nutrition_goals` n'avait pas d'objectif sante : seulement les cibles macros. Pour resoudre l'objectif applicable a `/generate-meal-plan` sans demander a l'utilisateur a chaque appel, on stocke le choix dans le profil. `NULL` est traite comme `balance` cote app, ce qui permet de creer un profil minimal sans choisir d'objectif.

La contrainte CHECK SQL bloque les valeurs incoherentes en BDD au cas ou l'API serait court-circuitee.

### Pas de schema dedie

Les 3 tables coexistent dans le schema `public` avec les tables ETL existantes (`exercises`, `nutrition_entries`, etc.). Aucun isolation cote BDD : chaque service a son propre utilisateur applicatif ou non, mais toutes les migrations sont centralisees dans `MSPR-DB`. C'est coherent avec le pattern adopte sur la plateforme et ca permet les jointures (le service utilise `nutrition_entries` pour le lookup nutritionnel).

### JSONB pour les structures variables

`detected_foods`, `macros`, `plan`, `constraints` sont des structures dont le schema peut evoluer (ajout de nouveaux champs au gre des iterations LLM ou des extensions du modele). `JSONB` permet ces evolutions sans migration. Tradeoff connu : pas de validation SQL des sous-structures, c'est Pydantic cote application qui assure le contrat.

## Liens avec les endpoints

| Endpoint | Tables touchees |
|----------|-----------------|
| `POST /api/v1/analyze-meal` | Lit `nutrition_entries` (lookup), `nutrition_goals` (profil). Ecrit `meal_analyses`. Lit/ecrit `meal_analyses` pour le cache de recommandations. |
| `GET /api/v1/meal-analyses/me` | Lit `meal_analyses` (filtre `user_id`, tri `created_at DESC`, pagination). |
| `POST /api/v1/generate-meal-plan` | Lit `nutrition_goals` (profil). Lit/ecrit `meal_plans` pour le cache (filtre `user_id` + `inputs_hash`, TTL 7j). |
| `GET /api/v1/meal-plans/me` | Lit `meal_plans` (filtre `user_id`, tri `generated_at DESC`, pagination). |
| `GET` / `PUT /api/v1/nutrition-goals/me` | Lit / upsert `nutrition_goals`. |

## Schema applicatif

Cote Python, les 3 tables sont mappees via SQLAlchemy 2.0 dans `app/db/models.py`. Les schemas Pydantic v2 sont declares dans `app/models/schemas.py` et exposes par FastAPI dans l'OpenAPI generee.
