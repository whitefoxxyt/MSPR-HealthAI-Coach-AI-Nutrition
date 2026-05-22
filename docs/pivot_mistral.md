# Justification du pivot Mistral et architecture multi-provider

Document destine a la narration soutenance MSPR2. Livre par le PRD #71 (issue #71) et son slice 6 de documentation (issue #77).

Ce fichier expose le pourquoi du pivot du LLM par defaut de Gemma3:4b local vers Mistral Small managed, ce que l'architecture multi-provider apporte par rapport au cahier des charges, et comment Gemma reste branche dans le service.

## 1. Constat infrastructure : Gemma3:4b CPU ne tient pas le prompt

Le pipeline `POST /generate-meal-plan` est passe par trois iterations qui ont elargi le prompt :

1. Squelette : prompt zero-shot, sortie JSON via Ollama `format: json`. Tenait en ~600 tokens et repondait en 60 a 90 s sur CPU pur (`torch==2.7.0+cpu`).
2. DeCRIM-light : validator + retry de relachement de contraintes. Pas d'impact sur la taille du prompt, mais multiplication des appels Ollama jusqu'a 3 fois par requete.
3. Few-shot (slice 9, PR #69) : ajout de 3 exemples de plans (7j / 5j / 1j) dans le prompt template pour cadrer la conformite aux contraintes. Le prompt atteint ~2245 tokens (~9000 caracteres).

Sur l'infra cible (CPU pur, container Docker, pas d'acceleration), Ollama / Gemma3:4b avec ce prompt :

- Trois runs d'eval consecutifs ont retourne 14 erreurs HTTP 500 cote Ollama, meme apres `_OLLAMA_TIMEOUT_S` passe a 180 s.
- Le serveur Ollama abandonne avec 500 / aborted sans avoir emis le premier token : ce n'est pas un timeout amont, c'est le moteur d'inference qui rend la main.
- Le smoke n=3 montre `static_fallback=1.0` : le LLM ne digere pas du tout le prompt few-shot, on tape systematiquement le fallback statique en sortie.

Le ceiling observe est aux alentours de 1500 tokens. Au-dela, Gemma3:4b CPU n'est plus utilisable dans un SLA UX raisonnable (objectif UX : reponse < 10 s, cf user story #1 du PRD).

Trace de ces echecs : `TODO_DEMAIN.md` entrees 2026-04-29 et 2026-04-30, `docs/metrics.md` section "Gemma3:4b local".

## 2. Decision : pivot vers Mistral managed + abstraction provider

Plutot que d'enlever le few-shot (qui apporte +38 points sur la conformite aux contraintes, cf `docs/metrics.md`) ou de degrader l'objectif UX, on isole l'appel reseau derriere une abstraction et on remplace le backend par defaut.

Decisions :

- **Interface `LLMProvider`** dans `app/services/llm_provider.py`, exposant `async generate(prompt, schema) -> str`. Deux implementations livrees : `OllamaProvider` (extrait de l'ancien `llm_client.py`) et `MistralProvider` (POST `/v1/chat/completions`, `response_format=json_schema strict:true`).
- **`SchemaSanitizer`** : applati les `$defs` Pydantic v2 et supprime les `default` / `anyOf` non supportes par Mistral en strict, sans alterer la semantique du schema metier.
- **`FallbackChain`** : sequence primaire utilisateur -> secondaire Ollama -> static_fallback. Attache un `compliance_warning` quand la bascule s'opere, pour que l'API restitue honnetement quel backend a repondu.
- **Selecteur utilisateur** : table `nutrition_goals.preferred_llm` (V13), endpoint `GET / PATCH /me/preferences`, resolution dans `get_preferences(user_id, db).effective_llm`. Defaut env `DEFAULT_LLM=mistral`.
- **Persistance audit** : `meal_plans.llm_backend_used` (V13) trace le backend qui a effectivement repondu (peut differer du primaire en cas de fallback). Sert aussi de cle de cache pour eviter un hit cross-backend apres switch utilisateur.

La logique metier preexistante (cache `inputs_hash`, retry DeCRIM-light, validator de contraintes, fallback statique) est inchangee. Les tests unitaires existants ont migre du mock `_call_ollama_generate` vers le mock `LLMProvider`.

## 3. Conformite au PDF MSPR2 section III.3

Le cahier des charges TPRE502 section III.3 exige explicitement :

> l'integration robuste des APIs externes avec gestion des pannes, mise en cache intelligente, gestion de la charge (rate limiting), et mecanismes de fallback pour assurer la continuite de service.

Avant le pivot, le service tournait sans aucune API externe (Ollama est un container local). L'exigence n'etait pas adressee. Apres le pivot, chacun des quatre items est materialise :

| Exigence PDF III.3 | Materialisation dans le service |
|--------------------|---------------------------------|
| API externe integree | `MistralProvider` : `POST https://api.mistral.ai/v1/chat/completions` avec `Authorization: Bearer $MISTRAL_API_KEY`. |
| Gestion des pannes | `FallbackChain` : sur 5xx / 429 / timeout / JSON invalide apres retries, bascule sur Ollama, puis static_fallback. Le caller recoit un `compliance_warning` explicitant la bascule. |
| Mise en cache intelligente | `meal_plans.inputs_hash` (SHA256 des inputs canonicalises) avec TTL 7 jours, filtre par `(user_id, inputs_hash, llm_backend_used)` pour eviter le cache hit cross-backend. |
| Rate limiting | `Limiter` slowapi keye par `user_id` JWT, limites `10/hour;3/minute` sur `POST /generate-meal-plan`. Applique a tous les backends. |

C'est la justification cle a defendre devant le jury : le service n'a pas seulement un LLM externe, il a l'architecture pour le gerer dans la duree.

## 4. Conservation de Gemma3:4b

Gemma n'a pas ete supprime du service. Le `OllamaProvider` reste en place et adresse trois usages distincts du primaire managed :

- **Backend de fallback** : si Mistral retourne 5xx / 429 / timeout, la `FallbackChain` route automatiquement sur Ollama avant le static_fallback. Le plan est servi, le `compliance_warning` indique la bascule. Garantit la continuite de service exigee par III.3.
- **Mode offline / on-premise** : un deploiement sans `MISTRAL_API_KEY` (par exemple un hopital qui refuse l'exfiltration de donnees sante, RGPD article 9) peut tourner Ollama-only en forcant `DEFAULT_LLM=ollama`. Le `MistralProvider` est instancie en mode degrade et leve `ValueError` avant le HTTP : la factory bascule alors automatiquement.
- **Eval differee** : `GPU_EVAL_PLAYBOOK.md` + `docker-compose.gpu-eval.yml` permettent de relancer une eval Gemma sur un host GPU externe pour produire les chiffres comparatifs hors UX temps reel. Non disponible pour la soutenance mais l'infra est cablee.

Narration : on n'abandonne pas Gemma, on le reclasse dans son perimetre pertinent (offline, fallback, eval) et on libere le chemin chaud pour un backend dimensionne.

## 5. Chiffres comparatifs (reprise de docs/metrics.md)

Les chiffres complets vivent dans `docs/metrics.md`. Synthese ici pour la soutenance.

Mistral Small managed, pipeline complet (validator + retry) sur seed 42 :

| Indicateur | N=20 | N=100 |
|------------|------|-------|
| `compliance_status=full` | 0.65 | 0.71 |
| `partial_compliance` | 0.05 | 0.03 |
| `static_fallback` | 0.00 | 0.00 |
| JSON validity rate | 1.00 | 1.00 |
| Latence p50 (ms) | 2 982 | 4 163 |
| Latence p95 (ms) | 9 743 | 9 760 |

Effet du validator DeCRIM-light (Mistral N=100, naive vs pipeline) :

| Indicateur | Naive | Pipeline | Gain |
|------------|-------|----------|------|
| `constraint_satisfaction` global | 0.33 | 0.71 | +38 pts |
| Allergies respectees | 0.67 | 1.00 | +33 pts |
| Diete respectee | 0.63 | 1.00 | +37 pts |
| Budget respecte | 0.76 | 0.96 | +20 pts |

Gemma3:4b : pas de chiffres N=20 disponibles sur cette infra CPU (cf. section 1). L'eval sur GPU est cablee mais non lancee.

Lecture pour le jury :

- `json_validity_rate = 1.0` confirme la pertinence du mode `response_format=json_schema strict:true` de Mistral : pas un JSON malforme sur 120 generations.
- p50 ~4 s tient le SLA UX du PRD (< 10 s).
- Le gain +38 points du validator DeCRIM-light montre que le pipeline maison apporte une valeur tangible par-dessus le LLM brut.
- 0 % de `static_fallback` sur 120 generations confirme que la chaine Mistral primaire suffit en regime nominal. Le fallback Ollama est la pour les exceptions, pas le quotidien.

## 6. Slide soutenance dediee

La diapositive "Justification des choix IA" (cf `rapport/soutenance-mspr2/slides.md`) presente la comparaison Mistral vs Gemma et la lecture honnete des chiffres. Elle anticipe les questions jury "pourquoi un LLM managed si tout est cense etre auto-heberge ?" en repondant sur trois axes : conformite PDF III.3, contrainte materielle CPU, conservation de Gemma pour les cas pertinents.
