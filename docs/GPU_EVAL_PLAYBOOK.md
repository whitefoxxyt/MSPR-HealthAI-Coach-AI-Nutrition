# Playbook : relancer l'eval LLM sur GPU

Ce guide decrit comment relancer l'evaluation comparative `avec / sans few-shot`
de MSPR-AI-Nutrition sur un poste avec GPU NVIDIA. L'objectif est de consolider
les chiffres de la section "Impact du few-shot" de `docs/metrics.md` en passant
de `n=5` (limite CPU) a `n=30` par condition, et en injectant les exemples
few-shot complets (7j + 5j + 1j) plutot que tronques a `days[:1]`.

Le run complet dure environ 1 a 2 heures sur RTX 3060 / 3070, 30 a 60 minutes
sur RTX 4080 ou plus, selon le tour de chauffe Ollama.

## Resultat attendu

A la fin de la procedure, tu auras 4 fichiers a renvoyer a Arthur :

```
docs/metrics.json                          metriques fusionnees naive + pipeline
docs/metrics.md                            rendu Markdown
docs/eval_runs/with_fewshot/metrics.json   detail run avec few-shot
docs/eval_runs/without_fewshot/metrics.json detail run sans few-shot
```

Soit en ouvrant une PR depuis ton fork, soit en zippant `docs/` et en le
partageant.

## 1. Prerequis poste GPU

- Linux (Ubuntu / Debian de preference) ou Windows + WSL2.
- Docker Engine 20.10+ avec compose v2 (`docker compose version` doit repondre).
- NVIDIA Container Toolkit installe et configure pour Docker.
  Verification : `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
  doit afficher la carte. Si non, suivre
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- 12 Go de RAM libres, 8 Go de VRAM minimum (gemma3:4b en fp16 tient en 6 Go).
- 10 Go d'espace disque libre (image Docker + modele Ollama).
- Acces internet pour pull les images et le modele.

## 2. Cloner le repo

```bash
git clone https://github.com/whitefoxxyt/MSPR-HealthAI-Coach-AI-Nutrition.git
cd MSPR-HealthAI-Coach-AI-Nutrition
git checkout master
cp .env.example .env
```

Pas besoin de modifier `.env` pour l'eval : les defauts pointent vers les
containers internes (`mspr-healthai-db`, `mspr-ollama`).

Cloner aussi MSPR-DB en sibling : les migrations sont referencees par
`MIGRATIONS_DIR` lors du bootstrap PostgreSQL.

```bash
cd ..
git clone https://github.com/whitefoxxyt/MSPR-HealthAI-Coach-DB.git MSPR-DB
cd MSPR-HealthAI-Coach-AI-Nutrition
```

## 3. Desactiver le slicing few-shot

Le code prod tronque les exemples few-shot a `days[:1]` pour tenir dans le
budget tokens de Gemma3:4b sur CPU. Sur GPU on garde les exemples complets.

Plus aucune modification de code n'est necessaire : le slicing est pilote par
la variable d'environnement `FEW_SHOT_FULL_EXAMPLES` (settings
`few_shot_full_examples`, defaut `false`). Elle est activee par defaut dans
`docker-compose.gpu-eval.yml` et passee explicitement (`-e
FEW_SHOT_FULL_EXAMPLES=true`) dans les commandes de la section 6.

## 4. Demarrer la stack via docker-compose.gpu-eval.yml

Le repo embarque un compose dedie a l'eval GPU qui orchestre db + ollama (GPU) +
ai-nutrition d'un coup, avec les bonnes config volumes / reseaux / migrations.

```bash
cp .env.example .env
docker compose -f docker-compose.gpu-eval.yml up -d --build
```

Verification :

```bash
# La carte doit etre visible dans le container ollama
docker exec mspr-ollama nvidia-smi

# gemma3:4b doit etre pulled (le compose le pull au demarrage, 2-5 min)
docker exec mspr-ollama ollama list

# Les 3 tables AI-Nutrition doivent exister (migrations V1-V12 jouees au boot db)
docker exec mspr-healthai-db psql -U healthai_user -d healthai -c \
  "SELECT table_name FROM information_schema.tables \
   WHERE table_schema='public' \
   AND table_name IN ('meal_plans','meal_analyses','nutrition_goals');"

# Le service repond
curl http://localhost:8001/health
# {"status":"ok","postgres":"up","ollama":"up","timestamp":"..."}
```

Si `nvidia-smi` echoue dans le container, c'est que NVIDIA Container Toolkit
n'est pas configure. Sans GPU, Ollama tombera sur CPU et la latence sera
identique au run CPU local (200 000+ ms par plan). Voir prerequis section 1.

## 5bis. (Optionnel) Relancer l'eval classifier avec eval terrain

L'eval classifier (Food-101) tourne deja a `n=1000` sur CPU dans les chiffres
actuels. La re-lancer sur GPU permet de la consolider et surtout d'ajouter la
section terrain (dataset `Matthijs/snacks` deja extrait dans
`data/eval_terrain/`) qui mesure le delta Food-101 vs photos amateures.

```bash
docker exec mspr-ai-nutrition pip install --quiet matplotlib==3.9.2 datasets==3.2.0

docker exec \
  -e MPLBACKEND=Agg \
  -e PYTHONPATH=/app \
  mspr-ai-nutrition \
  python scripts/eval_metrics.py classifier \
    --n-food101 5000 \
    --terrain-dir data/eval_terrain \
    --seed 42
```

Sortie : `docs/metrics.json` mis a jour avec `classifier.food101` (chiffres
consolides sur n=5000) et `classifier.terrain` (top-1, top-5, unknown_rate
sur les 49 photos amateures).

Si le dataset `Matthijs/snacks` n'a pas encore ete extrait sur le poste,
le faire avant :

```bash
docker exec mspr-ai-nutrition python scripts/download_terrain_dataset.py
```

## 6. Lancer les 2 runs (avec / sans few-shot)

L'eval s'execute dans le container `mspr-ai-nutrition` (acces a Ollama via le
reseau interne `ai_internal`, et aux scripts via le bind-mount).

### 6.1 Installer les deps eval dans le container

Le container prod n'inclut pas `matplotlib` et `datasets`. Soit on les pulle
dans le venv du container, soit on construit `Dockerfile.test`. Le plus simple :

```bash
docker exec mspr-ai-nutrition pip install --quiet matplotlib==3.9.2 datasets==3.2.0
```

### 6.2 Run avec few-shot

```bash
docker exec \
  -e FEW_SHOT_ENABLED=true \
  -e FEW_SHOT_FULL_EXAMPLES=true \
  -e MPLBACKEND=Agg \
  -e PYTHONPATH=/app \
  mspr-ai-nutrition \
  python scripts/eval_metrics.py llm \
    --n-generations 30 \
    --n-constraint-plans 30 \
    --seed 42 \
    --output-dir docs/eval_runs/with_fewshot
```

Duree estimee : 30 a 60 minutes selon la VRAM disponible. Le run produit :

```
docs/eval_runs/with_fewshot/metrics.json
docs/eval_runs/with_fewshot/metrics.md
docs/eval_runs/with_fewshot/llm_latency_distribution.png
```

### 6.3 Run sans few-shot

```bash
docker exec \
  -e FEW_SHOT_ENABLED=false \
  -e MPLBACKEND=Agg \
  -e PYTHONPATH=/app \
  mspr-ai-nutrition \
  python scripts/eval_metrics.py llm \
    --n-generations 30 \
    --n-constraint-plans 30 \
    --seed 42 \
    --output-dir docs/eval_runs/without_fewshot
```

Meme duree, memes inputs (seed 42 partage), donc strictement comparable.

### 6.4 Re-rendre le rapport principal

Les 2 runs ci-dessus ecrivent dans des sous-dossiers. Pour que `docs/metrics.md`
soit a jour avec le run "prod" (avec few-shot), relancer un dernier run vers la
racine `docs/` :

```bash
docker exec \
  -e FEW_SHOT_ENABLED=true \
  -e FEW_SHOT_FULL_EXAMPLES=true \
  -e MPLBACKEND=Agg \
  -e PYTHONPATH=/app \
  mspr-ai-nutrition \
  python scripts/eval_metrics.py llm \
    --n-generations 30 \
    --n-constraint-plans 30 \
    --seed 42
```

Pas de `--output-dir`, donc ecrit dans `docs/metrics.json` et regenere
`docs/metrics.md`.

## 7. Recuperer et envoyer les resultats

Les fichiers sont dans le bind-mount donc directement visibles sur le host :

```bash
ls -la docs/metrics.json docs/metrics.md \
       docs/eval_runs/with_fewshot/ \
       docs/eval_runs/without_fewshot/
```

Deux options pour les renvoyer a Arthur :

- **PR depuis ton fork** : forker le repo sur GitHub, push une branche
  `eval-gpu-n30` avec les fichiers `docs/metrics.json`, `docs/metrics.md`,
  `docs/eval_runs/with_fewshot/*`, `docs/eval_runs/without_fewshot/*`, ouvrir
  une PR sur master.
- **Archive** : `tar czf eval-gpu-n30.tar.gz docs/metrics.json docs/metrics.md docs/eval_runs/`,
  envoyer via Slack/email.

Les etapes 3 et 4 ne modifient plus aucun fichier versionne (slicing pilote
par `FEW_SHOT_FULL_EXAMPLES`, deploy GPU dans `docker-compose.gpu-eval.yml`) :
rien a exclure de la PR.

## 8. Verification rapide avant envoi

Avant d'envoyer, ouvrir `docs/metrics.md` et verifier que la section
"Impact du few-shot" affiche bien :

- `n=30` (et non `n=5`)
- des `latency_p50_ms` autour de 10 000 a 30 000 ms (GPU) au lieu de 200 000+ ms (CPU)
- un tableau comparatif avec / sans qui montre des deltas mesurables sur
  `constraint_satisfaction (full)`, `partial_compliance` et `abandoned_503`.

Si le tableau dit toujours `n=5` ou que la latence est dans les 100 000+ ms,
c'est que le GPU n'a pas ete utilise (Ollama est tombe sur CPU). Verifier
`docker exec mspr-ollama nvidia-smi` pendant un run : la VRAM doit augmenter
de quelques Go quand le modele est charge.

## 9. (Optionnel) Generer le dataset HITL et le noter

L'eval HITL (Human In The Loop) du PDF MSPR2 demande une notation qualitative
de plans LLM sur 3 dimensions (`nutrition`, `originalite`, `coherence`, echelle
1 a 5). Le script `scripts/generate_hitl_dataset.py` produit 20 plans avec les
memes inputs que l'eval automatique (seed 42), pretes a etre annotees.

Cette etape est optionnelle : le PDF ne l'exige pas explicitement. Si tu la
fais, elle s'integre dans le bloc HITL du `metrics.md`.

### 9.1 Generer les plans

```bash
docker exec mspr-ai-nutrition \
  python scripts/generate_hitl_dataset.py
```

Le script ecrit 3 fichiers dans `data/hitl/` :

- `plans.jsonl` : 1 plan par ligne, format machine-readable
- `plans.md` : version Markdown lisible (avec les contraintes en `<details>`
  pour permettre une notation en aveugle)
- `ratings_template.csv` : `plan_id,nutrition,originalite,coherence` avec
  les `plan_id` deja remplis et les notes vides

Duree : 20 plans x 5 a 30 s = 2 a 10 min sur GPU.

### 9.2 Noter les plans (3 voies possibles)

**Voie A : annotation humaine manuelle**

Ouvrir `data/hitl/plans.md` dans un editeur, lire chaque plan sans deplier
le bloc `<details>` (les contraintes), noter en aveugle sur les 3 dimensions
dans `data/hitl/ratings_template.csv`, puis verifier les notes en depliant
le bloc contraintes.

**Voie B : LLM-as-Judge via Google Colab ou Anthropic / OpenAI API**

Pour eviter le travail manuel, on peut faire noter les plans par un LLM
externe (Claude, GPT-4, Gemini Pro). C'est defendable scientifiquement en
le declarant explicitement comme proxy HITL (cf. Zheng et al. 2023,
"Judging LLM-as-a-Judge"). Workflow :

1. Recuperer `data/hitl/plans.jsonl` sur ton poste (ou Colab).
2. Pour chaque plan, envoyer le contenu a l'API LLM avec un prompt du type :

   ```
   Tu es un nutritionniste expert. Note ce plan repas (servi a l'utilisateur)
   sur 3 dimensions, echelle 1 a 5. Reponds en JSON :
   {"nutrition": int, "originalite": int, "coherence": int, "rationale": str}.

   nutrition  : coherence nutritionnelle (macros equilibrees pour l'objectif)
   originalite: variete des repas, pas de repetition
   coherence  : ingredients et nom du repas correspondent logiquement

   Plan a noter :
   <JSON du plan>

   Contraintes utilisateur (a respecter) :
   <JSON inputs>
   ```

3. Stocker les notes dans `data/hitl/ratings_template.csv`.

Un exemple de notebook Colab minimal :

```python
import json, csv, anthropic   # ou openai
client = anthropic.Anthropic(api_key="...")
plans = [json.loads(l) for l in open("plans.jsonl")]
with open("ratings.csv", "w") as fh:
    w = csv.writer(fh)
    w.writerow(["plan_id", "nutrition", "originalite", "coherence"])
    for p in plans:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": _build_prompt(p)}],
        )
        notes = json.loads(msg.content[0].text)
        w.writerow([p["plan_id"], notes["nutrition"],
                    notes["originalite"], notes["coherence"]])
```

**Voie C : panel HITL via amis / collegues**

Faire annoter par 2 ou 3 personnes en parallele, faire la moyenne. Plus
rigoureux statistiquement, plus de travail.

### 9.3 Reinjecter les notes et regenerer le rapport

Une fois `ratings_template.csv` rempli, le copier en place et relancer
l'eval LLM pour integrer le bloc HITL dans `metrics.md` :

```bash
cp data/hitl/ratings_template.csv docs/llm_hitl_ratings.csv

docker exec \
  -e FEW_SHOT_ENABLED=true \
  -e MPLBACKEND=Agg \
  -e PYTHONPATH=/app \
  mspr-ai-nutrition \
  python scripts/eval_metrics.py llm \
    --n-generations 30 \
    --n-constraint-plans 30 \
    --seed 42
```

Le bloc `Evaluation qualitative humaine (HITL, 1-5)` de `docs/metrics.md`
affichera desormais les moyennes calculees au lieu de `0.00`.

Ne pas oublier de mettre a jour `docs/llm_hitl_README.md` pour declarer la
methodologie reellement utilisee (humaine / LLM-as-Judge / panel).

## 10. Cleanup

```bash
docker compose down
docker volume rm mspr-healthai-coach-ai-nutrition_ollama_data  # optionnel
docker network rm mspr_data_network                            # si pas reutilise
```

## Probleme connu : timeout Ollama

Le runner utilise un timeout de 180 s par appel Ollama
(`_OLLAMA_TIMEOUT_S` dans `scripts/eval/llm_runner.py` et dans
`app/services/decrim_retry_orchestrator.py`). Sur GPU, un appel typique dure
5 a 30 s, donc 180 s est largement suffisant. Si jamais des timeouts
apparaissent : `docker logs mspr-ollama --tail 200` pour diagnostiquer (souvent
un cold-start sur le 1er appel apres demarrage du container).
