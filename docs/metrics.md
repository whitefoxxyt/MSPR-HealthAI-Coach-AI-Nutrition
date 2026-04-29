# Metriques IA : MSPR-AI-Nutrition

Genere par `scripts/eval_metrics.py`. Ne pas editer a la main.

Ce fichier est un placeholder. Pour le remplir, executer :

```bash
pip install -r requirements-eval.txt
python scripts/eval_metrics.py classifier --n-food101 1000 --seed 42
python scripts/eval_metrics.py llm --n-generations 100 --seed 42
```

Le rapport final couvrira :

## Classifier (HuggingFace nateraw/food)

- **Food-101 (test split, sous-echantillon)** : accuracy top-1 / top-5,
  precision/rappel/F1 par classe, matrice de confusion (PNG).
- **Terrain (50 photos manuelles)** : meme structure, sur photos prises au
  telephone (cf. `data/eval_terrain/`).
- **Comparaison** : ecart academique vs realite, hypothese sur le biais de
  domaine de Food-101.

## LLM (Ollama gemma3:4b)

- Taux de validite JSON au 1er essai (`format: <json_schema>`).
- Latence p50 / p95 / max sur 100 generations.
- Taux d'invocation fallback (Ollama down ou timeout).
- Respect simultanee allergies + budget + regime sur 30 plans contraints.
- Evaluation qualitative humaine (HITL) sur 20 plans notes 1-5.

## Discussion

Limitations, biais Food-101, cas d'echec frequents. Section auto-generee.
