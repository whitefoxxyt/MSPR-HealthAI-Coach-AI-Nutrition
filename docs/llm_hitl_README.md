# Evaluation qualitative humaine (HITL) du LLM

## Objectif

Le PDF MSPR demande une evaluation qualitative complementaire aux 4 metriques
quantitatives (validite JSON, latence, fallback, respect des contraintes).
20 plans generes par `gemma3:4b` sont notes manuellement sur 3 dimensions :

| Dimension     | Echelle | Question                                                            |
|---------------|---------|---------------------------------------------------------------------|
| `nutrition`   | 1-5     | Le plan est-il nutritionnellement coherent (macros equilibrees) ?   |
| `originalite` | 1-5     | Les repas proposes sont-ils varies, pas repetitifs ?                |
| `coherence`   | 1-5     | Les ingredients et le nom du repas correspondent-ils logiquement ?  |

## Format

`llm_hitl_ratings.csv` :

```
plan_id,nutrition,originalite,coherence
1,4,3,5
2,5,2,4
...
```

`plan_id` est l'id retourne par `POST /api/v1/generate-meal-plan` lors de la
generation. Il n'est pas necessaire que les ids soient consecutifs : on peut
exclure des plans manifestement invalides (fallback, JSON invalide).

## Workflow

1. Lancer 20 generations representatives (varier objective + diet) :
   ```
   python scripts/eval_metrics.py llm --n-generations 20
   ```
2. Recuperer les `plan_id` depuis la BDD (`SELECT id, plan FROM meal_plans
   ORDER BY generated_at DESC LIMIT 20`) ou via l'historique API.
3. Imprimer ou afficher le contenu de chaque plan.
4. Noter chaque plan sur les 3 dimensions, remplir `llm_hitl_ratings.csv`.
5. Relancer `python scripts/eval_metrics.py llm` : le CSV est integre dans
   `docs/metrics.json` + `docs/metrics.md`.

## Notes

- Les notes sont subjectives : pour reduire le biais, faire noter par
  plusieurs personnes et moyenner les CSV.
- Notation a froid : ne pas annoter directement apres avoir lu le prompt
  (biais de confirmation). Lire chaque plan en aveugle.
