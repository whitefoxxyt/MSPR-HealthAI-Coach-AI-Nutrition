# Evaluation qualitative humaine (HITL) du LLM

## Objectif

Le PDF MSPR2 demande une evaluation qualitative complementaire aux 4 metriques
quantitatives (validite JSON, latence, fallback, respect des contraintes).
Cette eval consiste a noter ~20 plans generes par `gemma3:4b` sur 3 dimensions :

| Dimension     | Echelle | Question                                                            |
|---------------|---------|---------------------------------------------------------------------|
| `nutrition`   | 1-5     | Le plan est-il nutritionnellement coherent (macros equilibrees) ?   |
| `originalite` | 1-5     | Les repas proposes sont-ils varies, pas repetitifs ?                |
| `coherence`   | 1-5     | Les ingredients et le nom du repas correspondent-ils logiquement ?  |

## Workflow recommande

L'eval HITL necessite un environnement complet (Ollama + PostgreSQL) pour
generer les plans avec les memes inputs que l'eval automatique. Sur CPU, cela
prend 30 a 60 minutes ; sur GPU, 2 a 10 minutes. Voir `GPU_EVAL_PLAYBOOK.md`
section 9 pour le pas-a-pas.

Etapes :

1. **Generer le dataset** : `python scripts/generate_hitl_dataset.py` produit
   `data/hitl/plans.jsonl`, `data/hitl/plans.md` et
   `data/hitl/ratings_template.csv`.
2. **Annoter** : remplir `ratings_template.csv` selon l'une des 3 voies
   ci-dessous.
3. **Reinjecter** : copier le CSV vers `docs/llm_hitl_ratings.csv`, relancer
   `python scripts/eval_metrics.py llm` pour mettre a jour `docs/metrics.md`.

## Trois voies pour l'annotation

### Voie A : annotation humaine manuelle

Le mode le plus rigoureux. Ouvrir `data/hitl/plans.md` dans un editeur, noter
chaque plan en aveugle (sans deplier le bloc `<details>` qui contient les
contraintes), puis verifier les contraintes apres avoir donne la note.

Notation a froid : ne pas annoter directement apres avoir lu le prompt
(biais de confirmation). Lire chaque plan en aveugle, sans deplier le bloc
contraintes.

### Voie B : LLM-as-Judge

Pour eviter le travail manuel, faire noter par un LLM externe (Claude, GPT-4,
Gemini Pro). Pratique documentee dans la litterature ML (Zheng et al. 2023,
"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS).

Methodologie a declarer explicitement dans le rapport et dans la section
Discussion de `metrics.md` : "Notation realisee par un LLM externe en proxy
d'evaluation humaine. Les chiffres servent d'indicateur grossier, pas de
verite terrain". Le PDF MSPR2 reste satisfait des lors que la methodologie
est transparente.

Voir `GPU_EVAL_PLAYBOOK.md` section 9.2 pour un exemple de notebook minimal.

### Voie C : panel HITL (2 ou 3 annotateurs)

Faire annoter par plusieurs personnes en parallele, moyenner les CSV.
Reduit le biais individuel. Demande 2 a 4 heures de travail au total.

## Format CSV

`docs/llm_hitl_ratings.csv` :

```
plan_id,nutrition,originalite,coherence
1,4,3,5
2,5,2,4
...
```

`plan_id` est l'identifiant donne par `generate_hitl_dataset.py` (1 a 20).
Les plans manifestement invalides (fallback statique, generation echouee)
peuvent etre laisses dans le CSV avec des notes basses, ou supprimes : le
loader (`scripts/eval/llm_metrics.py:load_hitl_ratings`) calcule simplement
la moyenne des lignes presentes.

## Plans evaluees : alignement avec l'eval automatique

`generate_hitl_dataset.py` utilise `seed=42` et reproduit la logique de
`scripts/eval/llm_runner._random_inputs(with_constraints=True)`. Les 20 plans
HITL sont donc exactement les 20 premiers plans evalues par
`_run_pipeline_eval`. Cela permet de croiser les chiffres quantitatifs
(`constraint_satisfaction`, `compliance_status`) avec les notes qualitatives :
un plan note 5/5 mais en `partial_budget` revele un compromis acceptable
visuel ; un plan note 2/5 mais en `compliance_status=full` revele que le LLM
peut etre formellement conforme mais qualitativement faible.
