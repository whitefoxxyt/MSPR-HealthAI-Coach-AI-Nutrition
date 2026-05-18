# Metriques IA : MSPR-AI-Nutrition

Genere par `scripts/eval_metrics.py`. Ne pas editer a la main.

## LLM (Ollama gemma3:4b)

### Niveau naive (LLM nu, sans DeCRIM-light)

- Taux de validite JSON (1er essai) : 1.0000
- Taux d'invocation Fallback : 0.0000
- Latence : p50 86101 ms, p95 95965 ms, max 95965 ms
- Respect simultanee allergies + budget + regime : 0.0000
- Par contrainte : allergies 1.0000, budget 0.0000, regime 0.8000

#### Evaluation qualitative humaine (HITL, 1-5)

- N ratings : 0
- Pertinence nutrition : 0.00
- Originalite : 0.00
- Coherence : 0.00

![Distribution latence LLM](docs/eval_runs/without_fewshot/llm_latency_distribution.png)

### Niveau pipeline (DeCRIM-light + cache bypass)

- N generations : 5
- compliance_status full : 0.2000
- compliance_status partial_budget : 0.4000
- compliance_status static_fallback : 0.0000
- abandoned_503 (contraintes infaisables) : 0.4000
- Par contrainte : allergies 1.0000, budget 0.3333, regime 1.0000
- Latence : p50 249907 ms, p95 296619 ms
- Distribution retries : 3 retry: 5

### Comparaison naive vs pipeline

- Gain compliance_status full vs naive : +0.2000. Un gain positif quantifie l'apport du retry cible (allergie/regime partiels, budget complet) et du fallback hierarchique.
- Delta allergies : naive 1.0000 -> pipeline 1.0000 (+0.0000)
- Delta budget : naive 0.0000 -> pipeline 0.3333 (+0.3333)
- Delta diet : naive 0.8000 -> pipeline 1.0000 (+0.2000)
- Surcout latence p95 : naive 95965 ms -> pipeline 296619 ms (ratio x3.09). Le pipeline paie le prix des retries internes pour reduire les violations critiques (allergies/regime).

## Discussion

- **Limitations dataset Food-101** : 101 classes academiques, photos cadrees, fond neutre. Tres different des photos prises au telephone (eclairage, angle, plat composite).
- **Biais du modele** : fine-tune sur Food-101 -> classes hors-distribution (ex : plats francais traditionnels, repas ethniques specifiques) sont systematiquement misclassifies vers la classe la plus proche visuellement.
- **Cas d'echec frequents** : plats mixtes (assiette avec plusieurs aliments), decoupes inhabituelles, photos en faible luminosite, gros plans non cadres.
- **LLM** : la latence p95 sur CPU reste contraignante ; le fallback statique garantit une UX correcte hors disponibilite Ollama. Les violations de contraintes proviennent souvent du regime alimentaire (vegan/sans gluten moins bien respectes que les allergies).

