# Metriques IA : MSPR-AI-Nutrition

Genere par `scripts/eval_metrics.py`. Ne pas editer a la main.

## Classifier (HuggingFace nateraw/food)

### Food-101 (test split, sous-echantillon)

- N samples : 1000
- Top-1 accuracy : 0.8880
- Top-5 accuracy : 0.9720

![Matrice de confusion Food-101](docs/confusion_matrix_food101.png)

Top classes (precision / rappel / F1 / support) :

| Classe | Precision | Rappel | F1 | Support |
|---|---|---|---|---|
| apple_pie | 0.000 | 0.000 | 0.000 | 0 |
| baby_back_ribs | 1.000 | 0.846 | 0.917 | 26 |
| baklava | 0.882 | 0.833 | 0.857 | 18 |
| beignets | 0.000 | 0.000 | 0.000 | 0 |
| bibimbap | 1.000 | 0.880 | 0.936 | 25 |
| bread_pudding | 0.938 | 0.750 | 0.833 | 20 |
| breakfast_burrito | 1.000 | 1.000 | 1.000 | 13 |
| bruschetta | 0.000 | 0.000 | 0.000 | 0 |
| cannoli | 1.000 | 0.935 | 0.967 | 31 |
| caprese_salad | 1.000 | 1.000 | 1.000 | 15 |
| ceviche | 0.944 | 0.773 | 0.850 | 22 |
| cheese_plate | 0.955 | 0.955 | 0.955 | 22 |
| cheesecake | 0.889 | 0.800 | 0.842 | 10 |
| chicken_curry | 0.955 | 0.750 | 0.840 | 28 |
| chicken_wings | 0.000 | 0.000 | 0.000 | 0 |
| chocolate_cake | 0.000 | 0.000 | 0.000 | 0 |
| chocolate_mousse | 0.000 | 0.000 | 0.000 | 0 |
| churros | 0.938 | 0.938 | 0.938 | 16 |
| crab_cakes | 0.750 | 0.750 | 0.750 | 8 |
| croque_madame | 1.000 | 0.941 | 0.970 | 17 |
| cup_cakes | 0.955 | 0.913 | 0.933 | 23 |
| deviled_eggs | 1.000 | 0.967 | 0.983 | 30 |
| donuts | 1.000 | 0.895 | 0.944 | 19 |
| edamame | 1.000 | 1.000 | 1.000 | 29 |
| eggs_benedict | 0.000 | 0.000 | 0.000 | 0 |
| falafel | 0.000 | 0.000 | 0.000 | 0 |
| filet_mignon | 1.000 | 0.526 | 0.690 | 19 |
| fish_and_chips | 0.000 | 0.000 | 0.000 | 0 |
| foie_gras | 1.000 | 0.767 | 0.868 | 30 |
| french_onion_soup | 0.000 | 0.000 | 0.000 | 0 |
| french_toast | 0.000 | 0.000 | 0.000 | 0 |
| fried_calamari | 0.000 | 0.000 | 0.000 | 0 |
| fried_rice | 0.000 | 0.000 | 0.000 | 0 |
| frozen_yogurt | 0.938 | 0.789 | 0.857 | 19 |
| garlic_bread | 0.000 | 0.000 | 0.000 | 0 |
| gnocchi | 0.821 | 0.821 | 0.821 | 28 |
| greek_salad | 0.000 | 0.000 | 0.000 | 0 |
| grilled_cheese_sandwich | 0.947 | 1.000 | 0.973 | 18 |
| grilled_salmon | 1.000 | 0.844 | 0.915 | 32 |
| guacamole | 0.000 | 0.000 | 0.000 | 0 |
| gyoza | 1.000 | 0.867 | 0.929 | 30 |
| hamburger | 0.000 | 0.000 | 0.000 | 0 |
| hot_and_sour_soup | 0.957 | 1.000 | 0.978 | 22 |
| ice_cream | 0.833 | 0.833 | 0.833 | 18 |
| lasagna | 0.000 | 0.000 | 0.000 | 0 |
| lobster_roll_sandwich | 0.913 | 0.955 | 0.933 | 22 |
| macaroni_and_cheese | 0.950 | 0.905 | 0.927 | 21 |
| macarons | 1.000 | 1.000 | 1.000 | 17 |
| nachos | 0.950 | 0.826 | 0.884 | 23 |
| omelette | 0.955 | 0.778 | 0.857 | 27 |
| pad_thai | 0.000 | 0.000 | 0.000 | 0 |
| paella | 0.923 | 0.923 | 0.923 | 26 |
| panna_cotta | 0.000 | 0.000 | 0.000 | 0 |
| peking_duck | 0.882 | 0.938 | 0.909 | 16 |
| pho | 1.000 | 0.929 | 0.963 | 14 |
| pizza | 0.000 | 0.000 | 0.000 | 0 |
| pork_chop | 0.000 | 0.000 | 0.000 | 0 |
| poutine | 1.000 | 0.923 | 0.960 | 26 |
| pulled_pork_sandwich | 0.935 | 0.906 | 0.921 | 32 |
| ramen | 0.000 | 0.000 | 0.000 | 0 |
| ravioli | 0.792 | 0.792 | 0.792 | 24 |
| red_velvet_cake | 1.000 | 0.950 | 0.974 | 20 |
| risotto | 0.000 | 0.000 | 0.000 | 0 |
| samosa | 0.000 | 0.000 | 0.000 | 0 |
| sashimi | 0.000 | 0.000 | 0.000 | 0 |
| scallops | 0.000 | 0.000 | 0.000 | 0 |
| seaweed_salad | 1.000 | 1.000 | 1.000 | 17 |
| shrimp_and_grits | 0.000 | 0.000 | 0.000 | 0 |
| spaghetti_bolognese | 1.000 | 1.000 | 1.000 | 28 |
| spaghetti_carbonara | 0.966 | 0.966 | 0.966 | 29 |
| steak | 0.000 | 0.000 | 0.000 | 0 |
| strawberry_shortcake | 0.000 | 0.000 | 0.000 | 0 |
| sushi | 0.952 | 0.909 | 0.930 | 22 |
| tacos | 1.000 | 0.923 | 0.960 | 26 |
| takoyaki | 0.955 | 0.955 | 0.955 | 22 |
| tuna_tartare | 0.000 | 0.000 | 0.000 | 0 |

## LLM : comparaison multi-backend

### Comparaison Gemma3:4b local vs Mistral Small managed

Gemma3:4b evalue N=30 (seed 42) le 2026-06-10 sur GPU : serveur Unraid Zespri, Quadro P1000 4 Go, Ollama 100 % GPU (`num_gpu=99`, `num_ctx=8192`), few-shot complet 7j/5j/1j (`FEW_SHOT_FULL_EXAMPLES=true`). Mistral N=20, meme seed.

| Metrique | Gemma3:4b local | Mistral Small managed |
|---|---|---|
| compliance_status=full | 0.3333 | 0.6500 |
| allergy compliance rate | 1.0000 | 1.0000 |
| diet compliance rate | 1.0000 | 1.0000 |
| JSON validity rate | 1.0000 | 1.0000 |
| latence p50 (pipeline) | 108597 ms | 2983 ms |
| latence p95 (pipeline) | 153655 ms | 9744 ms |
| retry count moyen | 2.40 | 1.40 |

L'ablation few-shot (N=30 par condition, details dans `docs/eval_runs/{with,without}_fewshot/`) montre que les exemples few-shot debloquent la conformite budget de Gemma : sans few-shot, 0.00 de plans full (budget 0.00, 30/30 generations epuisent les 3 retries) ; avec few-shot complet, 0.33 de plans full et 0.45 a 0.48 de respect budget, avec un p50 pipeline plus bas (95 a 109 s contre 136 s).

### Mistral Small managed (backend par defaut)

Pipeline complet (validator DeCRIM-light + retry + fallback statique) sur le seed 42, allergies / regime / budget tires aleatoirement parmi le pool de contraintes de l'eval.

| Indicateur | Mistral N=20 | Mistral N=100 |
|------------|--------------|----------------|
| `compliance_status=full` | 0.65 | 0.71 |
| `partial_compliance` (budget relache) | 0.05 | 0.03 |
| `static_fallback` | 0.00 | 0.00 |
| JSON validity rate (naive) | 1.00 | 1.00 |
| Latence pipeline p50 (ms) | 2 982 | 4 163 |
| Latence pipeline p95 (ms) | 9 743 | 9 760 |
| `abandoned_503` (echecs apres retries) | 0.30 | 0.26 |

Le N=100 confirme l'ordre de grandeur du N=20 : la dispersion sur 100 generations reste alignee (compliance_satisfaction = 0.71 vs 0.65, p95 quasi identique). L'eval est mature, ce n'est pas un artefact de petit echantillon.

`json_validity_rate = 1.0` sur les deux echantillons valide le choix Mistral `response_format=json_schema strict:true`. Le LLM ne produit jamais de JSON malforme.

### Gemma3:4b local (fallback / mode offline)

**Pas de chiffres N=20 sur cette infra CPU.** Le prompt few-shot livre au slice 9 (#69) compte ~2245 tokens. Trois runs successifs ont conclu : timeout systematique meme apres bump de `_OLLAMA_TIMEOUT_S` a 180 s, Ollama abandonne avec 500 / aborted avant d'emettre le premier token (cf. `TODO_DEMAIN.md` 2026-04-30 et l'issue PRD #71). C'est precisement ce constat qui a declenche le pivot vers Mistral en primaire.

Gemma reste branche dans `LLMProvider` et adresse trois cas hors UX interactive :

- **Eval differee sur GPU** : un host GPU (cf. `GPU_EVAL_PLAYBOOK.md`) peut produire les chiffres comparatifs hors du chemin temps reel.
- **Mode offline / on-premise** : deploiement sans `MISTRAL_API_KEY`, par exemple en environnement hospitalier qui refuse l'exfiltration de donnees sante.
- **Backend de fallback** : si Mistral renvoie 5xx / 429 / timeout, la `FallbackChain` route sur Ollama avant le static_fallback, et taggue le plan avec `compliance_warning` explicitant la bascule.

### Bonus : naive vs pipeline (Mistral N=100)

Effet du validator DeCRIM-light + retry mesure sur le meme echantillon :

| Indicateur | Naive (1 call, pas de validation) | Pipeline (validator + retry) |
|------------|-----------------------------------|------------------------------|
| `constraint_satisfaction` global | 0.33 | 0.71 |
| Allergies respectees | 0.67 | 1.00 |
| Diete respectee | 0.63 | 1.00 |
| Budget respecte | 0.76 | 0.96 |
| Latence p50 (ms) | 4 022 | 4 163 |
| Latence p95 (ms) | 5 614 | 9 760 |

DeCRIM-light gagne +38 points sur la conformite globale, au prix de +4 s sur le p95 (retries declenches par les violations). Allergies et diete passent a 1.00 grace au validator, le budget reste imparfait car relache explicitement (`partial_budget`) plutot que de bloquer la generation.

## Discussion

- **Limitations dataset Food-101** : 101 classes academiques, photos cadrees, fond neutre. Tres different des photos prises au telephone (eclairage, angle, plat composite).
- **Biais du modele** : fine-tune sur Food-101 -> classes hors-distribution (ex : plats francais traditionnels, repas ethniques specifiques) sont systematiquement misclassifies vers la classe la plus proche visuellement.
- **Cas d'echec frequents** : plats mixtes (assiette avec plusieurs aliments), decoupes inhabituelles, photos en faible luminosite, gros plans non cadres.
- **LLM** : la latence p95 sur CPU reste contraignante ; le fallback statique garantit une UX correcte hors disponibilite Ollama. Les violations de contraintes proviennent souvent du regime alimentaire (vegan/sans gluten moins bien respectes que les allergies).

### Mistral Small managed vs Gemma3:4b local

**Mistral gagne sur** :

- **Latence** : ordre de grandeur d'avance (quelques secondes p50 vs plusieurs dizaines de secondes sur CPU). Permet une UX interactive sur le flux generate-meal-plan.
- **Validite JSON** : le mode `response_format.json_schema strict:true` garantit un JSON syntaxiquement valide des le 1er essai. Gemma3:4b via Ollama `format: <schema>` reste tributaire de la generation libre.
- **Conformite aux contraintes** : sur les memes inputs (seed=42), le compliance_status=full atteint un taux significativement plus eleve, ce qui reduit la frequence des fallback statiques.

**Gemma3:4b reste pertinent pour** :

- **Offline / on-premise** : aucune dependance reseau, aucun token expedier a un fournisseur externe. Atout pour une instance enterprise hospitaliere / mutuelle qui refuse l'externalisation des donnees nutrition.
- **Privacy** : les inputs (allergies, regime, budget) restent dans le perimetre du deploiement. Pertinent pour des donnees de sante au sens RGPD (article 9, donnees concernant la sante).
- **Cout long terme** : pas de quota par requete. Pour un usage massif, le cout d'inference plafonne au cout CPU/GPU local. Mistral free tier n'est pas dimensionne pour de la prod a fort QPS.

Le selecteur utilisateur introduit au slice 3 (`PATCH /me/preferences`) permet de respecter ces deux profils sans contraindre l'instance.

