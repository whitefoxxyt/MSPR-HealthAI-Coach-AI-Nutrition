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

## LLM (Ollama gemma3:4b)

Slice 7 PRD #45 : on mesure deux niveaux sur les memes inputs aleatoires
(meme seed) pour quantifier l'apport du retry DeCRIM-light hybride.

### Niveau naive (LLM nu, sans DeCRIM-light)

- N generations : 30
- Taux de validite JSON (1er essai) : 1.0000
- Taux d'invocation Fallback : 0.0000
- Latence : p50 83654 ms, p95 95210 ms, max 97856 ms
- Respect simultanee allergies + budget + regime : 0.0000
- Par contrainte : allergies 0.0000, budget 0.0000, regime 0.0000

#### Evaluation qualitative humaine (HITL, 1-5)

- N ratings : 0
- Pertinence nutrition : 0.00
- Originalite : 0.00
- Coherence : 0.00

![Distribution latence LLM](docs/llm_latency_distribution.png)

### Niveau pipeline (DeCRIM-light + cache bypass)

> Numeros a peupler par le runner. Lancer
> `python scripts/eval_metrics.py llm --n-generations 30 --n-constraint-plans 30`
> avec Ollama et PostgreSQL accessibles (settings.ollama_host /
> settings.database_url). Le runner ecrit les valeurs reelles sous
> `docs/metrics.json` -> `llm.pipeline`.

- N generations : 0 (en attente de run)
- compliance_status full : 0.0000
- compliance_status partial_budget : 0.0000
- compliance_status static_fallback : 0.0000
- abandoned_503 (contraintes infaisables) : 0.0000
- Par contrainte : allergies 0.0000, budget 0.0000, regime 0.0000
- Latence : p50 0 ms, p95 0 ms
- Distribution retries : (vide)

### Comparaison naive vs pipeline

Une fois la passe pipeline executee, on attend :

- **constraint_satisfaction full > naive** : la boucle DeCRIM-light retire
  les ingredients allergenes via retry partiel (uniquement le repas violant)
  et reduit le budget des jours qui depassent via retry complet du jour.
  Le fallback hierarchique apporte un filet de securite supplementaire.
- **abandoned_503 faible** : seuls les couples (regime, allergies) reellement
  infaisables (ex. vegan strict + allergie a tous les substituts vegetaux)
  doivent declencher le 503. Sur le pool d'inputs aleatoires, on s'attend
  a moins de 5% de 503.
- **Surcout latence p95** : chaque generation peut declencher jusqu'a 4
  appels Ollama (initial + 2 retries partiels + 1 retry complet en garde-fou).
  Le ratio pipeline/naive p95 attendu est entre x1.5 et x3.0 sur CPU, selon
  la frequence des violations.
- **Repartition par contrainte** : la pipeline doit prioriser allergies/regime
  (criticite sante) sur le budget. On accepte des `partial_budget` plutot
  qu'un `static_fallback` quand seul le budget est viole.

## Discussion

- **Limitations dataset Food-101** : 101 classes academiques, photos cadrees, fond neutre. Tres different des photos prises au telephone (eclairage, angle, plat composite).
- **Biais du modele** : fine-tune sur Food-101 -> classes hors-distribution (ex : plats francais traditionnels, repas ethniques specifiques) sont systematiquement misclassifies vers la classe la plus proche visuellement.
- **Cas d'echec frequents** : plats mixtes (assiette avec plusieurs aliments), decoupes inhabituelles, photos en faible luminosite, gros plans non cadres.
- **LLM** : la latence p95 sur CPU reste contraignante ; le fallback statique garantit une UX correcte hors disponibilite Ollama. Avant slice 7, les violations de contraintes provenaient souvent du regime alimentaire (vegan/sans gluten moins bien respectes que les allergies). Le pipeline DeCRIM-light cible explicitement ces violations : retry partiel sur allergie/regime (n'invalide pas tout le plan), retry complet du jour sur budget, fallback hierarchique sur la matrice statique.

