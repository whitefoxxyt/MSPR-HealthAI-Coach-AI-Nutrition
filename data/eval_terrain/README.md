# Photos terrain (eval HITL classifier)

## Objectif

50 a 100 photos prises au telephone (repas reels, restaurants, plats maison)
avec annotation manuelle, pour mesurer l'ecart entre l'accuracy academique
(Food-101 test split) et la realite des photos utilisateur.

Cible : 50 a 100 photos couvrant un mix de classes Food-101 + quelques plats
hors-distribution (label `unknown`) pour mesurer le taux de plats absents
de Food-101.

## Format

- Photos : `.jpg` ou `.png` dans `data/eval_terrain/images/`
- Annotations : `data/eval_terrain/labels.csv` au format
  ```
  filename,label_food101
  IMG_001.jpg,pizza
  IMG_002.jpg,sushi
  IMG_003.jpg,unknown
  ```
- `label_food101` doit etre une classe Food-101 (snake_case, ex:
  `grilled_salmon`, `caesar_salad`). Liste complete : voir `nateraw/food`
  sur HuggingFace.
- Si la photo represente un plat hors des 101 classes, utiliser le label
  special `unknown` : il sera comptabilise dans `unknown_rate` mais exclu
  des metriques top-1 / top-5.

## Workflow

1. Prendre 50 a 100 photos variees : eclairage, angle, plats composites,
   restaurants, cantine, maison, plats hors Food-101.
2. Renommer en `IMG_NNN.jpg` (3 chiffres) pour rester ordonne.
3. Deposer les photos dans `data/eval_terrain/images/`.
4. Editer `labels.csv` en ajoutant une ligne par photo.
5. Lancer `python scripts/eval_metrics.py classifier --terrain-dir data/eval_terrain/`.

## Reproductibilite

L'annotation est manuelle donc subjective : pour des plats composites
(ex: assiette avec pates + salade), choisir la classe dominante
visuellement. Documenter les choix ambigus dans un commit message ou
en commentaire sur le PR.
