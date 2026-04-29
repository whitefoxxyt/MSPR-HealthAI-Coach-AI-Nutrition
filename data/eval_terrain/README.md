# Photos terrain (eval HITL classifier)

## Objectif

50 photos prises au telephone (repas reels, restaurants, plats maison)
avec annotation manuelle, pour mesurer l'ecart entre l'accuracy academique
(Food-101 test split) et la realite des photos utilisateur.

Cible : 50 photos couvrant un mix de classes Food-101.

## Format

- Photos : `.jpg` ou `.png` dans ce dossier (`data/eval_terrain/`)
- Annotations : `labels.csv` au format
  ```
  filename,label
  IMG_001.jpg,pizza
  IMG_002.jpg,sushi
  ```
- `label` doit etre une classe Food-101 (snake_case, ex: `grilled_salmon`,
  `caesar_salad`). Liste complete : voir `nateraw/food` sur HuggingFace.
- Si la photo represente un plat hors des 101 classes, l'omettre :
  l'eval ne couvre que la distribution Food-101.

## Workflow

1. Prendre 50 photos varie : eclairage, angle, plats composites,
   restaurants, cantine, maison.
2. Renommer en `IMG_NNN.jpg` (3 chiffres) pour rester ordonne.
3. Editer `labels.csv` en ajoutant une ligne par photo.
4. Lancer `python scripts/eval_metrics.py classifier --terrain-dir data/eval_terrain/`.

## Reproductibilite

L'annotation est manuelle donc subjective : pour des plats composites
(ex: assiette avec pates + salade), choisir la classe dominante
visuellement. Documenter les choix ambigus dans un commit message ou
en commentaire sur le PR.
