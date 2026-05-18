# Eval terrain : photos hors-distribution Food-101

## Objectif

Mesurer la generalisation du classifier `nateraw/food` (fine-tune Food-101)
sur des images **differentes de Food-101** : photos amateures, conditions
d'eclairage variables, plats hors des 101 classes du benchmark academique.

Le PDF MSPR2 (section IV livrable 1) demande des metriques IA defendables.
Une evaluation uniquement sur le test split Food-101 est sujette au biais
benchmark : memes conditions d'acquisition (studio, cadrage neutre) que le
train split. L'eval terrain mesure le delta entre `top-1 / top-5` academique
et performance hors studio.

## Dataset utilise : Matthijs/snacks (HuggingFace)

| Critere | Detail |
|---|---|
| Source | https://huggingface.co/datasets/Matthijs/snacks |
| Licence | CC-BY 4.0 (annotations) / CC-BY 2.0 (images Google Open Images) |
| Classes | 20 classes de snacks (apple, banana, cake, candy, carrot, cookie, doughnut, grape, hot dog, ice cream, juice, muffin, orange, pineapple, popcorn, pretzel, salad, strawberry, waffle, watermelon) |
| Provenance des photos | Google Open Images (web crawl), photos amateures (vs. studio Food-101) |
| Total images | 6 745 (train + val + test) |

Justification du choix : licence permissive compatible avec un repo public,
photos amateures qui simulent les conditions reelles d'usage du service
(photo de repas prise au telephone), 3 classes mappent directement vers
Food-101 (`doughnut` -> `donuts`, `hot dog` -> `hot_dog`, `ice cream` ->
`ice_cream`) pour mesurer top-1/top-5, 17 autres permettent de mesurer
`unknown_rate` (taux d'images hors distribution).

## Reproductibilite

Le sous-echantillon est genere par `scripts/download_terrain_dataset.py`
avec seed 42. Il extrait 49 images depuis l'archive HF :

- 5 images par classe Food-101-compatible (doughnut, hot dog, ice cream)
- 2 images par classe `unknown` (17 classes x 2 = 34)
- Total : 15 + 34 = 49 images

Re-generer le sous-echantillon :

```bash
python scripts/download_terrain_dataset.py
```

Le script tolere une seconde execution : il reutilise le zip s'il est deja
present dans `.cache/`, et reecrit le contenu de `images/` + `labels.csv`.

## Format

- `images/IMG_<class>_<idx>.jpg` : photos JPEG, plus petit cote 256 px,
  EXIF supprimes (par le dataset source).
- `labels.csv` : `filename,label_food101` ou `label_food101` est soit une
  classe Food-101 (en `snake_case`), soit `unknown` pour les plats hors
  distribution.

## Lancer l'eval terrain

L'eval terrain est integree au runner classifier
(`scripts/eval/classifier_runner.py:_eval_terrain`). Lancer dans le container
ai-nutrition (transformers + torch + datasets requis) :

```bash
docker exec mspr-ai-nutrition \
  python scripts/eval_metrics.py classifier \
    --n-food101 1000 \
    --terrain-dir data/eval_terrain \
    --seed 42
```

Le payload `classifier.terrain` est ajoute dans `docs/metrics.json` :

```json
{
  "n_samples": 49,
  "n_classified": 15,
  "top1_accuracy": <X>,
  "top5_accuracy": <X>,
  "unknown_rate": 0.69
}
```

`n_classified` = 15 (les images mappees vers Food-101), les 34 `unknown` sont
exclues du calcul top-1 / top-5 mais comptent dans `unknown_rate`.

## Limites connues

- Les 15 images "mappees" couvrent seulement 3 classes Food-101. Le top-1/top-5
  ne mesure donc pas la generalisation cross-domain de toutes les 101 classes,
  mais uniquement de 3 classes representatives (snack rapide, glace, donut).
- Les images viennent de Google Open Images : amateures mais pas strictement
  "photos telephone du jour-meme". C'est un proxy honnete, pas une mesure
  exacte des conditions d'usage de l'app HealthAI Coach.
- Pour aller plus loin : capturer 50-100 photos terrain reelles (telephone,
  conditions d'eclairage variees, plats francais traditionnels) et ajouter
  les fichiers/lignes au CSV. Le format est identique.
