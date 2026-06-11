# Metriques IA : MSPR-AI-Nutrition

Genere par `scripts/eval_metrics.py`. Ne pas editer a la main.

## LLM : comparaison multi-backend

### Comparaison Gemma3:4b local vs Mistral Small managed

_Tableau principal indisponible : un des deux runs N=20 manque (lancer `LLM_BACKEND=ollama` puis `LLM_BACKEND=mistral`)._

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

