#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Benchmark complet des modèles food classification
#
# Ce script :
#   1. Vérifie que le container mspr-ai-nutrition est démarré
#   2. Injecte les scripts Python dans le container
#   3. Installe la dépendance `datasets` (HuggingFace)
#   4. Télécharge 15 images de test depuis Food-101
#   5. Lance le benchmark sur les deux modèles candidats
#   6. Affiche le résumé et exporte les résultats JSON
#
# Usage :
#   chmod +x scripts/run_benchmark.sh
#   ./scripts/run_benchmark.sh
# =============================================================================

set -euo pipefail

CONTAINER="mspr-ai-nutrition"
RESULTS_FILE="docs/benchmark_results.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[benchmark]${NC} $*"; }
warn() { echo -e "${YELLOW}[benchmark]${NC} $*"; }
die()  { echo -e "${RED}[benchmark] ERREUR${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Vérification du container
# ---------------------------------------------------------------------------
log "Vérification du container '$CONTAINER'..."
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    die "Container '$CONTAINER' non trouvé. Lance d'abord : docker compose up -d"
fi
log "Container actif."

# ---------------------------------------------------------------------------
# 2. Injection des scripts dans le container
# ---------------------------------------------------------------------------
log "Copie des scripts dans le container..."
docker exec "$CONTAINER" mkdir -p /app/scripts /app/data/benchmark_images /app/docs
docker cp scripts/download_test_images.py "$CONTAINER":/app/scripts/download_test_images.py
docker cp scripts/benchmark_models.py     "$CONTAINER":/app/scripts/benchmark_models.py
log "Scripts copiés."

# ---------------------------------------------------------------------------
# 3. Installation de la dépendance datasets
# ---------------------------------------------------------------------------
log "Installation de 'datasets' (HuggingFace)..."
docker exec "$CONTAINER" pip install -q datasets
log "Dépendance installée."

# ---------------------------------------------------------------------------
# 4. Téléchargement des images Food-101
# ---------------------------------------------------------------------------
log "Téléchargement des images de test depuis Food-101..."
docker exec "$CONTAINER" python scripts/download_test_images.py \
    --output-dir data/benchmark_images \
    --n 15
log "Images prêtes."

# ---------------------------------------------------------------------------
# 5. Benchmark
# ---------------------------------------------------------------------------
log "Lancement du benchmark (nateraw/food vs Kaludi)..."
warn "Les modèles sont téléchargés au premier lancement (~350 Mo). Patience..."
docker exec "$CONTAINER" python scripts/benchmark_models.py \
    --images-dir data/benchmark_images \
    --output docs/benchmark_results.json

# ---------------------------------------------------------------------------
# 6. Export local des résultats
# ---------------------------------------------------------------------------
log "Récupération des résultats JSON..."
docker cp "$CONTAINER":/app/docs/benchmark_results.json "$RESULTS_FILE"
log "Résultats exportés dans ${RESULTS_FILE}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Benchmark terminé.${NC}"
echo -e "${GREEN}  Résultats complets : ${RESULTS_FILE}${NC}"
echo -e "${GREEN}========================================${NC}"
