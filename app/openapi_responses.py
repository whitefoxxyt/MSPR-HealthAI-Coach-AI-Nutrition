from __future__ import annotations

# Reponses OpenAPI baseline imposees par l'AC de l'issue 56 (livrable #6 MSPR2).
# Tous les endpoints metier (prefixe /api/v1/) doivent declarer ces codes en plus
# de leurs reponses specifiques. Un code documente n'est pas necessairement
# atteignable sur chaque endpoint : 403 ne se declenche qu'avec des entitlements
# specifiques (tier premium par exemple), 503 uniquement sur les endpoints qui
# appellent Ollama. La declaration sert d'enveloppe de contrat pour les clients.

AC_BUSINESS_RESPONSES: dict[int, dict[str, str]] = {
    400: {"description": "Requete malformee (body / parametre illisible)."},
    401: {"description": "JWT manquant, malforme ou invalide."},
    403: {
        "description": (
            "Permission insuffisante (entitlements ou tier requis pour cette "
            "operation)."
        )
    },
    422: {
        "description": (
            "Validation Pydantic en echec (type ou contrainte de champ violee)."
        )
    },
    503: {
        "description": (
            "Service degrade : dependance externe (Ollama) injoignable et "
            "fallback indisponible."
        )
    },
}


def with_ac_baseline(*specifics: dict) -> dict:
    """Fusionne les reponses specifiques d'un endpoint avec la baseline AC.

    Les entrees de `specifics` ecrasent la baseline pour un meme code (ex. un
    endpoint qui veut une description 401 plus precise garde sa propre version).
    Les nouveaux codes (404, 413, 415, 429) sont conserves tels quels.
    """
    merged: dict = dict(AC_BUSINESS_RESPONSES)
    for spec in specifics:
        merged.update(spec)
    return merged
