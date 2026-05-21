from __future__ import annotations

from typing import Any


def sanitize_for_mistral_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalise un schema Pydantic v2 pour `response_format.json_schema.strict=true`.

    Mistral en mode strict refuse les `$ref` / `$defs` non resolus et exige
    `additionalProperties: false` sur tous les objets. Cette fonction est
    pure et ne mute pas son entree.
    """
    defs = schema.get("$defs", {})
    resolved = _walk(schema, defs)
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
    return resolved


def _walk(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            target = _resolve_ref(ref, defs)
            return _walk(target, defs)
        result = {k: _walk(v, defs) for k, v in node.items() if k != "$defs"}
        if result.get("type") == "object":
            result["additionalProperties"] = False
        return result
    if isinstance(node, list):
        return [_walk(item, defs) for item in node]
    return node


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        raise ValueError(f"$ref non supporte : {ref}")
    name = ref[len(prefix) :]
    if name not in defs:
        raise ValueError(f"definition introuvable pour {ref}")
    return defs[name]
