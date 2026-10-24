"""Stable quantizer identity across Python AST dump-format versions."""

import ast
import hashlib
import json


def digest(node):
    def canonical(value):
        if isinstance(value, ast.AST):
            return {"node": type(value).__name__, **{
                key: canonical(item) for key, item in ast.iter_fields(value)
                if (item is not None and item != []) or key == "value"
            }}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value
    encoded = json.dumps(canonical(node), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
