"""Shared reproducibility primitives for all PMC experiment families."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentIdentity:
    """Stable identity for one experiment-method-target result."""

    experiment: str
    iteration: int
    method: str
    target: str
    seed: int
    n: int | None = None

    def key(self) -> str:
        return canonical_json(asdict(self))

    def digest(self) -> str:
        return hashlib.sha256(self.key().encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize values deterministically, independent of mapping order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def configuration_digest(configuration: Mapping[str, Any]) -> str:
    """Hash stable configuration fields while ignoring runtime timestamps."""
    stable = {
        key: value for key, value in configuration.items() if key not in {"timestamp"}
    }
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()
