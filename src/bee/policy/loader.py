from __future__ import annotations

from pathlib import Path

from bee.policy.rules import Policy, validate_policy_yaml, load_policy_from_dict


def load_policy(path: Path) -> Policy:
    """Load a BEE policy from a YAML file.

    The policy file can be either:
    - Top-level keys directly (e.g., `name: my-policy`)
    - Wrapped in a `policy:` key (e.g., `policy:\n  name: my-policy`)
    """
    yaml_text = path.read_text()
    data = validate_policy_yaml(yaml_text)
    return load_policy_from_dict(data)


def default_policy() -> Policy:
    """Return the default policy (empty blocks, all findings allowed)."""
    from bee.policy.rules import Policy
    return Policy()
