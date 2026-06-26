"""Domino model-registry client (via MLflow).

Domino's model registry is MLflow-backed and auto-configured inside a workload,
so we talk to it with ``MlflowClient`` rather than reverse-engineering a REST
surface. Used by the Settings page to populate the model picker (list models →
list versions). Everything degrades gracefully: if the registry is unreachable
the picker shows an explanatory note instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegisteredModelInfo:
    name: str
    versions: list[str] = field(default_factory=list)
    latest_version: str = ""
    description: str = ""


@dataclass
class RegistryListing:
    models: list[RegisteredModelInfo] = field(default_factory=list)
    error: str = ""
    available: bool = True


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    # In a Domino workload the tracking/registry URIs are pre-set in the env;
    # MlflowClient() picks them up. We don't override them.
    return MlflowClient()


def list_models(max_models: int = 200) -> RegistryListing:
    """List registered models and their versions from the project registry."""
    try:
        client = _client()
    except Exception as exc:  # noqa: BLE001
        return RegistryListing(error=f"MLflow unavailable: {exc}", available=False)

    try:
        listing: list[RegisteredModelInfo] = []
        registered = client.search_registered_models(max_results=max_models)
        for rm in registered:
            versions = sorted(
                (mv.version for mv in getattr(rm, "latest_versions", []) or []),
                key=lambda v: int(v) if str(v).isdigit() else 0,
                reverse=True,
            )
            # Fill in the full version list when cheap; fall back to latest only.
            try:
                all_versions = client.search_model_versions(f"name='{rm.name}'")
                versions = sorted(
                    {mv.version for mv in all_versions},
                    key=lambda v: int(v) if str(v).isdigit() else 0,
                    reverse=True,
                )
            except Exception:  # noqa: BLE001
                pass
            listing.append(
                RegisteredModelInfo(
                    name=rm.name,
                    versions=[str(v) for v in versions],
                    latest_version=str(versions[0]) if versions else "",
                    description=getattr(rm, "description", "") or "",
                )
            )
        listing.sort(key=lambda m: m.name.lower())
        return RegistryListing(models=listing)
    except Exception as exc:  # noqa: BLE001
        return RegistryListing(error=f"Could not list registered models: {exc}", available=False)
