"""Object profile enrollment helpers.

The package is intentionally light on import so CLI module execution does not
preload registry/scanner modules or touch camera-related state.
"""

__all__ = [
    "active_profile",
    "generic_object_profile",
    "list_profiles",
    "match_profiles",
    "profile_to_legacy_grasp_profile",
    "save_profile",
    "set_active_profile",
]


def __getattr__(name):
    if name in {"generic_object_profile", "match_profiles", "profile_to_legacy_grasp_profile"}:
        from . import object_profile

        return getattr(object_profile, name)
    if name in {"active_profile", "list_profiles", "save_profile", "set_active_profile"}:
        from . import object_registry

        return getattr(object_registry, name)
    raise AttributeError(name)
