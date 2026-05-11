"""Model profile persistence helpers."""

from app.model_profiles import merge_profiles, profiles_to_settings


def test_model_profiles_merge_custom_values() -> None:
    profiles = merge_profiles(
        {"classifier": {"model": "vision-a", "workers": 4, "api_workers": 4, "temperature": 0.0}},
        api_base="http://x",
        model="fallback",
    )
    assert profiles["classifier"].model == "vision-a"
    assert profiles["classifier"].workers == 4
    assert profiles["classifier"].api_workers == 4
    saved = profiles_to_settings(profiles)
    assert saved["classifier"]["model"] == "vision-a"
    assert saved["classifier"]["api_workers"] == 4
