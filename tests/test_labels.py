import pytest

from transform.labels import OTHER, map_category, map_severity


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["agents-window", "bug"], "AI/Chat"),
        (["chat-input"], "AI/Chat"),
        (["perf", "terminal"], "Performance"),  # priority order: Performance beats Integration
        (["freeze-slow-crash-leak"], "Performance"),
        (["editor-core"], "UI"),
        (["workbench-tabs"], "UI"),
        (["terminal"], "Integration"),
        (["remote", "ssh"], "Integration"),
        (["Component: Developer Tools", "Type: Bug"], "Integration"),
        (["Component: DOM"], "UI"),
        (["Component: Reconciler"], "Core"),
        (["languages-guessing"], "Core"),
        (["UI"], "UI"),  # case-insensitive
    ],
)
def test_area_labels_map_to_category(labels, expected):
    assert map_category(labels) == expected


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["bug"],  # legacy app mapped this to Backend; it's a type, not an area
        ["info-needed", "*duplicate", "verified"],
        ["Status: Unconfirmed", "Resolution: Stale", "Type: Bug"],
        ["error-telemetry"],  # area label we don't have a rule for
        ["good first issue"],
    ],
)
def test_process_or_unknown_labels_are_other(labels):
    assert map_category(labels) == OTHER


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["critical"], "Critical"),
        (["freeze-slow-crash-leak"], "Critical"),
        (["data-loss"], "Critical"),
        (["important", "bug"], "Critical"),
        (["bug"], "High"),
        (["Type: Bug"], "High"),
        (["regression"], "High"),
        (["feature-request"], "Medium"),
        ([], "Medium"),
    ],
)
def test_severity(labels, expected):
    assert map_severity(labels) == expected
