"""Quality gates for f1-points-finish@candidate. Every gate must pass before promotion.

The fixtures (criteria, races, holdout, model, input_columns, predictions,
probabilities) come from conftest.py. See exercises/quality-gates/README.md for the steps.

A gate that is still a TODO is skipped. A skipped gate checks nothing.

Usage:
    pytest exercises/quality-gates/test_gates.py
"""

from sklearn.metrics import roc_auc_score

from validate_data import validate

TARGET = "points_finish"


def test_data_passes_validation(races, record_property):
    """The data passes the lesson 02 schema."""
    failures = validate(races)
    record_property("failure_cases", len(failures))
    assert failures.empty, f"{len(failures)} failure cases in the data"


def test_model_predicts_0_or_1(predictions, holdout, record_property):
    """The model loads from the registry and returns 0 or 1 for every 2025 entry."""
    record_property("rows", len(predictions))
    assert len(predictions) == len(holdout)
    assert set(predictions.unique()) <= {0, 1}, f"unexpected predictions {sorted(predictions.unique())}"


def test_accuracy(predictions, holdout, criteria, record_property):
    """Accuracy on 2025 is at least min_accuracy."""
    accuracy = (predictions == holdout[TARGET]).mean()
    minimum = criteria["metrics"]["min_accuracy"]
    record_property("accuracy", f"{accuracy:.3f}")
    assert accuracy >= minimum, f"accuracy {accuracy:.3f} is below {minimum}"


def test_roc_auc(probabilities, holdout, criteria, record_property):
    """ROC AUC on 2025 is at least min_roc_auc."""
    auc = roc_auc_score(holdout[TARGET], probabilities)
    minimum = criteria["metrics"]["min_roc_auc"]
    record_property("roc_auc", f"{auc:.3f}")
    assert auc >= minimum, f"roc_auc {auc:.3f} is below {minimum}"


def test_beats_grid_rule(predictions, holdout, criteria, record_property):
    """Accuracy beats the grid rule by at least min_margin_over_grid_rule."""
    accuracy = (predictions == holdout[TARGET]).mean()
    baseline = ((holdout["grid"] <= 10).astype(int) == holdout[TARGET]).mean()
    margin = criteria["metrics"]["min_margin_over_grid_rule"]
    record_property("margin", f"{accuracy - baseline:.3f}")
    assert accuracy >= baseline + margin, (
        f"accuracy {accuracy:.3f} beats the grid rule ({baseline:.3f}) by less than {margin}"
    )


def test_no_forbidden_features(input_columns, criteria, record_property):
    """The model needs no input that is known only after the race."""
    forbidden = sorted(set(input_columns) & set(criteria["forbidden_features"]))
    record_property("inputs", len(input_columns))
    assert not forbidden, f"the model uses after-race columns: {', '.join(forbidden)}"


def test_rookie_accuracy(predictions, holdout, criteria, record_property):
    """Accuracy on 2025 rookie entries is at least min_accuracy_rookie."""
    rookies = holdout["rookie"]
    accuracy = (predictions[rookies] == holdout.loc[rookies, TARGET]).mean()
    minimum = criteria["metrics"]["min_accuracy_rookie"]
    record_property("rookie_accuracy", f"{accuracy:.3f}")
    assert accuracy >= minimum, f"rookie accuracy {accuracy:.3f} is below {minimum}"
