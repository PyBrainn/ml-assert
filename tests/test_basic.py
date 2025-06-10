import ml_assert


def test_import_ml_assert():
    # Ensure the package imports and version matches
    assert hasattr(ml_assert, "__version__")
    assert ml_assert.__version__ == "1.0.4"
