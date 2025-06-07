# Changelog

All notable changes to this project will be documented in this file.

## [1.0.2] - 2025-06-08

### Changed
-   **Major Refactoring of Schema Validation**: Replaced the simple dictionary-based schema assertion with a powerful, chainable `schema` builder DSL. This provides a more expressive and flexible API for defining complex validation rules.
    -   `Assertion.schema()` is now `Assertion.satisfies(schema(...))`.
    -   The new `schema()` builder supports column-level checks like `is_type`, `is_unique`, and `in_range`.
-   **CLI Update**: The `ml_assert run` and `ml_assert schema` commands were updated to support the new schema definition format in YAML configuration files.

### Fixed
-   Resolved numerous `ImportError` and `AttributeError` issues caused by previous refactoring.
-   Fixed issues with package discovery and installation from TestPyPI.
-   Corrected the public API exposed in the top-level `__init__.py` to ensure all user-facing classes and functions are accessible.

## [1.0.0] - 2025-06-08

### Added
- Initial official release of **ml-assert**.
- **Core Features**:
  - `DataFrameAssertion` DSL for data validation.
  - `assert_model` DSL for model performance validation.
  - `assert_no_drift` for high-level data drift detection.
  - Low-level statistical tests (`ks_test`, `chi2_test`, `wasserstein_distance`).
  - Plugin system with `file_exists` and `dvc_check`.
  - CLI runner with YAML configuration, producing JSON and HTML reports.
