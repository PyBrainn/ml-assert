# Changelog

All notable changes to this project will be documented in this file.

## [1.0.5] - 2025-06-12

### Added
- Added comprehensive cross-validation support with multiple strategies:
  - K-Fold Cross-Validation
  - Stratified K-Fold Cross-Validation
  - Leave-One-Out Cross-Validation
- Added cross-validation assertions for multiple metrics:
  - Accuracy Score
  - Precision Score
  - Recall Score
  - F1 Score
  - ROC AUC Score
- Added `get_cv_summary` function for detailed cross-validation metrics
- Added parallel processing support for faster cross-validation
- Added comprehensive documentation for cross-validation features
- Added cross-validation examples in documentation

### Changed
- Updated model evaluation to support cross-validation-based assertions
- Enhanced error handling for cross-validation operations
- Improved documentation structure to include cross-validation section

### Fixed
- Fixed potential memory issues in large-scale cross-validation
- Fixed documentation formatting for cross-validation examples

## [1.0.4] - 2025-06-11

### Added
- Added comprehensive documentation for all core modules and functions
- Added detailed API reference in the documentation
- Added more examples in the documentation for common use cases

### Changed
- Improved error messages for better debugging experience
- Enhanced documentation with more detailed examples and explanations
- Updated contributing guidelines with more detailed instructions

### Fixed
- Fixed documentation formatting issues
- Fixed typos and inconsistencies in documentation
- Fixed minor formatting issues in error messages

## [1.0.3] - 2025-06-08

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
