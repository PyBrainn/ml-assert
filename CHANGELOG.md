# Changelog

All notable changes to this project will be documented in this file.

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
