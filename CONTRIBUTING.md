# Contributing to ml-assert

First off, thank you for considering contributing to `ml-assert`! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, please [open an issue](https://github.com/HeyShinde/ml-assert/issues/new). It's generally best to discuss changes in an issue before you start working on a large pull request.

## Fork & create a branch

If you've decided to contribute code, the first step is to fork the repository and create a new branch from `main` for your changes.

```bash
# Fork the repo, then clone it
git clone https://github.com/YourUsername/ml-assert.git
cd ml-assert

# Create a new branch
git checkout -b my-new-feature
```

## Setting up for development

We use Poetry to manage dependencies and a virtual environment.

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Making your changes

Make your changes in the code. A few things to keep in mind:
- **Code Style**: We use `ruff` for linting and formatting. Please run it before committing.
- **Testing**: We use `pytest`. Please add tests for any new features or bug fixes. Ensure all tests pass before submitting a pull request.

```bash
# Run the linter
poetry run ruff check .

# Run the formatter
poetry run ruff format .

# Run tests
poetry run pytest
```

## Submitting a Pull Request

Once your changes are ready, commit them with a descriptive message and push the branch to your fork. Then, open a pull request from your fork's branch to the `main` branch of the `ml-assert` repository.

In your pull request description, please explain the "why" behind your changes and reference any related issues.

## Release Process

For maintainers, here's the process for releasing a new version:

1. **Update Version**:
   - Update version in `pyproject.toml`
   - Update version in `ml_assert/__init__.py`
   - Update `CHANGELOG.md` with new version and changes

2. **Documentation**:
   - Ensure all new features are documented
   - Update examples if needed
   - Check for any broken links or formatting issues

3. **Testing**:
   - Run full test suite: `poetry run pytest`
   - Run linting: `poetry run ruff check .`
   - Run formatting: `poetry run ruff format .`

4. **Build and Publish**:
   ```bash
   # Build the package
   poetry build

   # Publish to PyPI
   poetry publish
   ```

5. **GitHub Release**:
   - Create a new release on GitHub
   - Use the changelog entry as the release notes
   - Tag the release with the version number

6. **Post-Release**:
   - Update development version in `pyproject.toml`
   - Create a new branch for the next version

Thank you for your contribution!
