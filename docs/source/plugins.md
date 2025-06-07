# Plugins

The `ml-assert` library can be extended with plugins to add custom checks. Plugins are discovered automatically if they are installed in the same environment and registered via entry points.

## `FileExistsPlugin`

Asserts that a file exists at a given path.

-   **Type Name**: `file_exists`
-   **Configuration**:
    -   `path` (`str`): The path to the file to check.
-   **Example `config.yaml`**:
    ```yaml
    steps:
      - type: file_exists
        path: path/to/my/file.csv
    ```

## `DVCArtifactCheckPlugin`

Asserts that a DVC-tracked artifact is in sync with its `.dvc` file, meaning it has not been modified since `dvc add`.

-   **Type Name**: `dvc_check`
-   **Installation**: Requires the `dvc` extra. Install with `pip install ml-assert[dvc]`.
-   **Configuration**:
    -   `path` (`str`): The path to the DVC-tracked artifact.
-   **Example `config.yaml`**:
    ```yaml
    steps:
      - type: dvc_check
        path: data/raw/training_data.csv
    ```

## Creating Your Own Plugin

To create a custom plugin:

1.  Create a class that inherits from `ml_assert.plugins.base.Plugin`.
2.  Implement the `run(self, config: dict)` method. This method should raise an `AssertionError` on failure. The `config` dictionary contains the step's configuration from the YAML file.
3.  Register your plugin as an entry point in your package's `pyproject.toml` under the `ml_assert.plugins` group.

    ```toml
    [project.entry-points."ml_assert.plugins"]
    my_plugin_name = "my_package.plugins:MyCustomPlugin"
    ```
