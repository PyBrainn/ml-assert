# Plugins

The `ml-assert` library can be extended with plugins to add custom checks. Plugins are discovered automatically if they are installed in the same environment and registered via entry points.

## API Reference: Plugin

### `class Plugin`
Abstract base class for all plugins.

**Method:**
- `run(self, config: dict) -> AssertionResult`: Execute the plugin's logic. Must return an `AssertionResult`.

**Example:**
```python
from ml_assert.plugins.base import Plugin
from ml_assert.core.base import AssertionResult
from datetime import datetime

class MyPlugin(Plugin):
    def run(self, config: dict) -> AssertionResult:
        # ... your logic ...
        if some_condition:
            return AssertionResult(
                success=True,
                message="Check passed!",
                timestamp=datetime.now(),
                metadata={"info": "details"}
            )
        else:
            return AssertionResult(
                success=False,
                message="Check failed!",
                timestamp=datetime.now(),
                metadata={"error": "details"}
            )
```

## Error Handling & Result Reporting

- Plugins must return an `AssertionResult` from `run()`.
- The CLI and integrations use the `success`, `message`, and `metadata` fields for reporting and alerting.
- If a plugin raises an exception, it is caught by the CLI and reported as a failed step.
- Use the `metadata` field to provide detailed results or debugging info.

## Plugin Workflow Diagram

```mermaid
graph TD
    A[CLI loads plugins] --> B[Plugin.run(config)]
    B --> C{Check passes?}
    C -- Yes --> D[Return AssertionResult(success=True)]
    C -- No --> E[Return AssertionResult(success=False, metadata with errors)]
    D & E --> F[CLI/integrations handle result]
```

## Advanced Usage: Custom Plugin Example

Suppose you want to check that a DataFrame has at least N rows:

```python
from ml_assert.plugins.base import Plugin
from ml_assert.core.base import AssertionResult
from datetime import datetime
import pandas as pd

class MinRowsPlugin(Plugin):
    def run(self, config: dict) -> AssertionResult:
        df = pd.read_csv(config["file"])
        min_rows = config.get("min_rows", 10)
        if len(df) >= min_rows:
            return AssertionResult(
                success=True,
                message=f"DataFrame has {len(df)} rows (min required: {min_rows})",
                timestamp=datetime.now(),
                metadata={"row_count": len(df)}
            )
        else:
            return AssertionResult(
                success=False,
                message=f"DataFrame has only {len(df)} rows (min required: {min_rows})",
                timestamp=datetime.now(),
                metadata={"row_count": len(df)}
            )
```

Register your plugin in your `pyproject.toml`:
```toml
[project.entry-points."ml_assert.plugins"]
min_rows = "my_package.plugins:MinRowsPlugin"
```

Use in YAML config:
```yaml
steps:
  - type: min_rows
    file: data/my_data.csv
    min_rows: 100
```

## Built-in Plugins

### `FileExistsPlugin`
Asserts that a file exists at a given path.

- **Type Name**: `file_exists`
- **Configuration**:
    - `path` (`str`): The path to the file to check.
-   **Example `config.yaml`**:
    ```yaml
    steps:
      - type: file_exists
        path: path/to/my/file.csv
    ```

### `DVCArtifactCheckPlugin`
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

1.  Inherit from `ml_assert.plugins.base.Plugin`.
2.  Implement `run(self, config: dict) -> AssertionResult`.
3.  Register your plugin as an entry point in your package's `pyproject.toml` under the `ml_assert.plugins` group.

    ```toml
    [project.entry-points."ml_assert.plugins"]
    my_plugin_name = "my_package.plugins:MyCustomPlugin"
    ```
