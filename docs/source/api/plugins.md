# Plugins API Reference

## Creating Custom Plugins

### Plugin Base Class

All plugins must inherit from the base plugin class:

```python
from ml_assert.plugins.base import Plugin

class MyCustomPlugin(Plugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(self):
        # Implement validation logic
        pass
```

### Required Methods

#### `validate()`
Must be implemented by all plugins. Should raise `AssertionError` if validation fails.

### Optional Methods

#### `setup()`
Called before validation. Use for initialization.

#### `teardown()`
Called after validation. Use for cleanup.

## Built-in Plugins

### FileExistsPlugin

Checks if files exist.

```python
from ml_assert.plugins.file_exists import FileExistsPlugin

# Check if file exists
plugin = FileExistsPlugin(path="models/model.pkl")
plugin.validate()
```

### Parameters

- `path`: Path to file
- `exists`: Whether file should exist (default: True)

### DVCArtifactCheckPlugin

Checks DVC artifacts.

```python
from ml_assert.plugins.dvc_check import DVCArtifactCheckPlugin

# Check DVC artifact
plugin = DVCArtifactCheckPlugin(
    path="models/model.pkl",
    exists=True,
    size_mb=10
)
plugin.validate()
```

### Parameters

- `path`: Path to DVC artifact
- `exists`: Whether artifact should exist
- `size_mb`: Expected size in megabytes (optional)
- `md5`: Expected MD5 hash (optional)

## Plugin Registration

### Entry Point

Register your plugin in `pyproject.toml`:

```toml
[project.entry-points."ml_assert.plugins"]
my_plugin = "my_package.plugins:MyCustomPlugin"
```

### CLI Usage

Use registered plugins in your YAML configuration:

```yaml
steps:
  - type: my_plugin
    path: "models/model.pkl"
    exists: true
```
