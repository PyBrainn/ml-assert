# Data Assertions

The `ml_assert` module provides a fluent, chainable API for validating the integrity and structure of your `pandas` DataFrames.

---

## Quick Start

```python
import pandas as pd
from ml_assert import Assertion, schema

# Create a DataFrame
df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "age": [25, 30, 35],
    "plan_type": ["basic", "premium", "basic"]
})

# Create a schema
s = schema()
s.col("user_id").is_unique()
s.col("age").in_range(18, 70)
s.col("plan_type").is_type("object")

# Validate
Assertion(df).satisfies(s).no_nulls().validate()
```

---

## Schema Builder

The `schema()` builder provides a fluent interface for defining DataFrame validation rules.

### Basic Usage

```python
from ml_assert import schema

# Create a schema
s = schema()
s.col("user_id").is_unique()
s.col("age").in_range(18, 70)
s.col("plan_type").in_set(["basic", "premium", "free"])
```

### Available Validators

#### `is_unique()`
Checks if column values are unique.

#### `in_range(min_val, max_val)`
Checks if column values are within a range.

**Parameters:**
- `min_val`: Minimum allowed value
- `max_val`: Maximum allowed value

#### `is_type(dtype)`
Checks if column has the specified data type.

**Parameters:**
- `dtype`: Expected data type (e.g., "int64", "float64", "object")

#### `in_set(allowed_values)`
Checks if column values are in a set of allowed values.

**Parameters:**
- `allowed_values`: Set or list of allowed values

#### `matches(pattern)`
Checks if column values match a regex pattern.

**Parameters:**
- `pattern`: Regular expression pattern to match

#### `is_not_null()`
Checks if column has no null values.

#### `is_sorted(ascending=True)`
Checks if column is sorted.

**Parameters:**
- `ascending`: Whether to check ascending order (default: True)

---

## DataFrameAssertion

The main class for DataFrame validation.

### Methods

#### `satisfies(schema)`
Validates the DataFrame against a schema definition.

**Parameters:**
- `schema`: A schema object created using the `schema()` builder

**Returns:**
- `self` for method chaining

#### `no_nulls(columns=None)`
Checks for null values in specified columns.

**Parameters:**
- `columns`: Optional list of column names to check. If None, checks all columns.

**Returns:**
- `self` for method chaining

#### `validate()`
Executes all chained assertions.

**Raises:**
- `AssertionError` if any assertion fails

---

## Error Handling & Result Reporting

- All assertion methods raise `AssertionError` if a check fails during chaining, unless `.validate()` is called.
- `.validate()` returns an `AssertionResult` object:
    - `success` (bool): True if all assertions passed.
    - `message` (str): Summary message.
    - `timestamp` (datetime): When the check was run.
    - `metadata` (dict): Details of each assertion (name, args, success, error if any).

---

## Examples

### Basic Schema Validation

```python
from ml_assert import Assertion, schema

# Create a schema
s = schema()
s.col("id").is_unique()
s.col("age").in_range(18, 70)
s.col("email").matches(r"^[^@]+@[^@]+\.[^@]+$")

# Validate
Assertion(df).satisfies(s).validate()
```

### Complex Schema with Multiple Rules

```python
from ml_assert import Assertion, schema

# Create a schema
s = schema()
s.col("user_id").is_unique().is_not_null()
s.col("age").in_range(18, 70).is_type("int64")
s.col("plan_type").in_set(["basic", "premium", "free"])
s.col("subscription_date").is_sorted(ascending=True)

# Validate
Assertion(df).satisfies(s).no_nulls(["user_id", "plan_type"]).validate()
```

For more detailed API reference, see [Data API](api/data.md).
