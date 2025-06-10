# Core API Reference

## DataFrameAssertion

The main assertion class for DataFrame validation.

```python
from ml_assert import Assertion

# Create an assertion instance
assertion = Assertion(df)

# Chain assertions
assertion.satisfies(schema).no_nulls().validate()
```

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

## schema

A builder for creating DataFrame validation schemas.

```python
from ml_assert import schema

# Create a schema
s = schema()
s.col("user_id").is_unique()
s.col("age").in_range(18, 70)
```

### Methods

#### `col(column_name)`
Starts a column validation chain.

**Parameters:**
- `column_name`: Name of the column to validate

**Returns:**
- A column validator object

### Column Validator Methods

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
