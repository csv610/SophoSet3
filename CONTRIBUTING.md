# Contributing to SophoSet

Thank you for considering contributing to SophoSet! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

### 1. Fork and Clone the Repository

```bash
git clone https://github.com/yourusername/sophoset.git
cd sophoset
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
# Create a feature branch for your work
git checkout -b feature/my-feature-name
```

## Development Guidelines

### Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following tools:

- **Black**: Code formatting (line length: 100 characters)
- **isort**: Import sorting
- **Flake8**: Linting
- **mypy**: Type checking

#### Format Your Code

```bash
# Format with Black
black sophoset/

# Sort imports
isort sophoset/

# Check for linting issues
flake8 sophoset/

# Type check
mypy sophoset/
```

### Type Hints

All functions must include type hints:

```python
def my_function(param1: str, param2: int) -> Dict[str, Any]:
    """Function with type hints.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    pass
```

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstring format
- Include examples for complex functions
- Update README.md if adding new features

```python
def get_samples(self, max_samples: Optional[int] = None) -> List[QAData]:
    """Get samples from the loaded dataset.

    Args:
        max_samples: Maximum number of samples to return. If None, returns all.

    Returns:
        List[QAData]: A list of samples from the dataset.

    Raises:
        RuntimeError: If no dataset is loaded.
        ValueError: If max_samples is not a positive integer.

    Example:
        >>> dataset = MMLUDataset()
        >>> dataset.load_dataset('test', 'anatomy')
        >>> samples = dataset.get_samples(max_samples=10)
        >>> print(len(samples))
        10
    """
```

### Logging

Use the logging module instead of print statements:

```python
import logging

logger = logging.getLogger(__name__)

# In your code
logger.info("Processing dataset...")
logger.warning("Could not load dataset")
logger.error("Failed to process sample", exc_info=True)
```

### Error Handling

Raise specific exception types with clear messages:

```python
# Good ✓
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")

# Avoid ✗
if not os.path.exists(path):
    raise Exception("File not found")
```

## Adding a New Dataset

### 1. Create Dataset Class

Create a new file in the appropriate directory (`sophoset/text/mcq/`, `sophoset/text/oeq/`, etc.):

```python
from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from typing import Dict, Any

class MyDataset(BaseHFDataset):
    """Handler for MyDataset from Hugging Face Hub."""

    DATASET_NAME = "org/dataset-name"  # HF Hub name

    def __init__(self):
        """Initialize the dataset handler."""
        super().__init__(self.DATASET_NAME)

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        return QAData(
            key=self.get_key(index),
            question=row['question_field'],
            options=self.get_formatted_options(row['options_field']),
            answer=row['answer_field'],
            context=row.get('context_field', ''),
            explanation=row.get('explanation_field', ''),
        )
```

### 2. Test Your Dataset

```python
# Test script
from sophoset.text.mcq.my_dataset import MyDataset

dataset = MyDataset()
dataset.load_dataset(split_name='train')
sample = dataset.get_row_data(0)
print(f"Question: {sample.question}")
print(f"Options: {sample.options}")
print(f"Answer: {sample.answer}")
```

### 3. Update Documentation

Add your dataset to the appropriate section in README.md.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=sophoset tests/

# Run specific test file
pytest tests/test_mmlu.py

# Run with verbose output
pytest -v tests/
```

## Pull Request Process

1. **Ensure all tests pass**:
   ```bash
   pytest tests/
   ```

2. **Format your code**:
   ```bash
   black sophoset/
   isort sophoset/
   ```

3. **Run linting and type checks**:
   ```bash
   flake8 sophoset/
   mypy sophoset/
   ```

4. **Write a clear PR description** that includes:
   - What changes you made and why
   - Any breaking changes
   - Relevant issues (use `Fixes #123`)
   - Test coverage for new features

5. **Example PR Title and Description**:
   ```
   Title: Add support for MMLU Pro dataset

   ## Description
   Adds support for MMLU Pro, an expanded version of MMLU with more subsets.

   ## Changes
   - Created `sophoset/text/mcq/mmlu_pro_data.py`
   - Added 57 new subsets
   - Updated dataset list in README

   ## Testing
   - All tests pass
   - Manually tested with 5 subsets
   - Coverage: 100%

   Fixes #456
   ```

6. **Review Process**:
   - One approval required before merging
   - All CI checks must pass
   - Address review comments promptly

## Reporting Bugs

When reporting bugs, include:

1. **Environment**:
   ```
   Python version: 3.10
   SophoSet version: 0.1.0
   OS: Ubuntu 22.04
   ```

2. **Reproduction Steps**:
   ```python
   from sophoset.text.mcq.mmlu_data import MMLUDataset
   dataset = MMLUDataset()
   dataset.load_dataset(split_name='test')
   ```

3. **Expected vs Actual Behavior**

4. **Error Traceback** (if applicable)

5. **Minimal Reproducible Example**

## Requesting Features

When requesting features:

1. Describe the problem you're trying to solve
2. Explain your proposed solution
3. Provide examples of how it would be used
4. Discuss alternatives you've considered

## Documentation

### Updating Docstrings

All public APIs must have docstrings:

```python
class MyClass:
    """Brief description of the class.

    Longer description explaining the purpose and usage of the class.

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
    """

    def my_method(self) -> str:
        """Brief description of the method.

        Returns:
            Description of the return value

        Raises:
            ValueError: When something is invalid
        """
```

## Performance Considerations

- Avoid loading entire datasets into memory when possible
- Use generators for streaming data
- Cache computed values when appropriate
- Profile code for bottlenecks

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for backward-compatible new features
- PATCH version for bug fixes

Version is specified in:
- `setup.py`
- `pyproject.toml`
- Release tags on GitHub

## Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/yourusername/sophoset/discussions)
- **Bugs**: File an [Issue](https://github.com/yourusername/sophoset/issues)
- **Documentation**: Check the [README](README.md) and examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

We appreciate all contributions, no matter how small. Thank you for helping make SophoSet better!

---

**Happy contributing! 🎉**
