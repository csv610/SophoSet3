# Dataset Migration Guide

This guide helps maintainers update existing dataset implementations to follow SophoSet best practices.

## Recent Improvements

We've made the following improvements to ensure code quality and consistency:

1. **Option Formatting Consolidation**: Use `BaseHFDataset.get_formatted_options()` instead of duplicating formatting logic
2. **Type Hints**: Added comprehensive type hints to improve IDE support and catch errors
3. **Logging**: Replaced `print()` statements with proper logging
4. **Docstrings**: Enhanced documentation with usage examples

## Migration Checklist

### 1. Import Organization

**Before:**
```python
from typing import Dict, Any, List, Optional
from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter  # Usually unused
from sophoset.utils.dataset_explorer import DatasetExplorer  # Usually unused
```

**After:**
```python
"""Brief description of the dataset module."""

from typing import Any, Dict

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
```

Remove unused imports and add a module docstring.

### 2. Class Documentation

**Before:**
```python
class MMLUDataset(BaseHFDataset):
    DATASET_NAME = "cais/mmlu"

    def __init__(self):
        super().__init__(self.DATASET_NAME)
```

**After:**
```python
class MMLUDataset(BaseHFDataset):
    """Handler for MMLU dataset from Hugging Face Hub."""

    DATASET_NAME = "cais/mmlu"

    def __init__(self):
        """Initialize the MMLU dataset handler."""
        super().__init__(self.DATASET_NAME)
```

Add docstrings to class and `__init__` method.

### 3. Option Formatting (Most Important!)

**Before (Duplicated in 50+ files):**
```python
def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
    options_list = row.get('choices', [])

    # Format options as dict with letter keys (A, B, C, D, E, etc.)
    formatted_options = {}
    if options_list:
        letters = [chr(65 + i) for i in range(26)]
        for i, opt in enumerate(options_list):
            if i < len(letters):
                formatted_options[letters[i]] = opt
```

**After (Using base class method):**
```python
def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
    options_list = row.get('choices', [])

    # Format options using base class method
    formatted_options = self.get_formatted_options(options_list)
```

### 4. Error Handling

**Before:**
```python
try:
    answer = chr(ord('A') + int(answer))
except ValueError:
    answer = "NA"
```

**After:**
```python
try:
    answer = chr(ord('A') + int(answer))
except (ValueError, TypeError):
    answer = "NA"
```

Catch multiple exception types explicitly.

### 5. Type Hints

**Before:**
```python
def extract_row_data(self, row, index):
    # Missing type hints
```

**After:**
```python
def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
    """Extract and format data from a dataset row.

    Args:
        row: The dataset row to extract data from
        index: The index of the row in the dataset

    Returns:
        QAData object containing the formatted row data
    """
```

### 6. Code Formatting

**Before:**
```python
question = row.get('question', '')
options_list = row.get('choices', [])
answer  = row.get('answer', '')  # Extra space before =
context : str = ""  # Colon instead of annotation
```

**After:**
```python
question = row.get('question', '')
options_list = row.get('choices', [])
answer = row.get('answer', '')
context = row.get('context', '')
```

Fix spacing and formatting inconsistencies.

## Example: Complete Migration

### Original MMLU Dataset File
```python
from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class MMLUDataset(BaseHFDataset):
    DATASET_NAME = "cais/mmlu"

    def __init__(self):
        super().__init__(self.DATASET_NAME)

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        question = row.get('question', '')
        options_list = row.get('choices', [])
        answer  = row.get('answer', '')

        try:
           answer = chr(ord('A') + int(answer))
        except ValueError:
           answer =  "NA"

        formatted_options = {}
        if options_list:
            letters = [chr(65 + i) for i in range(26)]
            for i, opt in enumerate(options_list):
                if i < len(letters):
                    formatted_options[letters[i]] = opt

        return QAData(
            key=self.get_key(index),
            question=question,
            options=formatted_options,
            answer=answer
        )
```

### Updated MMLU Dataset File
```python
"""MMLU Dataset handler for Hugging Face Hub.

This module provides a dataset handler for the MMLU (Massive Multitask Language Understanding)
dataset from Hugging Face Hub.
"""

from typing import Any, Dict

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData


class MMLUDataset(BaseHFDataset):
    """Handler for MMLU dataset from Hugging Face Hub."""

    DATASET_NAME = "cais/mmlu"

    def __init__(self):
        """Initialize the MMLU dataset handler."""
        super().__init__(self.DATASET_NAME)

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from an MMLU dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', '')
        options_list = row.get('choices', [])
        answer = row.get('answer', '')

        # Convert numeric answer to letter
        try:
            answer = chr(ord('A') + int(answer))
        except (ValueError, TypeError):
            answer = "NA"

        # Format options using base class method
        formatted_options = self.get_formatted_options(options_list)

        return QAData(
            key=self.get_key(index),
            question=question,
            options=formatted_options,
            answer=answer,
        )
```

## Automated Migration Script

To help with bulk migrations, you can use this Python script:

```python
import re
from pathlib import Path

def migrate_dataset_file(file_path: Path) -> str:
    """Migrate a dataset file to follow best practices."""
    with open(file_path) as f:
        content = f.read()

    # 1. Remove unused imports
    content = re.sub(
        r'from sophoset\.utils\.dataset_exporter import DatasetExporter\n',
        '',
        content
    )
    content = re.sub(
        r'from sophoset\.utils\.dataset_explorer import DatasetExplorer\n',
        '',
        content
    )

    # 2. Fix option formatting
    pattern = r'''letters = \[chr\(65 \+ i\) for i in range\(26\)\]
            for i, opt in enumerate\(options.*?\):
                if i < len\(letters\):
                    formatted_options\[letters\[i\]\] = opt'''
    replacement = r'''formatted_options = self.get_formatted_options(options_list)'''
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # 3. Fix spacing issues
    content = content.replace('answer  = ', 'answer = ')
    content = content.replace('context : str = ', 'context = ')

    return content

# Usage
dataset_file = Path('sophoset/text/mcq/mmlu_data.py')
migrated = migrate_dataset_file(dataset_file)
```

## Testing After Migration

After migrating a dataset file, test it:

```python
from sophoset.text.mcq.your_dataset import YourDataset

# Test basic functionality
dataset = YourDataset()
dataset.load_dataset(split_name='train')

# Test sampling
samples = dataset.get_samples(max_samples=5)
assert len(samples) == 5

# Test data structure
for sample in samples:
    assert sample.question
    assert sample.options
    assert sample.answer

print("✓ Migration successful!")
```

## Common Issues

### Issue: "Maximum 26 options" warning

If you see this warning, it means a dataset has more than 26 options. You need to:

1. Check if the option formatting is correct for that dataset
2. Update the base class or the dataset handler to support more options

### Issue: Type hint errors

If you're adding type hints and mypy complains:

```bash
mypy sophoset/text/mcq/your_dataset.py
```

Make sure all parameter types are correctly specified.

### Issue: Import cycles

If you get import cycle errors, make sure you're not importing from submodules that import the base class.

## Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Type Hints Documentation](https://docs.python.org/3/library/typing.html)

## Contributing

When submitting PRs:

1. Migrate at least 5 dataset files as examples
2. Update this guide if you discover new patterns
3. Ensure all tests pass
4. Run formatting tools:
   ```bash
   black sophoset/
   isort sophoset/
   flake8 sophoset/
   mypy sophoset/
   ```

---

Thank you for helping improve SophoSet! 🎉
