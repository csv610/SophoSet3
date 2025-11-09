# SophoSet

A unified framework for managing and evaluating datasets from Hugging Face Hub for vision-language models (VLMs) and large language models (LLMs). SophoSet provides a consistent interface to work with 100+ standardized datasets including text MCQ/OEQ datasets and vision datasets.

## Features

- **Unified Dataset Interface**: Consistent API across 100+ datasets from Hugging Face Hub
- **Multiple Data Types**: Support for text (MCQ, OEQ) and vision (MCQ, OEQ) datasets
- **Flexible Data Export**: Export datasets to JSON or LMDB format for efficient data storage
- **Dataset Exploration**: Stream through datasets with automatic subset/split detection
- **Robust Error Handling**: Comprehensive error handling and logging for production use
- **Type-Safe**: Full type hints for better IDE support and code quality
- **Easy to Extend**: Abstract base class pattern for adding new datasets

## Installation

### Basic Installation

```bash
pip install sophoset
```

### Installation with UI (Streamlit/Gradio)

```bash
pip install sophoset[ui]
```

### Development Installation

```bash
git clone https://github.com/yourusername/sophoset.git
cd sophoset
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.8
- PyTorch or TensorFlow (depending on the datasets you use)
- See `requirements.txt` for all dependencies

## Quick Start

### Loading a Dataset

```python
from sophoset.text.mcq.mmlu_data import MMLUDataset

# Initialize the dataset handler
dataset = MMLUDataset()

# Load a specific split and subset
dataset.load_dataset(split_name='test', subset_name='anatomy')

# Get total number of samples
print(f"Total samples: {dataset.get_row_count()}")

# Get a single sample
sample = dataset.get_row_data(0)
print(f"Question: {sample.question}")
print(f"Options: {sample.options}")
print(f"Answer: {sample.answer}")
```

### Sampling from Datasets

```python
# Get first 10 samples
samples = dataset.get_samples(max_samples=10, random_sample=False)

# Get 5 random samples
random_samples = dataset.get_samples(max_samples=5, random_sample=True, seed=42)

# Get all samples
all_samples = dataset.get_samples()
```

### Exploring Available Data

```python
from sophoset.utils.dataset_explorer import DatasetExplorer

explorer = DatasetExplorer(dataset)

# Iterate through all questions in all subsets/splits
for qa_data in explorer.next_question():
    print(f"Question: {qa_data.question}")
    print(f"Options: {qa_data.options}")
    print(f"Answer: {qa_data.answer}")
```

### Exporting Datasets

```python
from sophoset.utils.dataset_exporter import DatasetExporter

exporter = DatasetExporter(dataset)

# Export to JSON
exporter.export_json(output_dir='./data/mmlu_json')

# Export to LMDB (more efficient for large datasets)
exporter.export_lmdb(output_dir='./data/mmlu_lmdb')
```

### Working with LMDB Storage

```python
from sophoset.utils.lmdb_storage import LMDBStorage

# Open LMDB database
storage = LMDBStorage('./data/mmlu_lmdb')

# Retrieve a sample
sample = storage.get('anatomy/test/0')
print(sample)

# Check if key exists
exists = storage.has_key('anatomy/test/0')

# Get all keys
all_keys = storage.get_keys()

# Close connection
storage.close()
```

## Supported Datasets

### Text MCQ Datasets (13+)
- MMLU (Massive Multitask Language Understanding)
- MMLU Pro
- AI2 ARC
- SciQ
- AIME
- BigBench Hard
- Winogrande
- MedMCQA
- And more...

### Text OEQ Datasets (27+)
- GSM8K (Grade School Math)
- MathPlus
- GPQA
- TruthfulQA
- Pubmed QA
- Medical datasets
- And more...

### Vision MCQ Datasets (10+)
- AI2D
- MathVista
- ScienceQA
- Blink
- And more...

### Vision OEQ Datasets (30+)
- ChartQA
- DocVQA
- TextVQA
- VQA-RAD
- And more...

## Architecture

### Core Components

- **BaseHFDataset**: Abstract base class for all dataset implementations
- **QAData**: Dataclass for standardized question-answer representation
- **DatasetExplorer**: Generator-based dataset exploration
- **DatasetExporter**: Export datasets to multiple formats
- **LMDBStorage**: Efficient key-value storage for datasets

### Directory Structure

```
sophoset/
├── core/
│   └── base_hf_dataset.py      # Base classes and abstractions
├── text/
│   ├── mcq/                     # Text multiple-choice datasets
│   └── oeq/                     # Text open-ended datasets
├── vision/
│   ├── mcq/                     # Vision multiple-choice datasets
│   └── oeq/                     # Vision open-ended datasets
└── utils/
    ├── lmdb_storage.py         # LMDB database abstraction
    ├── dataset_exporter.py     # Export functionality
    ├── dataset_explorer.py     # Dataset exploration
    └── ...
```

## Advanced Usage

### Custom Dataset Implementation

Create your own dataset handler by extending `BaseHFDataset`:

```python
from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from typing import Dict, Any

class MyCustomDataset(BaseHFDataset):
    """Handler for my custom dataset."""

    DATASET_NAME = "my_org/my_dataset"

    def __init__(self):
        super().__init__(self.DATASET_NAME)

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a dataset row."""
        return QAData(
            key=f"{self.subset}/{self.split}/{index}",
            question=row['my_question_field'],
            options=BaseHFDataset.get_formatted_options(row['my_choices_field']),
            answer=row['my_answer_field'],
            context=row.get('my_context_field', ''),
        )

# Use your custom dataset
dataset = MyCustomDataset()
dataset.load_dataset(split_name='train')
samples = dataset.get_samples(max_samples=10)
```

### Logging Configuration

Control logging output:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now all SophoSet operations will log to your configured handler
```

### Batch Processing

```python
from sophoset.text.mcq.mmlu_data import MMLUDataset

dataset = MMLUDataset()

# Get all available subsets
subsets = dataset.get_subsets()

# Process each subset
for subset in subsets:
    try:
        dataset.load_dataset(split_name='test', subset_name=subset)
        samples = dataset.get_samples()
        print(f"Processed {len(samples)} samples from {subset}")
    except Exception as e:
        print(f"Error processing {subset}: {e}")
```

## Error Handling

SophoSet provides comprehensive error handling:

```python
from sophoset.text.mcq.mmlu_data import MMLUDataset

dataset = MMLUDataset()

try:
    # This will raise ValueError if subset doesn't exist
    dataset.load_dataset(split_name='test', subset_name='invalid_subset')
except ValueError as e:
    print(f"Invalid dataset configuration: {e}")
    # Get available subsets
    print(f"Available subsets: {dataset.get_subsets()}")

try:
    # This will raise IndexError if index is out of range
    sample = dataset.get_row_data(99999)
except IndexError as e:
    print(f"Invalid index: {e}")
    print(f"Valid range: 0-{dataset.get_row_count()-1}")
```

## Performance Tips

1. **Use LMDB for Large Datasets**: LMDB is more efficient than JSON for large-scale data
2. **Stream Processing**: Use `DatasetExplorer` for streaming through large datasets
3. **Random Sampling**: Use `random_sample=True` to sample without loading everything
4. **Batch Processing**: Process multiple datasets in parallel using multiprocessing

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and formatting
- Testing requirements
- Pull request process
- Adding new datasets

## Citation

If you use SophoSet in your research, please cite:

```bibtex
@software{sophoset2024,
    title={SophoSet: Unified Dataset Framework for VLM and LLM Evaluation},
    author={SophoSet Contributors},
    year={2024},
    url={https://github.com/yourusername/sophoset}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/sophoset/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/sophoset/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sophoset/discussions)

## Acknowledgments

This project integrates datasets from:
- Hugging Face Hub
- Research institutions
- Academic datasets
- Community contributions

## Roadmap

- [ ] Add support for more dataset formats
- [ ] Implement distributed dataset processing
- [ ] Create web UI for dataset exploration
- [ ] Add dataset version management
- [ ] Support for streaming large datasets
- [ ] Integration with popular ML frameworks

---

**Happy exploring! 🚀**
