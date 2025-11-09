# SophoSet

## The Problem: Inconsistent Hugging Face Datasets

Hugging Face Hub has 100+ amazing datasets for LLM evaluation, but they're **completely inconsistent**:

```python
# MMLU: "question", "choices", numeric answer
mmlu = load_dataset("cais/mmlu")
# {"question": "...", "choices": [...], "answer": 0}

# GSM8K: "question", "answer" (no options!)
gsm8k = load_dataset("openai/gsm8k")
# {"question": "...", "answer": "..."}

# AI2 ARC: "question", "choices.text", "answerKey"
ai2_arc = load_dataset("allenai/ai2_arc")
# {"question": "...", "choices": {"text": [...]}, "answerKey": "A"}

# MathVista: "query", "options", "answer"
mathvista = load_dataset("...")
# {"query": "...", "options": [...], "answer": "..."}

# ChartQA: "question", "image", "answer"
chartqa = load_dataset("...")
# {"question": "...", "image": Image, "answer": "..."}
```

**Result**: You spend **DAYS/WEEKS cleaning and normalizing data** before you can evaluate a single model.

---

## The Solution: One Standardized Format

SophoSet transforms all 100+ datasets into a **single, consistent JSON structure**:

```json
{
  "key": "anatomy/test/0",
  "question": "The medial malleolus is part of which bone?",
  "options": {
    "A": "tibia",
    "B": "fibula",
    "C": "talus",
    "D": "calcaneus"
  },
  "answer": "A",
  "explanation": "The medial malleolus is a bony prominence on the medial surface of the tibia...",
  "context": "",
  "images": []
}
```

**Same structure. Every dataset. No cleaning required.**

---

## Before & After

### ❌ WITHOUT SophoSet (Traditional Approach)

```python
from datasets import load_dataset

# Load 100 datasets with inconsistent APIs
datasets = {}
for ds_name in ["cais/mmlu", "openai/gsm8k", "allenai/ai2_arc", ...]:
    datasets[ds_name] = load_dataset(ds_name)

# Write custom cleaning code for EACH dataset
cleaned_data = []

# MMLU cleaning
for item in datasets["cais/mmlu"]:
    cleaned_data.append({
        "question": item["question"],
        "options": {chr(65+i): opt for i, opt in enumerate(item["choices"])},
        "answer": chr(65 + item["answer"])
    })

# GSM8K cleaning (no options!)
for item in datasets["openai/gsm8k"]:
    cleaned_data.append({
        "question": item["question"],
        "options": {},  # No options for math
        "answer": item["answer"]
    })

# AI2 ARC cleaning (different field names)
for item in datasets["allenai/ai2_arc"]:
    cleaned_data.append({
        "question": item["question"],
        "options": {k: v for k, v in zip("ABCDEFG", item["choices"]["text"])},
        "answer": item["answerKey"]
    })

# MathVista cleaning (yet another format)
for item in datasets["..."]:
    # ... different cleaning logic ...
    pass

# Result: Weeks of cleaning, hundreds of lines of code
# Then YOU have to write custom evaluation code for each format
```

### ✅ WITH SophoSet (Standardized Approach)

```python
from sophoset.text.mcq.mmlu_data import MMLUDataset
from sophoset.text.oeq.gsm8k_data import GSM8KDataset
from sophoset.vision.mcq.mathvista_data import MathVistaDataset
from sophoset.vision.oeq.chartqa_data import ChartQADataset

# All datasets return the SAME standardized QAData structure
datasets = [
    MMLUDataset(),
    GSM8KDataset(),
    MathVistaDataset(),
    ChartQADataset()
]

# ONE evaluation code works for ALL 100+ datasets
def evaluate_model(model, dataset_instance):
    dataset_instance.load_dataset(split_name='test')
    accuracy = 0

    for sample in dataset_instance.get_samples():
        # Same fields. Same format. Every time.
        response = model.generate(
            question=sample.question,
            options=sample.options,      # Always {"A": "...", "B": "..."}
            context=sample.context,      # Always a string
            images=sample.images         # Always a list
        )
        if response == sample.answer:
            accuracy += 1

    return accuracy / dataset_instance.get_row_count()

# Use with ANY dataset - code never changes
for dataset in datasets:
    accuracy = evaluate_model(my_llm, dataset)
    print(f"Accuracy: {accuracy:.2%}")
```

**Result: Minutes of setup. No data cleaning. Pure evaluation.**

---

## Key Benefits

✅ **Zero Data Cleaning**: All 100+ datasets normalized to standardized format
✅ **One Evaluation Code**: Write once, test on all datasets
✅ **Consistent QAData**: Every dataset has the same JSON structure
✅ **Ready for LLMs**: Data formatted exactly as needed for model evaluation
✅ **Production Ready**: Handles vision data, missing options, and edge cases
✅ **Time Savings**: Days of cleaning → Minutes of setup

---

## Features

- **100+ Standardized Datasets**: MMLU, GSM8K, MathVista, ChartQA, and 96+ more
- **Unified QAData Format**: Consistent structure across all datasets (no cleaning!)
- **Text & Vision Support**: MCQ/OEQ for text, vision datasets with image handling
- **Flexible Data Export**: Export to JSON or LMDB for efficient storage
- **Dataset Exploration**: Stream through datasets with automatic detection
- **Robust Error Handling**: Comprehensive logging and validation
- **Type-Safe**: Full type hints for IDE support
- **Easy to Extend**: Simple pattern for adding new datasets

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
