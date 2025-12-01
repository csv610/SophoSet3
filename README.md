# SophoSet

## Problem Statement

Hugging Face Hub contains over 100 publicly available datasets suitable for LLM evaluation. However, these datasets use inconsistent field naming conventions and data structures, requiring significant preprocessing before use in evaluation pipelines.

Example field name variations across datasets:

```python
# MMLU dataset
{"question": str, "choices": list, "answer": int}

# GSM8K dataset
{"question": str, "answer": str}  # No multiple choice options

# AI2 ARC dataset
{"question": str, "choices": {"text": list}, "answerKey": str}

# MathVista dataset
{"query": str, "options": list, "answer": str}

# ChartQA dataset
{"question": str, "image": Image, "answer": str}
```

This heterogeneity in schema design requires custom data transformation code for each dataset before they can be used in a unified evaluation framework.

---

## Solution: Standardized Data Format

SophoSet provides a unified data schema across all 100+ datasets. All datasets are normalized to a single QAData structure:

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
  "explanation": "The medial malleolus is a bony prominence on the medial surface of the tibia.",
  "context": "",
  "images": []
}
```

---

## Comparison: Traditional vs. Standardized Approach

### Traditional Approach (Dataset-Specific Processing)

```python
from datasets import load_dataset

# Load multiple datasets
mmlu = load_dataset("cais/mmlu")
gsm8k = load_dataset("openai/gsm8k")
ai2_arc = load_dataset("allenai/ai2_arc")

# Dataset-specific transformation code
def process_mmlu(item):
    return {
        "question": item["question"],
        "options": {chr(65+i): opt for i, opt in enumerate(item["choices"])},
        "answer": chr(65 + item["answer"])
    }

def process_gsm8k(item):
    return {
        "question": item["question"],
        "options": None,
        "answer": item["answer"]
    }

def process_ai2_arc(item):
    return {
        "question": item["question"],
        "options": {k: v for k, v in zip("ABCDEFG", item["choices"]["text"])},
        "answer": item["answerKey"]
    }

# Multiple evaluation pipelines
def evaluate_mmlu(model):
    # Custom logic for MMLU
    pass

def evaluate_gsm8k(model):
    # Custom logic for GSM8K (no options)
    pass

def evaluate_ai2_arc(model):
    # Custom logic for AI2 ARC
    pass
```

### Standardized Approach (SophoSet)

```python
from sophoset.text.mcq.mmlu_data import MMLUDataset
from sophoset.text.oeq.gsm8k_data import GSM8KDataset
from sophoset.text.mcq.ai2_arc_data import Ai2ArcDataset

# Unified evaluation function
def evaluate_model(model, dataset_instance):
    dataset_instance.load_dataset(split_name='test')
    correct = 0

    for sample in dataset_instance.get_samples():
        response = model.generate(
            question=sample.question,
            options=sample.options,
            context=sample.context,
            images=sample.images
        )
        if response == sample.answer:
            correct += 1

    return correct / dataset_instance.get_row_count()

# Single evaluation code for all datasets
for dataset_class in [MMLUDataset, GSM8KDataset, Ai2ArcDataset]:
    dataset = dataset_class()
    accuracy = evaluate_model(my_model, dataset)
    print(f"{dataset_class.__name__}: {accuracy:.4f}")
```

---

## Core Features

- **100+ Standardized Datasets**: Consistent schema across all supported datasets (MMLU, GSM8K, MathVista, ChartQA, etc.)
- **Unified QAData Format**: Single data structure for all evaluation datasets
- **Multi-modal Support**: Handles text (MCQ, OEQ) and vision datasets uniformly
- **Data Export Options**: JSON and LMDB serialization formats
- **Dataset Discovery**: Automatic subset and split detection
- **Error Handling**: Comprehensive exception handling and logging
- **Type Annotations**: Full type hints for static analysis
- **Extensible Design**: Clear pattern for adding new datasets

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

### Text MCQ Datasets (11)
- AI2 ARC
- AIME
- BigBench Hard
- MedMCQA
- Medical Concepts QA
- Medical Meadow MedicalQA
- MedQA USMLE 4 Options
- MMLU (Massive Multitask Language Understanding)
- MMLU Pro
- SciQ
- Winogrande

### Text OEQ Datasets (26)
- AIME 2025
- DeepScaleR
- GPQA (Google-Proof Question Answering)
- GSM+ (Enhanced Grade School Math)
- GSM8K (Grade School Math)
- IMO Geometry
- Math LightEval
- MathPlus
- MedQA
- Medical Meadow Flashcards
- Medical Meadow Wikidoc
- Medical Questions
- Medication QA
- MedQnA V3
- MedQuad
- MedQuad MedQnA
- MetaMathQA
- MetaMathQA 40K
- Open Medical Cases
- Olympiads
- Pubmed QA
- SafetyBench
- SciBench
- SimpleQA
- Toxic Prompts
- TruthfulQA

### Vision MCQ Datasets (9)
- AI2D (Artificial Intelligence 2 Diagrams)
- Blink
- Hidden Flaws GPT-4V
- MathV360K
- MathVerse
- MathVision
- MathVista
- ScienceQA
- World Medical QA

### Vision OEQ Datasets (26)
- Animals
- CAMO (Camouflaged Object Detection)
- ChartQA
- Chest X-Ray Pneumonia
- CulturalVQA
- DocVQA (Document Visual Question Answering)
- IAM Line (Handwriting Recognition)
- IAM Sentences (Handwriting Recognition)
- IllusionBench
- InfoVQA
- Kvasir VQA (Medical)
- Kvasir VQA X1 (Medical)
- MMSTAR
- OCR Bench V2
- Olympiad Bench
- Olympic Arena
- Open Medical Case Images
- Open Medical Images
- PD12M (Panoramic Dental)
- RealWorld QA
- ROCO (Radiology Common)
- SNLI-VE (Visual Entailment)
- TextVQA (Text Visual Question Answering)
- VisitBench
- VLMs Are Blind
- VQA-RAD (Visual Question Answering - Radiology)

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
