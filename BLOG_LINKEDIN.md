# LinkedIn Article: "SophoSet: Standardizing Inconsistent Dataset Schemas for LLM Evaluation"

---

## The Problem

When evaluating large language models across multiple benchmarks, researchers encounter a fundamental challenge: dataset schemas are inconsistent. The Hugging Face Hub contains over 100 publicly available evaluation datasets, but they use different field naming conventions and data structures.

**Examples:**

```
MMLU: {"question": str, "choices": list, "answer": int}
GSM8K: {"question": str, "answer": str}
AI2 ARC: {"question": str, "choices": {"text": list}, "answerKey": str}
MathVista: {"query": str, "options": list, "answer": str}
ChartQA: {"question": str, "image": Image, "answer": str}
```

This heterogeneity requires custom transformation code for each dataset before they can be used in unified evaluation frameworks. For researchers working with multiple datasets, this preprocessing step introduces significant overhead.

---

## SophoSet: A Unified Data Framework

SophoSet is an open-source framework that provides a consistent data schema across 100+ evaluation datasets. All datasets are normalized to a single structure:

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
  "answer": "A"
}
```

The framework covers text and vision domains, supporting both multiple-choice and open-ended questions:

- **Text MCQ:** AI2-ARC, AIME, BigBench Hard, MMLU-Pro, SciQ, WinoGrande, and others
- **Text OEQ:** AIME 2025, GPQA, GSM8K, IMO Geometry, MedQA, PubMedQA, and others
- **Vision MCQ:** AI2D, MathVista, MathVerse, ScienceQA, and others
- **Vision OEQ:** ChartQA, DocVQA, VQA-RAD, and others

---

## Technical Implementation

SophoSet uses inheritance and a standardized extraction pattern:

```python
from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class Ai2ArcDataset(BaseHFDataset):
    DATASET_NAME = "ai2_arc"

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        return QAData(
            key=self.get_key(index),
            question=row.get('question', ''),
            options=self._format_options(row),
            answer=row.get('answerKey', '')
        )
```

Dataset classes implement `extract_row_data()` to transform raw data into the standard QAData format. The base class handles data loading and iteration.

---

## Recent Developments

- Migration to litellm for unified LLM inference across providers
- Pydantic-based structured response validation
- DatasetExplorer utility for interactive dataset inspection
- Standardized options format across all MCQ datasets
- GitHub Actions CI/CD pipeline across Python 3.9-3.12
- MIT license for unrestricted use

---

## Use Case

Researchers evaluating models across multiple benchmarks can load any dataset with consistent code:

```python
from sophoset.text.mcq.mmlu_pro_data import MmluProDataset

dataset = MmluProDataset()
for qa_data in dataset:
    # Consistent interface across all datasets
    evaluate_model(qa_data.question, qa_data.options, qa_data.answer)
```

---

## Availability

**Repository:** https://github.com/csv610/SophoSet3

The framework is open source (MIT license) with full documentation, contributing guidelines, and issue templates.

**#LLM #MachineLearning #OpenSource #DataEngineering #Research #AI**
