# Medium Article: "Standardizing Dataset Schemas for LLM Evaluation: Introducing SophoSet"

---

## Introduction

The Hugging Face Hub contains over 100 publicly available evaluation datasets covering diverse domains—medical question answering, mathematical reasoning, visual question answering, and open-ended reasoning tasks. These datasets provide valuable benchmarks for evaluating large language models.

However, a systematic inconsistency exists across these datasets: they use different field names, data structures, and answer formats. This heterogeneity creates a preprocessing burden for researchers building evaluation frameworks that utilize multiple datasets.

This article describes the problem, presents SophoSet as a solution, and demonstrates its implementation and usage.

---

## Dataset Schema Inconsistencies

Multiple-choice question answering datasets from the Hugging Face Hub demonstrate significant structural variation:

**MMLU Dataset:**
```python
{
    "question": str,
    "choices": list,
    "answer": int
}
```

**GSM8K Dataset:**
```python
{
    "question": str,
    "answer": str
}
```

**AI2 ARC Dataset:**
```python
{
    "question": str,
    "choices": {
        "text": list,
        "label": list
    },
    "answerKey": str
}
```

**MathVista Dataset:**
```python
{
    "query": str,
    "options": list,
    "answer": str
}
```

**ChartQA Dataset:**
```python
{
    "question": str,
    "image": PIL.Image,
    "answer": str
}
```

These datasets contain functionally equivalent information but with different field names and data structures. This variation requires custom transformation code for each dataset.

### Implementation Burden

Building an evaluation framework across multiple datasets requires:

1. **Data format specification** — Understanding each dataset's structure
2. **Adapter implementation** — Writing transformation code
3. **Validation** — Testing transformations for correctness
4. **Maintenance** — Handling edge cases and updates

This preprocessing step introduces overhead proportional to the number of datasets used in an evaluation pipeline.

---

## SophoSet: A Unified Data Framework

SophoSet provides a consistent data schema across 100+ evaluation datasets. All datasets are normalized to a single structure:

```python
@dataclass
class QAData:
    key: str                          # Unique identifier
    question: str                     # The question text
    options: Dict[str, str]           # {"A": "option1", "B": "option2", ...}
    answer: str                       # The correct answer
    explanation: str = ""             # Optional explanation
    context: str = ""                 # Optional context
    images: List[str] = field(default_factory=list)  # For vision tasks
```

That's it. Every dataset, everywhere in the codebase, returns this same structure.

---

## How It Works: Under the Hood

SophoSet uses inheritance and a standardized extraction pattern:

```python
from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class Ai2ArcDataset(BaseHFDataset):
    DATASET_NAME = "ai2_arc"

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Transform raw Hugging Face row into standard QAData format"""
        question = row.get('question', '')
        choices = row.get('choices', {})
        options_list = choices.get('text', []) if isinstance(choices, dict) else []
        answer = row.get('answerKey', '')

        # Format options as dict with letter keys (A, B, C, D, etc.)
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

Every dataset class follows this pattern:
1. Load data from Hugging Face
2. Implement `extract_row_data()` to transform raw rows
3. Return standardized QAData objects

That's all. The base class handles everything else.

---

## Using SophoSet in Your Code

Here's how simple it is:

```python
from sophoset.text.mcq.ai2_arc_data import Ai2ArcDataset
from sophoset.text.mcq.mmlu_pro_data import MmluProDataset
from sophoset.vision.mcq.mathvista_data import MathVistaDataset

# Load any dataset the same way
arc_dataset = Ai2ArcDataset()
mmlu_dataset = MmluProDataset()
mathvista_dataset = MathVistaDataset()

# All return the same QAData structure
for qa_data in arc_dataset:
    print(qa_data.question)  # Works the same everywhere
    print(qa_data.options)   # Works the same everywhere
    print(qa_data.answer)    # Works the same everywhere
```

No custom preprocessing. No dataset-specific code paths. One consistent interface.

---

## Supported Datasets

SophoSet currently covers 100+ datasets across multiple categories:

**Text - Multiple Choice:**
- AI2-ARC, AIME, BigBench Hard, Medical Meadow MedicalQA, Medical Concepts QA, MedMCQA, MedQA USMLE 4-Option, MMLU-Pro, SciQ, WinoGrande

**Text - Open-Ended:**
- AIME 2025, DeepScaler, GPQA, GSM8K, GSM+ (Private), IMO Geometry, Math Lighteval, MathPlus, Medical Meadow Flashcards, Medical Meadow WikiDoc, Medical Questions, Medication QA, MedQA, MedQN V3, MedQuad, MetaMath QA, Olympiads, Open Med Cases, PubMedQA, Safety Bench, SciBench, SimpleQA, Toxic Prompts, TruthfulQA

**Vision - Multiple Choice:**
- AI2D, BLINK, Hidden Flaws (GPT-4V), MathV360K, MathVerse, MathVision, MathVista, ScienceQA, World Medical QA

**Vision - Open-Ended:**
- Animals, Camouflage, ChartQA, Chest X-Ray Pneumonia, Cultural VQA, DocVQA, IAM Line, IAM Sentences, Illusion Bench, InfoVQA, Kvasir VQA, MMStar, OCRBench V2, Olympiad Bench, Olympic Arena, Open Med Case Images, Open Med Images, PD12M, Real World QA, ROCO Radiology, SNLI-VE, TextVQA, Visit Bench, VLMs Are Blind, VQA-RAD

---

## Production Features

SophoSet isn't just for prototyping. It's built for production:

**✅ Structured Output with Pydantic**
```python
# Validated response format for LLM inference
from pydantic import BaseModel
from litellm import completion

class ModelResponse(BaseModel):
    answer: str
    confidence: float
    reasoning: str

response = completion(
    model="gpt-4",
    messages=messages,
    response_format=ModelResponse  # Auto-validated
)
```

**✅ Automated Testing & CI/CD**
- Runs pytest across Python 3.9, 3.10, 3.11, 3.12
- Linting with black, isort, flake8
- Coverage reporting
- Triggered on every push

**✅ Professional Documentation**
- MIT License (open source, zero restrictions)
- Code of Conduct (inclusive community)
- Contributing guidelines
- CHANGELOG tracking versions
- Issue & PR templates

---

## Recent Improvements

We just pushed a major refactor that includes:

1. **Migration to litellm** — Unified LLM interface across providers (OpenAI, Claude, Llama, etc.)
2. **DatasetExplorer** — Interactive way to browse and test datasets
3. **Standardized Options** — All MCQ datasets now use consistent {"A": "...", "B": "...", ...} format
4. **Structured Responses** — Pydantic validation ensures clean data

```python
from sophoset.utils.dataset_explorer import DatasetExplorer
from sophoset.text.mcq.ai2_arc_data import Ai2ArcDataset

dataset = Ai2ArcDataset()
explorer = DatasetExplorer(dataset)

# Interactive exploration
for qa_data in explorer.next_question():
    explorer.print_question(qa_data)
```

---

## Usage

SophoSet can be installed from source:

```bash
git clone https://github.com/csv610/SophoSet3.git
cd SophoSet3
pip install -e .
```

Datasets are accessed through their respective classes:

```python
from sophoset.text.mcq.mmlu_pro_data import MmluProDataset

dataset = MmluProDataset()
for qa_data in dataset:
    question = qa_data.question
    options = qa_data.options
    answer = qa_data.answer
```

All datasets implement the same interface, providing consistent field access across different sources.

---

## Utility and Scope

SophoSet addresses a systematic problem in LLM evaluation: dataset schema inconsistency. By providing a unified interface, it reduces development overhead for researchers and practitioners building evaluation frameworks.

The framework is intended for:
- Researchers evaluating models across multiple benchmarks
- Teams developing evaluation pipelines
- Practitioners requiring consistent dataset access patterns

SophoSet does not address model development, training, or inference optimization—it focuses on dataset standardization for evaluation workflows.

---

## Contributing

This is an open-source project. We need:
- **Additional datasets** — Submit PRs for datasets you want standardized
- **Bug reports** — Found an issue? Open a GitHub issue
- **Feedback** — What features would help your workflow?

**GitHub:** https://github.com/csv610/SophoSet3

---

## Conclusion

Dataset schema inconsistency is a systematic issue in LLM evaluation. SophoSet provides a standardized framework that reduces implementation overhead when working across multiple evaluation datasets.

The framework is open source (MIT license) with full documentation, automated testing, and community contribution guidelines.

**Repository:** https://github.com/csv610/SophoSet3

For additional information, questions, or contributions, please refer to the repository documentation and contributing guidelines.

---

*SophoSet is MIT licensed. Free to use, modify, and distribute.*
