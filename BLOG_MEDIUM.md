# Medium Article: "Standardizing 100+ Datasets for LLM Evaluation: Introducing SophoSet"

---

## Introduction

The Hugging Face Hub is a treasure trove of over 100 publicly available datasets perfect for evaluating large language models. From medical question answering to mathematical reasoning, from vision-language tasks to open-ended reasoning, the variety is incredible.

But there's a dirty secret nobody wants to talk about: **they're all incompatible**.

This article walks you through the problem, the solution, and how to use SophoSet to build evaluation pipelines without custom preprocessing for every single dataset.

---

## The Inconsistency Problem

Let me show you what I mean. Here are real dataset schemas from Hugging Face:

**MMLU Dataset:**
```python
{
    "question": "What is...",
    "choices": ["A", "B", "C", "D"],
    "answer": 0
}
```

**GSM8K Dataset:**
```python
{
    "question": "A bakery...",
    "answer": "42"
}
```

**AI2 ARC Dataset:**
```python
{
    "question": "What is...",
    "choices": {
        "text": ["A", "B", "C", "D"],
        "label": ["A", "B", "C", "D"]
    },
    "answerKey": "A"
}
```

**MathVista Dataset:**
```python
{
    "query": "What is...",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
}
```

**ChartQA Dataset:**
```python
{
    "question": "What is...",
    "image": PIL.Image,
    "answer": "123"
}
```

See the problem? Same type of data (question answering), completely different structures.

### The Real Cost

When I was building an LLM evaluation framework, I spent **3 weeks** writing custom adapters for 20 datasets:

1. **Data exploration** — Understanding each dataset's structure (2-3 hours per dataset)
2. **Adapter code** — Writing transformation logic (1-2 hours per dataset)
3. **Testing** — Ensuring transformations work correctly (1-2 hours per dataset)
4. **Debugging** — Fixing edge cases (2-3 hours per dataset)

That's roughly **80-100 hours** of work before I could even start actual evaluation research.

**This is preventable.**

---

## Enter SophoSet

SophoSet is a unified data framework that normalizes 100+ datasets into one consistent schema. No matter which dataset you use, you get back the same data structure:

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

## Getting Started

Install SophoSet:

```bash
pip install sophoset
# Or from source:
git clone https://github.com/csv610/SophoSet3.git
cd SophoSet3
pip install -e .
```

Load a dataset:

```python
from sophoset.text.mcq.mmlu_pro_data import MmluProDataset

dataset = MmluProDataset()
for qa_data in dataset:
    print(f"Q: {qa_data.question}")
    print(f"Options: {qa_data.options}")
    print(f"Answer: {qa_data.answer}")
```

That's it. No preprocessing. No custom code. Just standardized data.

---

## Why This Matters

As AI researchers and practitioners, we spend too much time on data plumbing. We should be focusing on:
- Model evaluation strategies
- Novel benchmarking approaches
- Research insights

Not:
- Figuring out which field contains the answer
- Writing custom transformations for each dataset
- Debugging edge cases in data loading

SophoSet eliminates that friction. It's not sexy. It's not a novel algorithm. But it saves you days of work.

---

## Contributing

This is an open-source project. We need:
- **Additional datasets** — Submit PRs for datasets you want standardized
- **Bug reports** — Found an issue? Open a GitHub issue
- **Feedback** — What features would help your workflow?

**GitHub:** https://github.com/csv610/SophoSet3

---

## Conclusion

LLM evaluation is becoming a critical part of AI development. But the tooling shouldn't get in the way. SophoSet removes one major obstacle: dataset inconsistency.

No more custom preprocessing. No more dataset-specific code paths. One unified interface for 100+ datasets.

If you're evaluating LLMs, building benchmarks, or researching model capabilities, try SophoSet. I think it'll save you significant time.

And if you have feedback or ideas, reach out. Open source thrives on community input.

Happy evaluating.

---

**Want to connect?** Find me on LinkedIn or GitHub @csv610
**Interested in contributing?** Check out CONTRIBUTING.md in the repo

---

*SophoSet is MIT licensed and completely open source. Build, modify, and distribute freely.*
