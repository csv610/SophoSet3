# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT License for unrestricted open-source use
- GitHub Actions CI/CD pipeline with automated testing across Python 3.9-3.12
- Code of Conduct for community standards
- Dataset explorer integration across all dataset modules
- Structured response format with Pydantic validation in MCQ modules

### Changed
- Migrated from `any_llm` to `litellm` for LLM inference
- Standardized options format from lists to dictionaries with letter keys (A, B, C, etc.)
- Improved docstring formatting and consistency across modules
- Refactored data processing for production-readiness

### Removed
- Unused `parse_llm_response` parsing logic with structured output

## [0.1.0] - 2024-11-09

### Added
- Initial release of SophoSet
- Support for 100+ datasets across text and vision domains
- Dataset classes for MCQ (multiple-choice questions) and OEQ (open-ended questions)
- Text datasets: AI2-ARC, AIME, BigBench, Medical QA, MMLU-Pro, SciQ, WinoGrande, and more
- Vision datasets: AI2D, ChartQA, DocVQA, MathVista, ScienceQA, VQA, and more
- Unified data format with QAData model
- LMDB dataset export functionality
- Streamlit and Gradio UI support
- Comprehensive documentation and dataset migration guide

[Unreleased]: https://github.com/csv610/SophoSet3/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/csv610/SophoSet3/releases/tag/v0.1.0
