"""SophoSet - Unified Hugging Face datasets management framework for VLMs and LLMs.

SophoSet provides a consistent interface to work with 100+ standardized datasets from
Hugging Face Hub for evaluating vision-language models (VLMs) and large language models (LLMs).

Example:
    Basic usage of SophoSet:

    >>> from sophoset.text.mcq.mmlu_data import MMLUDataset
    >>> dataset = MMLUDataset()
    >>> dataset.load_dataset(split_name='test', subset_name='anatomy')
    >>> samples = dataset.get_samples(max_samples=10)
    >>> for sample in samples:
    ...     print(f"Question: {sample.question}")
    ...     print(f"Answer: {sample.answer}")
"""

import logging
from typing import Optional

__version__ = "0.1.0"

# Configure logging for the package
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for SophoSet.

    Args:
        level: Logging level (default: logging.INFO).
               Use logging.DEBUG for verbose output.

    Example:
        >>> import logging
        >>> from sophoset import setup_logging
        >>> setup_logging(level=logging.DEBUG)
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Set logging for all sophoset modules
    logger = logging.getLogger('sophoset')
    logger.setLevel(level)
    logger.addHandler(handler)


# Make key classes easily importable
from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_explorer import DatasetExplorer
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.lmdb_storage import LMDBStorage

__all__ = [
    'BaseHFDataset',
    'QAData',
    'DatasetExplorer',
    'DatasetExporter',
    'LMDBStorage',
    'setup_logging',
]
