"""
Base Dataset Handler for Hugging Face Datasets

This module provides a base class for handling various Hugging Face datasets
with common functionality for loading, accessing, and processing dataset samples.
"""

import logging
import base64
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class QAData:
    """Standardized data structure for question-answer pairs.

    Attributes:
        key: Unique identifier for the sample
        context: Optional contextual information
        question: The question text
        images: List of image data (URLs, base64, or PIL Images)
        options: Dictionary of answer options (e.g., {'A': 'option1', 'B': 'option2'})
        answer: The correct answer
        explanation: Explanation of the correct answer
        metadata: Additional metadata about the sample
    """
    key: str
    context: str = ""
    question: str = ""
    images: List[Any] = field(default_factory=list)
    options: Dict[str, str] = field(default_factory=dict)
    answer: str = ""
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseHFDataset(ABC):
    """Base class for Hugging Face dataset handlers.
    
    This class provides common functionality for loading and interacting with
    Hugging Face datasets. Subclasses should implement dataset-specific logic
    in the extract_row_data method.
    """
    
    def __init__(self, dataset_name: str):
        """Initialize the dataset handler.

        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.subset = None
        self.split = None
        logger.debug(f"Initialized dataset handler for {dataset_name}")

    def load_dataset(self, split_name: str, subset_name: str = 'default') -> None:
        """Load the dataset from Hugging Face Hub with the specified split and subset.

        Args:
            split_name: The name of the split to load (e.g., 'train', 'validation', 'test')
            subset_name: Optional subset/configuration name. Defaults to 'default'.

        Raises:
            ValueError: If the specified subset or split is not available
            RuntimeError: If there's an error loading the dataset
        """
        try:
            logger.info(f"Loading dataset {self.dataset_name} (subset: {subset_name}, split: {split_name})")
            self.split = split_name
            self.subset = subset_name
            self.dataset = load_dataset(self.dataset_name, subset_name)[split_name]
            logger.info(f"Successfully loaded {len(self.dataset)} samples")
        except KeyError as e:
            available_splits = self.get_splits(subset_name)
            error_msg = (
                f"Split '{split_name}' not found in dataset {self.dataset_name} "
                f"(subset: {subset_name}). Available splits: {available_splits}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Error loading {self.dataset_name} dataset "
                f"(split: {split_name}, subset: {subset_name}): {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def is_dataset_loaded(self) -> bool:
        """Check if a dataset is currently loaded.
        
        Returns:
            bool: True if dataset is loaded, False otherwise
        """
        return self.dataset is not None
    
    def get_column_names(self) -> set:
        """Get the column names of the loaded dataset.
        
        Returns:
            set: Set of column names
            
        Raises:
            RuntimeError: If no dataset is loaded
        """
        if not self.is_dataset_loaded():
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
        return set(self.dataset.column_names)
        
    @staticmethod
    def get_formatted_options(options: Union[List[str], Dict[str, str]]) -> Dict[str, str]:
        """Format choices with letters (A, B, C, ..., Z) for display.

        Converts a list of options to a dictionary with letter keys (A, B, C, etc.).
        If the input is already a dictionary, returns it as-is.

        Args:
            options: List or Dict of choice strings

        Returns:
            Dict with letter keys (A, B, C, ..., Z) and option values

        Example:
            >>> options = ['Python', 'Java', 'C++', 'JavaScript']
            >>> formatted = BaseHFDataset.get_formatted_options(options)
            >>> print(formatted)
            {'A': 'Python', 'B': 'Java', 'C': 'C++', 'D': 'JavaScript'}

        Note:
            Maximum 26 options (A-Z) are supported. Options beyond Z are ignored.
        """
        if not options:
            return {}

        # If already a dict, return as is
        if isinstance(options, dict):
            return options

        # If it's a list, convert to dict with letter keys
        formatted = {}
        # Generate letters from A to Z
        letters = [chr(65 + i) for i in range(26)]
        for i, opt in enumerate(options):
            if i < len(letters):
                formatted[letters[i]] = opt
            else:
                logger.warning(
                    f"Option {i} ({opt}) exceeds maximum of 26 options. Skipping."
                )

        return formatted
    
    def get_row_count(self) -> int:
        """Return the total number of questions in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
            
        Raises:
            RuntimeError: If no dataset is loaded
        """
        if not self.is_dataset_loaded():
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
        return len(self.dataset)
    
    def get_row_data(self, index: int) -> 'QAData':
        """Get data for a specific row by index.

        Args:
            index: The index of the row to retrieve.

        Returns:
            QAData object containing the row data.

        Raises:
            RuntimeError: If no dataset is loaded
            IndexError: If the index is out of range

        Example:
            >>> from sophoset.text.mcq.mmlu_data import MMLUDataset
            >>> dataset = MMLUDataset()
            >>> dataset.load_dataset(split_name='test', subset_name='anatomy')
            >>> sample = dataset.get_row_data(0)
            >>> print(f"Question: {sample.question}")
            >>> print(f"Answer: {sample.answer}")
        """
        if not self.is_dataset_loaded():
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Row index {index} out of range (0-{len(self.dataset)-1})")
        return self.extract_row_data(self.dataset[index], index)

    def get_key(self, index: int) -> str:
        return f"{self.subset}/{self.split}/{index}"
        
    def get_samples(self, max_samples: Optional[int] = None, random_sample: bool = False, seed: Optional[int] = None) -> List[QAData]:
        """Get samples from the loaded dataset with optional random sampling and sample limiting.
        
        Args:
            max_samples: Maximum number of samples to return. If None, returns all samples.
            random_sample: If True, samples are selected randomly. If False, returns first n samples.
            seed: Optional random seed for reproducible random sampling.
            
        Returns:
            List[QAData]: A list of samples from the dataset.
            
        Raises:
            RuntimeError: If no dataset is loaded.
            ValueError: If max_samples is not a positive integer.
        """
        if not self.is_dataset_loaded():
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
            
        if max_samples is not None and (not isinstance(max_samples, int) or max_samples <= 0):
            raise ValueError("max_samples must be a positive integer or None")
            
        # Get all indices
        indices = list(range(len(self.dataset)))
        
        # Apply random sampling if requested
        if random_sample:
            if seed is not None:
                random.seed(seed)
            # Sample without replacement to ensure unique indices
            if max_samples is not None and max_samples < len(indices):
                indices = random.sample(indices, max_samples)
            else:
                random.shuffle(indices)
        # If not random and max_samples is specified, take first n samples
        elif max_samples is not None:
            indices = indices[:max_samples]
            
        # Ensure indices are in ascending order for consistent processing
        indices = sorted(indices)
        
        # Collect samples
        samples = []
        for idx in indices:
            try:
                row_data = self.get_row_data(idx)
                samples.append(row_data)
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")
                continue

        logger.debug(f"Successfully loaded {len(samples)} samples (requested: {max_samples})")
        return samples
    
    def get_subsets(self) -> List[str]:
        """Get the list of available subsets/configurations for a dataset.

        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.

        Returns:
            List of available subset/configuration names.

        Raises:
            ValueError: If the dataset cannot be accessed.
        """
        try:
            configs = get_dataset_config_names(self.dataset_name)

            # If we only get 'default' and the dataset isn't available on Hub,
            # try to detect actual configs from the cache
            if configs == ['default']:
                cached_configs = self._get_cached_configs()
                if cached_configs:
                    return cached_configs

            return configs
        except Exception as e:
            # If getting configs from Hub fails, try to get them from cache
            cached_configs = self._get_cached_configs()
            if cached_configs:
                return cached_configs
            raise ValueError(f"Error getting available subsets for dataset {self.dataset_name}: {str(e)}")

    def _get_cached_configs(self) -> List[str]:
        """Get available configurations from the local Hugging Face cache.

        Returns:
            List of configuration names found in cache, or empty list if none found.
        """
        try:
            cache_dir = Path.home() / '.cache/huggingface/datasets'
            # Convert dataset name format: 'allenai/ai2_arc' -> 'allenai___ai2_arc'
            dataset_cache_name = self.dataset_name.replace('/', '___')
            dataset_cache_path = cache_dir / dataset_cache_name

            if dataset_cache_path.exists() and dataset_cache_path.is_dir():
                # Get all subdirectories which represent configs
                configs = [
                    d.name for d in dataset_cache_path.iterdir()
                    if d.is_dir()
                ]
                return sorted(configs) if configs else []
        except Exception:
            pass

        return []
    
    def get_splits(self, subset) -> List[str]:
        """Get the list of available splits for a dataset (and optionally a specific subset).
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.
            subset: Optional subset/configuration name. If None, tries to get splits for the dataset directly.
            
        Returns:
            List of available split names.
            
        Raises:
            ValueError: If the dataset or subset cannot be accessed.
        """
        try:
            if subset:
                return get_dataset_split_names(self.dataset_name, subset)
            return get_dataset_split_names(self.dataset_name)
        except Exception as e:
            raise ValueError(f"Error getting available splits for dataset {self.dataset_name} (subset: {subset}): {str(e)}")
    
    @abstractmethod
    def extract_row_data(self, row: Dict[str, Any], index: int) -> 'QAData':
        """Extract and format data from a dataset row.
        
        This method must be implemented by subclasses to define how to extract
        and format data from a single row of the dataset.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        pass
    
    def get_random_row_index(self) -> int:
        """Get a random question index.
        
        Returns:
            int: A random valid index within the dataset.
            
        Raises:
            RuntimeError: If no dataset is loaded
        """
        if not self.is_dataset_loaded():
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
        return random.randint(0, len(self.dataset) - 1)
    
    def validate_index(self, index: int) -> int:
        """Ensure the index is within valid range.
        
        Args:
            index: The index to validate.
            
        Returns:
            A valid index within the dataset bounds.
            
        Raises:
            RuntimeError: If no dataset is loaded
        """
        if not self.is_dataset_loaded():
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
        return max(0, min(index, len(self.dataset) - 1))
        
    def get_image_data(self, image: Union[str, Image.Image, None], format: str = "PNG") -> str:
        """Get image data in a consistent format.
        
        Handles different types of image inputs:
        - If input is a URL, returns it as-is
        - If input is a file path, loads and converts to base64
        - If input is a PIL Image, converts to base64
        
        Args:
            image: PIL Image, file path, or URL string
            format: Output format for conversion (default: "PNG")
            
        Returns:
            str: Either the original URL or a base64-encoded data URL
            
        Raises:
            ValueError: If the image format is not supported or image cannot be loaded
            FileNotFoundError: If the image file path doesn't exist
        """
        if not image:
            return ""
            
        # If it's already a URL, return it as is
        if isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
            return image
            
        # If it's a string but not a URL, try to open it as a local file
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Could not open image from path: {image}. Error: {str(e)}") from e
        
        # At this point, image should be a PIL Image
        if not hasattr(image, 'save'):
            raise ValueError("Input must be a PIL Image, file path, or URL")
            
        if format.upper() not in ["PNG", "JPEG", "GIF"]:
            raise ValueError(f"Unsupported image format: {format}. Supported formats: PNG, JPEG, GIF")
            
        buffered = BytesIO()
        
        # Handle transparency for formats that don't support it
        if format.upper() in ["JPEG"] and image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        
        # Save image to buffer
        try:
            image.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/{format.lower()};base64,{img_str}"
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}") from e
