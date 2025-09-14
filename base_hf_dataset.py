"""
Base Dataset Handler for Hugging Face Datasets

This module provides a base class for handling various Hugging Face datasets
with common functionality for loading, accessing, and processing dataset samples.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
import base64
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import random
import json
import os

from dataclasses import dataclass, field, asdict

@dataclass
class QAData:
    key: str
    context : str = ""
    question: str = ""
    images: List[Any] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    answer: str = ""
    explanation: str = ""

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
            split: The data split to load (e.g., 'train', 'validation', 'test').
                  If None, will try to determine available splits dynamically.
            subset: The dataset subset/configuration to use. If None, will use the default subset.
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.subset  = None

    def load_dataset(self, split_name: str, subset_name: str = 'default') -> None:
        """Load the dataset from Hugging Face Hub with the specified split and subset
        
        Args:
            split_name: The name of the split to load (e.g., 'train', 'validation', 'test')
            subset_name: Optional subset/configuration name. If None, loads the default subset.
            
        Raises:
            ValueError: If the specified subset or split is not available
            RuntimeError: If there's an error loading the dataset
        """
        try:
            self.split = split_name
            self.subset = subset_name
            self.dataset = load_dataset(self.dataset_name, subset_name)[split_name]
        except KeyError as e:
            raise ValueError(
                f"Split '{split_name}' not found in dataset {self.dataset_name} (subset: {subset_name}). "
                f"Available splits: {self.get_splits(subset_name)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error loading {self.dataset_name} dataset (split: {split_name}, subset: {subset_name}): {str(e)}"
            ) from e

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
    def get_formatted_options(options: List[str]) -> List[str]:
        """Format choices with letters (A, B, C, ...) for display.
        
        Args:
            choices: List of choice strings
            
        Returns:
            List of formatted choice strings with letters
        """
        if not options:
            return []
        
        return [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    
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
                print(f"Error processing row {idx}: {str(e)}")
                continue
                
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
            return configs 
        except Exception as e:
            raise ValueError(f"Error getting available subsets for dataset {self.dataset_name}: {str(e)}")
    
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
