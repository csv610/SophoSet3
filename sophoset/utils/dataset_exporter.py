"""
Dataset Exporter Module

This module provides functions for exporting a BaseHFDataset
instance to various file formats, such as JSON and LMDB.
"""

import json
import os
import logging
from dataclasses import asdict
from tqdm import tqdm
from typing import Any, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.lmdb_storage import LMDBStorage, Config

# Set up logging
logger = logging.getLogger(__name__)

class DatasetExporter:
    """A utility class for exporting BaseHFDataset objects to files."""

    @staticmethod
    def save_to_json(dataset: BaseHFDataset, output_dir: str = 'datasets', indent: int = 4, ensure_ascii: bool = False) -> None:
        """
        Save the entire dataset to a single JSON file with nice formatting,
        writing incrementally. The filename is automatically generated.

        Args:
            dataset: The dataset to export
            output_dir: Directory to save the JSON file.
            indent: Number of spaces for indentation (default: 4).
            ensure_ascii: If True, escape non-ASCII characters (default: False).
            
        Raises:
            ValueError: If dataset is None or invalid parameters
            IOError: If file writing fails
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        if not isinstance(output_dir, str):
            raise ValueError("output_dir must be a string")
        if not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer")
        # Get all available subsets
        subsets = dataset.get_subsets()
        if not subsets:
            subsets = ['default']
        
        os.makedirs(output_dir, exist_ok=True)
        dname = dataset.dataset_name.replace('/', '_')
        filename = os.path.join(output_dir, f"{dname}.json")
            
        total_rows = 0
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write('[')
                if indent > 0:
                    file.write('\n')
                first_item = True
                
                for subset in subsets:
                    splits = dataset.get_splits(subset)
                    for split in splits:
                        try:
                            dataset.load_dataset(split, subset)
                        except RuntimeError as e:
                            logger.warning(f"Skipping subset '{subset}' and split '{split}' due to error: {e}")
                            continue
                            
                        nrows = dataset.get_row_count()
                        
                        for irow in tqdm(range(nrows), desc=f"{subset}-{split}"):
                            try:
                                row = dataset.get_row_data(irow)
                                row_dict = asdict(row)
                                
                                if not first_item:
                                    file.write(',')
                                    if indent > 0:
                                        file.write('\n')
                                else:
                                    first_item = False
                                
                                json.dump(
                                    row_dict,
                                    file,
                                    ensure_ascii=ensure_ascii,
                                    indent=indent if indent > 0 else None
                                )
                                total_rows += 1
                            except Exception as e:
                                logger.error(f"Error processing row {irow} in {subset}-{split}: {e}")
                                continue
                
                if total_rows > 0 and indent > 0:
                    file.write('\n')
                file.write(']')
        except IOError as e:
            logger.error(f"Error writing to file {filename}: {e}")
            raise
            
        logger.info(f"Saved {total_rows} rows to {filename}")

    @staticmethod
    def _process_image_for_storage(image_data: Any) -> Optional[bytes]:
        """
        Process image data for storage in LMDB.
        
        Args:
            image_data: Can be a file path (str), URL (str), PIL Image, or bytes
            
        Returns:
            bytes: Serialized image data in JPEG format, or None if processing fails
            
        Raises:
            ValueError: If image_data is None or invalid
            FileNotFoundError: If local file path doesn't exist
            requests.RequestException: If URL download fails
        """
        from PIL import Image
        import io
        
        if image_data is None:
            return None
            
        try:
            if isinstance(image_data, str):
                # Handle file path or URL
                if image_data.startswith(('http://', 'https://')):
                    # Download image from URL
                    import requests
                    try:
                        response = requests.get(image_data, timeout=10)
                        response.raise_for_status()
                        img = Image.open(io.BytesIO(response.content))
                    except requests.RequestException as e:
                        logger.error(f"Failed to download image from URL {image_data}: {e}")
                        return None
                else:
                    # Handle local file path
                    if not os.path.exists(image_data):
                        logger.error(f"Image file not found: {image_data}")
                        return None
                    img = Image.open(image_data)
            elif hasattr(image_data, 'read') and callable(image_data.read):
                # Handle file-like objects
                img = Image.open(image_data)
            elif hasattr(image_data, 'save'):
                # Assume it's already a PIL Image or compatible
                img = image_data
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save to bytes buffer
            buffered = io.BytesIO()
            img.save(buffered, format='JPEG', quality=85)  # Add quality parameter
            return buffered.getvalue()
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    @staticmethod
    def save_to_lmdb(dataset: BaseHFDataset, output_dir: str = 'datasets') -> None:
        """
        Saves the entire dataset to an LMDB database.
        
        The method iterates through all available subsets and splits,
        and for each row, it generates a unique key and stores the
        serialized QAData object in the LMDB database.

        Args:
            dataset: The dataset to export
            output_dir: The directory to save the LMDB database.
            
        Raises:
            ValueError: If dataset is None or invalid parameters
            RuntimeError: If LMDB operations fail
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        if not isinstance(output_dir, str):
            raise ValueError("output_dir must be a string")
        dname = dataset.dataset_name.replace('/', '_') if hasattr(dataset, 'dataset_name') else 'dataset'
        db_path = os.path.join(output_dir, f"{dname}_lmdb")
        
        # Use the LMDBStorage context manager to ensure the connection is closed
        try:
            with LMDBStorage(config=Config(db_path=db_path)) as storage:
                total_rows = 0
                subsets = dataset.get_subsets()
                
                for subset in subsets:
                    splits = dataset.get_splits(subset)
                    for split in splits:
                        try:
                            dataset.load_dataset(split, subset)
                            nrows = dataset.get_row_count()
                            
                            for irow in tqdm(range(nrows), desc=f"{subset}-{split}"):
                                try:
                                    row_data = dataset.get_row_data(irow)
                                    
                                    # Create a copy to avoid modifying the original
                                    from copy import deepcopy
                                    row_data_copy = deepcopy(row_data)
                                    
                                    # Process images if they exist
                                    if hasattr(row_data_copy, 'images') and row_data_copy.images:
                                        processed_images = []
                                        for img in row_data_copy.images:
                                            if img:  # Only process non-None images
                                                processed_img = DatasetExporter._process_image_for_storage(img)
                                                if processed_img is not None:
                                                    processed_images.append(processed_img)
                                        # Replace the original images with processed byte data
                                        row_data_copy.images = processed_images
                                    
                                    # Generate a unique key for the LMDB entry
                                    key = dataset.get_key(irow)
                                    
                                    # Put the QAData object into the LMDB database
                                    storage.put(key, row_data_copy)
                                    total_rows += 1
                                except Exception as e:
                                    logger.error(f"Error processing row {irow} in {subset}-{split}: {e}")
                                    continue
                                
                        except Exception as e:
                            logger.error(f"Error processing subset '{subset}', split '{split}': {e}")
                            continue
            
            logger.info(f"Successfully saved {total_rows} rows to LMDB database at {db_path}")

        except Exception as e:
            logger.error(f"An error occurred while saving to LMDB: {str(e)}")
            raise

    @staticmethod
    def save(dataset: BaseHFDataset, format: str = 'lmdb', output_dir: str = 'datasets', **kwargs) -> None:
        """
        Save the dataset in the specified format.
        
        Args:
            dataset: The dataset to export
            format: The output format ('json' or 'lmdb')
            output_dir: Directory to save the output files
            **kwargs: Additional arguments to pass to the specific save method
            
        Raises:
            ValueError: If dataset is None or format is unsupported
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        if not isinstance(format, str):
            raise ValueError("format must be a string")
            
        format = format.lower()
        if format == 'json':
            return DatasetExporter.save_to_json(dataset, output_dir=output_dir, **kwargs)
        elif format == 'lmdb':
            return DatasetExporter.save_to_lmdb(dataset, output_dir=output_dir, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}. Must be 'json' or 'lmdb'.")
