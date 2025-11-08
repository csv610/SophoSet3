import importlib
import logging
import os
import sys
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

# Configure logging to only write to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

from sophoset.core.base_hf_dataset import BaseHFDataset
from sophoset.utils.lmdb_storage import DataStorage

from text_oeq import TextOEQ

@dataclass
class DataConfig:
    """Configuration for dataset loading and processing."""
    name: str
    subset: str = ""
    split: str = ""
    max_samples: Optional[int] = None
    random_sample: bool = False
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'name': self.name,
            'subset': self.subset,
            'split': self.split,
            'max_samples': self.max_samples,
            'random_sample': self.random_sample,
            'seed': self.seed
        }

@dataclass
class ModelConfig:
    """Configuration for the model used in evaluation."""
    name: str = 'mistralai/mistral-7b-instruct'
    provider: str = 'openrouter'
    temperature: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'name': self.name,
            'provider': self.provider,
            'temperature': self.temperature
        }


class DatasetProcessor:
    """Processes datasets using a specified model and handles storage of responses."""
    
    # Dictionary mapping dataset names to their module names
    DATASET_MODULES = {
        "aime2025": "aime2025_data",
        "gpqa": "gpqa_data",
        "gsm8k": "gsm8k_data",
        "gsmplus": "gsmplus_data",
        "imo_geometry": "imo_geometry_data",
        "math_lighteval": "math_lighteval_data",
        "mathplus": "mathplus_data",
        "medical_meadow_flashcards": "medical_meadow_flashcards_data",
        "medical_meadow_wikidoc": "medical_meadow_wikidoc_data",
        "medicalquestions": "medicalquestions_data",
        "medicationqa": "medicationqa_data",
        "medqa": "medqa_data",
        "medqnav3": "medqnav3_data",
        "medquad": "medquad_data",
        "medquad_medqna": "MedQuad-MedQnA_data",
        "metamathqa_40k": "metamathqa-40k_data",
        "metamathqa": "metamathqa_data",
        "olympiads": "olympiads_data",
        "pubmedqa": "pubmedqa_data",
        "scibench": "scibench_data",
        "simpleqa": "simpleqa_data",
        "truthfulqa": "truthfulqa_data"
    }
    """Processes datasets using a specified model and handles storage of responses."""
    
    @classmethod
    def load_dataset(cls, dataset_name: str) -> BaseHFDataset:
        """Dynamically load a dataset module and return an instance of the dataset class."""
        if dataset_name not in cls.DATASET_MODULES:
            error_msg = f"Unknown dataset: {dataset_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        module_name = cls.DATASET_MODULES[dataset_name]
        logger.debug(f"Loading dataset module: {module_name}")
        try:
            module = importlib.import_module(module_name)
            for obj in module.__dict__.values():
                if (isinstance(obj, type) and 
                    issubclass(obj, BaseHFDataset) and 
                    obj != BaseHFDataset):
                    return obj()
            error_msg = f"Could not find dataset class in {module_name}"
            logger.error(error_msg)
            raise ImportError(error_msg)
        except ImportError as e:
            error_msg = f"Failed to import dataset {dataset_name}: {str(e)}"
            logger.error(error_msg)
            raise ImportError(error_msg)
            
    def __init__(self, data_config: 'DataConfig', model_config: 'ModelConfig', update_existing: bool = False):
        """Initialize the dataset processor.
        
        Args:
            data_config: Configuration for dataset loading and processing
            model_config: Configuration for the model
            update_existing: Whether to update existing responses
        """
        self.data_config = data_config
        self.model_config = model_config
        self.update_existing = update_existing

        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        dbname = data_config.name
        dbname = dbname.replace(" ", "_").replace("/", "_")
        dbname = os.path.join(results_dir, f"{dbname}_lmdb")

        self.storage = DataStorage(dbname)

        self.llm_model = TextOEQ(
            model=model_config.name,
            provider=model_config.provider,
            temperature=model_config.temperature
        )
    
    def process_dataset(self) -> None:
        """Process the entire dataset across all subsets and splits."""
        try:
            data_object = self._load_dataset()
            subsets = data_object.get_subsets() or [self.data_config.subset]
            
            for subset in subsets:
                self._process_subset(data_object, subset)
                
        except Exception as e:
            print(f"Error processing dataset {self.data_config.name}: {str(e)}")
            raise
    
    def _load_dataset(self) -> BaseHFDataset:
        """Load and return the dataset using the class method."""
        return self.__class__.load_dataset(self.data_config.name)
    
    def _process_subset(self, data_object: BaseHFDataset, subset: str) -> None:
        """Process a single subset of the dataset across all available splits.
        
        Args:
            data_object: The dataset object to process
            subset: The name of the subset to process
            
        Raises:
            Exception: If there's an error processing the subset
        """
        # Get available splits for this subset
        try:
            splits = data_object.get_splits(subset)
            if not splits:
                logger.warning(f"No splits found for subset: {subset}")
                return
                
            logger.info(f"Processing subset: {subset}")
            logger.info(f"Available splits: {', '.join(splits)}")
                
            for split in splits:
                logger.info(f"Processing split: {split}")
                try:
                    data_object.load_dataset(
                        split_name=split,
                        subset_name=subset
                    )
                except Exception as e:
                    logger.error(f"Error loading dataset for split {split}: {str(e)}")
                    raise
                    
                samples = data_object.get_samples(
                    max_samples=self.data_config.max_samples,
                    random_sample=self.data_config.random_sample,
                    seed=self.data_config.seed
                )
                
                if not samples:
                    logger.warning(f"No samples found for split: {split}")
                    continue
                
                # Process samples with progress bar
                total_samples = len(samples)
                logger.info(f"Processing {total_samples} samples...")
                
                # Initialize tqdm progress bar
                progress_bar = tqdm(
                    samples,
                    desc=f"{subset}-{split}",
                    unit="sample",
                    leave=True  # Ensures the progress bar stays visible after completion
                )
                
                for sample in progress_bar:
                    try:
                        self._process_sample(sample)
                    except Exception as e:
                        logger.error(f"Error processing sample: {str(e)}")
                        if not self.update_existing:
                            raise
                    
        except Exception as e:
            logger.error(f"Error processing subset {subset}: {str(e)}")
            raise
    
    def _process_sample(self, sample: Any) -> None:
        """Process a single sample from the dataset."""
        try:
            storage_key = f"{sample.key}:{self.model_config.name}"
            
            if not self.update_existing and self.storage.get(storage_key) is not None:
                logger.debug(f"Skipping existing response for key: {storage_key}")
                return
            
            try:
                logger.debug(f"Getting response for sample: {storage_key}")
                response = self.llm_model.get_response(sample.question)
                self.storage.put(storage_key, response)
                logger.info(f"Successfully stored response for key: {storage_key}")
            except Exception as e:
                logger.error(f"Error getting/storing response for {storage_key}: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error processing sample with key {storage_key}: {str(e)}")
            raise
                
