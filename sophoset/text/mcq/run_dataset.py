import argparse
import importlib
import json
import os
import random
import sys

from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Any, Union

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from text_mcq import TextMCQ

@dataclass
class DataConfig:
    """Configuration for dataset loading and processing."""
    name: str
    subset: str = ''
    split: str = ''
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


from sophoset.core.base_hf_dataset import BaseHFDataset, QAData


# Dictionary mapping dataset names to their module names
DATASET_MODULES = {
    'ai2_arc': 'ai2_arc_data',
    'aime_1983_2024': 'aime-1983-2024_data',
    'bigbenchhard': 'bigbenchhard_data',
    'deepscaler': 'deepscaler_data',
    'medicalconceptsqa': 'medicalconceptsqa_data',
    'medical_meadow_medicalqa': 'medical_meadow_medicalqa_data',
    'medmcqa': 'medmcqa_data',
    'medqa_usmle': 'medqa_usmle_4options_data',
    'mmlu': 'mmlu_data',
    'mmlu_pro': 'mmlu_pro_data',
    'sciq': 'sciq_data',
    'winogrande': 'winogrande_data',
}

def load_dataset(dataset_name: str) -> Type[BaseHFDataset]:
    """Load and return the specified dataset class."""
    if dataset_name not in DATASET_MODULES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {', '.join(DATASET_MODULES.keys())}")
    
    module_name = DATASET_MODULES[dataset_name]
    try:
        module = importlib.import_module(module_name)
        for name, obj in module.__dict__.items():
            if (isinstance(obj, type) and 
                issubclass(obj, BaseHFDataset) and 
                obj != BaseHFDataset):
                return obj()
        raise ImportError(f"Could not find dataset class in {module_name}")
    except ImportError as e:
        raise ImportError(f"Failed to import dataset {dataset_name}: {str(e)}")

def process_subset(data_object: BaseHFDataset, data_config: DataConfig, model_config: ModelConfig) -> None:
    """Process a dataset using the TextMCQ model.
    
    Args:
        dataset: The dataset to process
        data_config: Configuration for dataset loading and processing
        model_config: Configuration for the model
    """
    # Load the dataset with specified subset and split
    data_object.load_dataset(split_name=data_config.split, subset_name=data_config.subset)
    
    # Initialize the TextMCQ model
    mcq_model = TextMCQ(
        model=model_config.name,
        provider=model_config.provider, 
        temperature=model_config.temperature
    )
    
    # Get the dataset samples
    samples = data_object.get_samples(
        max_samples=data_config.max_samples,
        random_sample=data_config.random_sample,
        seed=data_config.seed
    )
    
    correct = 0
    total = len(samples)
    
    print(f"\nEvaluating {total} samples from {data_object.__class__.__name__}...")
    
    for i, sample in enumerate(samples, 1):
        try:
            # Get model response
            response = mcq_model.get_response(sample.question, sample.options)
            
            # Check if the answer is correct
            is_correct = (response.answer.lower() == sample.answer.lower())
            if is_correct:
                correct += 1
            
            # Print progress
            print(f"\rProcessed {i}/{total} | Accuracy: {correct/i:.2%}", end="")
            
        except Exception as e:
            print(f"\nError processing sample {i}: {str(e)}")
    
    # Print final results
    print(f"\n\nFinal Results for {data_object.__class__.__name__}:")
    print(f"Total samples: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {correct/total:.2%}")

def process_dataset(data_config: DataConfig, model_config: ModelConfig) -> None:
    """Process a dataset across all its subsets and splits.
    
    Args:
        data_config: Configuration for dataset loading and processing
        model_config: Configuration for the model
    """
    try:
        print(f"\nProcessing dataset: {data_config.name}")
        # Load the specified dataset
        data_object = load_dataset(data_config.name)
        
        # Get available subsets (configurations) for this dataset
        subsets = data_object.get_subsets() or [data_config.subset]

        for subset in subsets:
            try:
                # Update the subset in data_config
                data_config.subset = subset
                
                # Get available splits for this subset
                splits = data_object.get_splits(subset) or [data_config.split]
                
                for split in splits:
                    try:
                        # Update the split in the config
                        data_config.split = split
                        
                        print(f"Subset: {subset}, Split: {split}")
                        
                        # Process this specific subset and split
                        process_subset(
                            data_object= data_object,
                            data_config=data_config,
                            model_config=model_config
                        )
                        
                    except Exception as e:
                        print(f"Warning: Could not process {subset}/{split}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Could not process subset {subset}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error loading dataset {data_config.name}: {str(e)}")
        raise
                
def main():
    parser = argparse.ArgumentParser(description='Run MCQs from different datasets using TextMCQ')
    
    # Dataset arguments
    parser.add_argument("-d", '--dataset', type=str, required=True,
                      help=f'Name of the dataset to run. Available: {", ".join(DATASET_MODULES.keys())}')
    
    # Sampling arguments
    parser.add_argument("-n", '--max-samples', type=int, default=None,
                      help='Maximum number of samples to process per split')
    parser.add_argument("-r", '--random-sample', action='store_true',
                      help='Whether to sample randomly')
    parser.add_argument("-s", "--seed", type=int, default=None,
                      help='Random seed for reproducibility')
    
    # Model arguments
    parser.add_argument("-m", "--model", type=str, default='mistralai/mistral-7b-instruct',
                      help='Model name to use for evaluation')
    parser.add_argument("--provider", type=str, default='openrouter',
                      help='Model provider (e.g., openrouter, openai)')
    parser.add_argument("--temperature", type=float, default=0.7,
                      help='Sampling temperature for model generation')
    
    args = parser.parse_args()
    
    try:
        # Create configurations
        data_config = DataConfig(
            name=args.dataset,
            max_samples=args.max_samples,
            random_sample=args.random_sample,
            seed=args.seed
        )
        
        model_config = ModelConfig(
            name=args.model,
            provider=args.provider,
            temperature=args.temperature
        )
        
        process_dataset(data_config, model_config)
                
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
