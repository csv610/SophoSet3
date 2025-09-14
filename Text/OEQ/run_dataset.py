import argparse

from data_processor import DataConfig, ModelConfig, DatasetProcessor

def process_arguments():
    """Process command line arguments."""
    parser = argparse.ArgumentParser(description='Run open-ended questions from different datasets using TextOEQ')
    
    # Dataset arguments
    parser.add_argument("-d", '--dataset', type=str, required=True,
                      help=f'Name of the dataset to run. Available: {", ".join(DatasetProcessor.DATASET_MODULES.keys())}')

    # Add update flag
    parser.add_argument('--update', action='store_true',
                      help='Update existing responses if they exist')
    
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
    return args
                
def run_dataset():
    args = process_arguments()

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
    
    # Create and run the processor
    processor = DatasetProcessor(
        data_config=data_config,
        model_config=model_config,
        update_existing=args.update
    )
    
    processor.process_dataset()
                
if __name__ == "__main__":
    run_dataset()
