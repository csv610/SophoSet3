import os
import sys
import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import gradio as gr

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

@dataclass
class StateParams:
    """Dataclass to manage application state parameters."""
    modality: Optional[str] = None
    questions_type: Optional[str] = None
    dataset_name: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None

class DataViewer:
    """A class to handle dataset viewing and interaction in Gradio."""
    
    def __init__(self):
        """Initialize the DataViewer with default settings."""
        self.current_question = 0
        self.dataset = None
        self.dataset_loaded = False
        self.total_questions = 0
        self.current_state = StateParams()
        self.previous_state = StateParams()

    def get_available_datasets(self, modality: str, questions_type: str) -> List[str]:
        """
        Scan the project structure to find available datasets for the selected modality and questions type.
        
        Returns:
            List of available dataset names
        """
        # Return empty list if required fields are not set
        if not modality or not questions_type:
            logger.warning("Modality or question type not selected. Skipping dataset scan.")
            return []
            
        base_dir = Path(__file__).parent
        dataset_dir = base_dir / modality / questions_type
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return []
            
        # Look for Python files that match the pattern [dataset_name]_data.py
        dataset_files = list(dataset_dir.glob("*_data.py"))
        
        # Extract dataset names by removing '_data.py' from filenames
        datasets = [f.stem.replace('_data', '') for f in dataset_files]
        logger.info(f"Found datasets: {datasets}")
        
        return sorted(datasets)
    
    def get_dataset_class(self, modality: str, questions_type: str):
        """Dynamically import and return the dataset class."""
        try:
            # Format the module path
            module_name = f"{modality}.{questions_type}.datasets.{questions_type.lower()}_dataset"
            logger.info(f"Attempting to import module: {module_name}")
            
            # Import the module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                logger.error(f"Module not found: {module_name}")
                return None
                
            module = importlib.import_module(module_name)
            
            # Get the dataset class (assumes class name is like 'MCQDataset' or 'OEQDataset')
            class_name = f"{questions_type}Dataset"
            if hasattr(module, class_name):
                logger.info(f"Successfully imported class: {class_name}")
                return getattr(module, class_name)
                
            logger.error(f"Class not found: {class_name} in module {module_name}")
            return None
                
        except Exception as e:
            logger.exception("An exception occurred while importing the dataset class.")
            return None

    def get_dataset_instance(self, modality: str, questions_type: str, dataset_name: str):
        """
        Create and return a dataset instance without loading data.
        
        Returns:
            The initialized dataset instance
        """
        try:
            # Get the dataset class
            dataset_class = self.get_dataset_class(modality, questions_type)
            
            if dataset_class is None:
                return None
                
            # Create an instance of the dataset
            dataset = dataset_class()
            logger.info(f"Successfully created dataset instance for: {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.exception("An exception occurred while creating the dataset instance.")
            return None

    def get_subsets(self, modality: str, questions_type: str, dataset_name: str) -> List[str]:
        """Get available subsets for the selected dataset."""
        if not all([modality, questions_type, dataset_name]):
            return []
            
        try:
            dataset = self.get_dataset_instance(modality, questions_type, dataset_name)
            if dataset is None:
                return []
                
            available_subsets = dataset.get_subsets()
            return available_subsets if available_subsets else []
            
        except Exception as e:
            logger.exception("An exception occurred while getting subsets.")
            return []

    def get_splits(self, modality: str, questions_type: str, dataset_name: str, subset: str) -> List[str]:
        """Get available splits for the selected dataset and subset."""
        if not all([modality, questions_type, dataset_name]):
            return ['train']
            
        try:
            dataset = self.get_dataset_instance(modality, questions_type, dataset_name)
            if dataset is None:
                return ['train']
                
            if hasattr(dataset, 'get_splits'):
                available_splits = dataset.get_splits(subset=subset)
                return available_splits if available_splits else ['train']
            else:
                return ['train']
                
        except Exception as e:
            logger.exception("An exception occurred while getting splits.")
            return ['train']

    def load_dataset(self, modality: str, questions_type: str, dataset_name: str, subset: str, split: str) -> Tuple[str, bool]:
        """Load the selected dataset."""
        if not all([modality, questions_type, dataset_name]):
            return "Please select all required fields (modality, question type, and dataset)", False
            
        try:
            # Get the dataset instance
            dataset = self.get_dataset_instance(modality, questions_type, dataset_name)
            if dataset is None:
                return "Failed to initialize dataset", False
                
            # Load the dataset with the selected subset and split
            subset = subset if subset else None
            split = split if split else 'train'
            
            # Load the data
            dataset_loaded = dataset.load(subset=subset, split=split)
            
            if dataset_loaded is None:
                return "Failed to load dataset", False
                
            # Update instance state
            self.dataset = dataset_loaded
            self.dataset_loaded = True
            self.total_questions = len(dataset_loaded)
            self.current_question = 0
            
            # Update current state
            self.current_state = StateParams(
                modality=modality,
                questions_type=questions_type,
                dataset_name=dataset_name,
                subset=subset,
                split=split
            )
            
            logger.info("Dataset loading process completed successfully.")
            return f"Dataset loaded successfully! Total questions: {self.total_questions}", True
            
        except Exception as e:
            logger.exception("An exception occurred during the dataset loading process.")
            return f"Error loading dataset: {str(e)}", False

    def navigate_question(self, direction: str) -> Tuple[int, str, str]:
        """Navigate through questions."""
        if not self.dataset_loaded or self.total_questions == 0:
            return 0, "No dataset loaded", ""
            
        if direction == "previous" and self.current_question > 0:
            self.current_question -= 1
        elif direction == "next" and self.current_question < self.total_questions - 1:
            self.current_question += 1
        
        # Get current question data
        question_html, answer_html = self.get_question_display(self.current_question)
        
        return self.current_question + 1, question_html, answer_html

    def set_question_index(self, question_num: int) -> Tuple[str, str]:
        """Set specific question index."""
        if not self.dataset_loaded or self.total_questions == 0:
            return "No dataset loaded", ""
            
        # Convert to 0-based index and clamp to valid range
        self.current_question = max(0, min(question_num - 1, self.total_questions - 1))
        
        # Get current question data
        question_html, answer_html = self.get_question_display(self.current_question)
        
        return question_html, answer_html

    def get_question_display(self, question_idx: int) -> Tuple[str, str]:
        """Get HTML display for the current question and answer."""
        if not self.dataset_loaded or question_idx >= self.total_questions:
            print(f"Cannot display question: loaded={self.dataset_loaded}, idx={question_idx}, total={self.total_questions}")
            return "No question available", ""
            
        try:
            print(f"Getting question data for index: {question_idx}")
            question_data = self.dataset[question_idx]
            print(f"Question data type: {type(question_data)}")
            print(f"Question data attributes: {dir(question_data) if hasattr(question_data, '__dict__') else 'No attributes'}")
            
            # Build question HTML
            question_html = ""
            
            # Add question header
            if hasattr(question_data, 'key'):
                question_html += f"<h3>Question {question_data.key}</h3>"
            else:
                question_html += f"<h3>Question {question_idx + 1}</h3>"
            
            # Add context if available
            if hasattr(question_data, 'context') and question_data.context:
                question_html += f"<p><strong>Context:</strong> {question_data.context}</p>"
            
            # Add question text
            if hasattr(question_data, 'question') and question_data.question:
                question_html += f"<p><strong>Question:</strong> {question_data.question}</p>"
            elif hasattr(question_data, 'text') and question_data.text:
                question_html += f"<p><strong>Question:</strong> {question_data.text}</p>"
            
            # Add options if available (for MCQ)
            if hasattr(question_data, 'options') and question_data.options:
                question_html += "<p><strong>Options:</strong></p><ol>"
                for option in question_data.options:
                    question_html += f"<li>{option}</li>"
                question_html += "</ol>"
            elif hasattr(question_data, 'choices') and question_data.choices:
                question_html += "<p><strong>Options:</strong></p><ol>"
                for option in question_data.choices:
                    question_html += f"<li>{option}</li>"
                question_html += "</ol>"
            
            # If no content was found, try to display the raw data
            if question_html == f"<h3>Question {question_idx + 1}</h3>":
                question_html += f"<p><strong>Raw Data:</strong> {str(question_data)}</p>"
            
            # Build answer HTML
            answer_html = ""
            if hasattr(question_data, 'answer') and question_data.answer:
                answer_html += f"<p><strong>Answer:</strong> {question_data.answer}</p>"
                
                # Add explanation if available
                if hasattr(question_data, 'explanation') and question_data.explanation:
                    answer_html += f"<p><strong>Explanation:</strong> {question_data.explanation}</p>"
            elif hasattr(question_data, 'label') and question_data.label:
                answer_html += f"<p><strong>Answer:</strong> {question_data.label}</p>"
            
            print(f"Generated question HTML length: {len(question_html)}")
            print(f"Generated answer HTML length: {len(answer_html)}")
            
            return question_html, answer_html
            
        except Exception as e:
            print(f"Exception in get_question_display: {str(e)}")
            logger.exception("An exception occurred while getting question display.")
            return f"Error displaying question: {str(e)}", ""

    def get_question_image(self, question_idx: int):
        """Get image for the current question if available."""
        if not self.dataset_loaded or question_idx >= self.total_questions:
            return None
            
        try:
            question_data = self.dataset[self.current_question]
            
            # Check if images are available
            if hasattr(question_data, 'images') and question_data.images:
                if isinstance(question_data.images, (list, tuple)) and len(question_data.images) > 0:
                    return question_data.images[0]  # Return first image
                elif not isinstance(question_data.images, (list, tuple)):
                    return question_data.images
            
            return None
            
        except Exception as e:
            logger.exception("An exception occurred while getting question image.")
            return None

def create_interface():
    """Create the Gradio interface."""
    
    viewer = DataViewer()
    
    with gr.Blocks(title="Dataset Viewer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Dataset Viewer")
        gr.Markdown("View and interact with Text and Vision datasets from your project")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Dataset Selection")
                
                # Modality selection
                modality = gr.Radio(
                    choices=["Text", "Vision"],
                    label="Select Modality",
                    value="Text"
                )
                
                # Question type selection
                questions_type = gr.Radio(
                    choices=["MCQ", "OEQ"],
                    label="Select Question Type",
                    value="MCQ"
                )
                
                # Dataset selection
                dataset_name = gr.Dropdown(
                    choices=[],
                    label="Select Dataset",
                    interactive=True
                )
                
                # Subset selection
                subset = gr.Dropdown(
                    choices=[],
                    label="Select Subset",
                    interactive=True
                )
                
                # Split selection
                split = gr.Dropdown(
                    choices=["train"],
                    label="Select Split",
                    value="train",
                    interactive=True
                )
                
                # Load button
                load_btn = gr.Button("Load Dataset", variant="primary")
                
                # Status message
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=3
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## Question Viewer")
                
                # Navigation controls
                with gr.Row():
                    prev_btn = gr.Button("← Previous")
                    question_slider = gr.Slider(
                        minimum=1,
                        maximum=1,
                        value=1,
                        step=1,
                        label="Question Number",
                        interactive=True
                    )
                    next_btn = gr.Button("Next →")
                
                # Question display
                question_image = gr.Image(
                    label="Question Image",
                    visible=False
                )
                
                question_content = gr.HTML(
                    label="Question",
                    value="Please select and load a dataset to begin."
                )
                
                # Answer section
                with gr.Accordion("View Answer", open=False):
                    answer_content = gr.HTML(
                        label="Answer",
                        value="No answer available."
                    )
        
        # Event handlers
        def update_datasets(modality_val, questions_type_val):
            datasets = viewer.get_available_datasets(modality_val, questions_type_val)
            if datasets:
                return gr.update(choices=datasets, value=datasets[0] if datasets else None)
            else:
                return gr.update(choices=[], value=None)
        
        def update_subsets(modality_val, questions_type_val, dataset_name_val):
            if dataset_name_val:
                subsets = viewer.get_subsets(modality_val, questions_type_val, dataset_name_val)
                if subsets:
                    return gr.update(choices=subsets, value=subsets[0] if subsets else None)
            return gr.update(choices=[], value=None)
        
        def update_splits(modality_val, questions_type_val, dataset_name_val, subset_val):
            if dataset_name_val:
                splits = viewer.get_splits(modality_val, questions_type_val, dataset_name_val, subset_val)
                return gr.update(choices=splits, value=splits[0] if splits else "train")
            return gr.update(choices=["train"], value="train")
        
        def load_dataset_handler(modality_val, questions_type_val, dataset_name_val, subset_val, split_val):
            msg, success = viewer.load_dataset(modality_val, questions_type_val, dataset_name_val, subset_val, split_val)
            
            if success:
                # Update slider maximum and get first question
                question_html, answer_html = viewer.get_question_display(0)
                image = viewer.get_question_image(0)
                
                return (
                    msg,  # status
                    gr.update(maximum=viewer.total_questions, value=1),  # slider
                    question_html,  # question content
                    answer_html,  # answer content
                    gr.update(value=image, visible=image is not None)  # image
                )
            else:
                return (
                    msg,  # status
                    gr.update(),  # slider (no change)
                    "Failed to load dataset",  # question content
                    "",  # answer content
                    gr.update(visible=False)  # image
                )
        
        def navigate_handler(direction):
            question_num, question_html, answer_html = viewer.navigate_question(direction)
            image = viewer.get_question_image(viewer.current_question)
            
            return (
                question_num,  # slider value
                question_html,  # question content
                answer_html,  # answer content
                gr.update(value=image, visible=image is not None)  # image
            )
        
        def slider_handler(question_num):
            question_html, answer_html = viewer.set_question_index(question_num)
            image = viewer.get_question_image(viewer.current_question)
            
            return (
                question_html,  # question content
                answer_html,  # answer content
                gr.update(value=image, visible=image is not None)  # image
            )
        
        # Connect event handlers
        modality.change(
            fn=update_datasets,
            inputs=[modality, questions_type],
            outputs=[dataset_name]
        )
        
        questions_type.change(
            fn=update_datasets,
            inputs=[modality, questions_type],
            outputs=[dataset_name]
        )
        
        dataset_name.change(
            fn=update_subsets,
            inputs=[modality, questions_type, dataset_name],
            outputs=[subset]
        )
        
        subset.change(
            fn=update_splits,
            inputs=[modality, questions_type, dataset_name, subset],
            outputs=[split]
        )
        
        load_btn.click(
            fn=load_dataset_handler,
            inputs=[modality, questions_type, dataset_name, subset, split],
            outputs=[status_msg, question_slider, question_content, answer_content, question_image]
        )
        
        prev_btn.click(
            fn=lambda: navigate_handler("previous"),
            outputs=[question_slider, question_content, answer_content, question_image]
        )
        
        next_btn.click(
            fn=lambda: navigate_handler("next"),
            outputs=[question_slider, question_content, answer_content, question_image]
        )
        
        question_slider.change(
            fn=slider_handler,
            inputs=[question_slider],
            outputs=[question_content, answer_content, question_image]
        )
    
    return demo

def main():
    """Main entry point for the Gradio application."""
    logger.info("Starting the Dataset Viewer application.")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
    
    logger.info("Application finished running.")

if __name__ == "__main__":
    main()
