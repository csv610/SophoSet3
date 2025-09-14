import os
import sys
import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import streamlit as st

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

# Set page config
st.set_page_config(
    page_title="Dataset Runner",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    .success-msg {
        color: #0f9d58;
        font-weight: bold;
    }
    .error-msg {
        color: #db4437;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@dataclass
class StateParams:
    """Dataclass to manage application state parameters."""
    modality: Optional[str] = None
    questions_type: Optional[str] = None
    dataset_name: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None

class DataViewer:
    """A class to handle dataset viewing and interaction in Streamlit."""
    
    def __init__(self):
        """Initialize the DataViewer with default settings."""
        self._init_session_state()

    def run(self):
        """Run the DataViewer application."""
        st.title("🤖 Dataset Viewer")
        st.markdown("View and interact with Text and Vision datasets from your project")
        
        # Render the interface
        self.input_params_from_sidebar()
        self.render_main_content()

    def input_params_from_sidebar(self):
        """Renders the sidebar and handles user input for dataset selection."""
        with st.sidebar:
            st.header("Dataset Selection")
            
            # 1. Modality selection
            self._select_modality()
            
            # 2. Question Type selection
            self._select_question_type()
            
            # 3. Get or create dataset instance with UI selection
            self._get_selected_dataset_instance()
            
            # 4. Get and handle subset selection
            self._get_subset()
            
            # 5. Get and handle split selection
            self._get_split()
            
            # 6. Load button - always show but validation happens in _load_selected_dataset
            if st.button("Load Dataset"):
                self._load_selected_dataset()
    
    def render_main_content(self):
        """Render the main content area based on the loaded dataset."""
        if st.session_state.get('dataset_loaded') and st.session_state.dataset is not None:
            try:
                dataset = st.session_state.dataset
                questions_type = st.session_state.curr_state.questions_type
                
                if questions_type == 'MCQ':
                    self._render_mcq_interface(dataset)
                else:  # OEQ
                    self._render_oeq_interface(dataset)
                    
            except Exception as e:
                # User-facing error message
                st.error(f"Error loading dataset: {str(e)}")
                # Log the full traceback for debugging
                logger.exception("An exception occurred while rendering the main content interface.")
        else:
            st.info("Please select a dataset and click 'Load Dataset' to begin.")
    
    def _init_session_state(self):
        """Initialize the Streamlit session state variables."""
        # State management
        if 'prev_state' not in st.session_state:
            st.session_state.prev_state = StateParams()
        if 'curr_state' not in st.session_state:
            st.session_state.curr_state = StateParams()
    
    def _update_selections(self, **kwargs):
        """
        Update the current selections and track previous values.
        
        Args:
            **kwargs: Key-value pairs to update in current selections
            
        Returns:
            bool: True if any values were changed, False otherwise
        """
        changed = False
        
        # Check for changes before updating
        for key, new_value in kwargs.items():
            current_value = getattr(st.session_state.curr_state, key, None)
            if current_value != new_value:
                changed = True
                break
                
        if changed:
            # Store current state as previous state before updating
            st.session_state.prev_state = StateParams(**asdict(st.session_state.curr_state))
            
            # Update current state with new values
            for key, value in kwargs.items():
                if hasattr(st.session_state.curr_state, key):
                    setattr(st.session_state.curr_state, key, value)
            
        return changed

    def _select_modality(self):
        """
        Render radio buttons for modality selection and update the session state.
        """
        modalities = ["Text", "Vision"]
        
        # Get the current selection or default to the first option
        current_modality = st.session_state.curr_state.modality
        default_idx = modalities.index(current_modality) if current_modality in modalities else 0
        
        # Display the radio button and update session state if changed
        selected_modality = st.radio(
            "Select Modality:",
            modalities,
            index=default_idx
        )
        
        # Update the selection and check if it changed
        if self._update_selections(modality=selected_modality):
            st.session_state.curr_state.modality = selected_modality
            logger.info(f"Modality changed to: {selected_modality}")
            st.rerun()
    
    def _select_question_type(self):
        """
        Render radio buttons for question type selection and update the session state.
        """
        question_types = ["MCQ", "OEQ"]
        
        # Get the current selection or default to the first option
        current_type = st.session_state.curr_state.questions_type
        default_idx = question_types.index(current_type) if current_type in question_types else 0
        
        # Display the radio button and update session state if changed
        selected_type = st.radio(
            "Select Question Type:",
            question_types,
            index=default_idx
        )
        
        # Update the selection and check if it changed
        if self._update_selections(questions_type=selected_type):
            st.session_state.curr_state.questions_type = selected_type
            logger.info(f"Question type changed to: {selected_type}")
            st.rerun()

    def _get_available_datasets(self) -> List[str]:
        """
        Scan the project structure to find available datasets for the selected modality and questions type.
        
        Returns:
            List of available dataset names
        """
        # Get current state
        curr_state = st.session_state.curr_state
        
        # Return empty list if required fields are not set
        if not curr_state.modality or not curr_state.questions_type:
            logger.warning("Modality or question type not selected. Skipping dataset scan.")
            return []
            
        base_dir = Path(__file__).parent
        dataset_dir = base_dir / curr_state.modality / curr_state.questions_type
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return []
            
        # Look for Python files that match the pattern [dataset_name]_data.py
        dataset_files = list(dataset_dir.glob("*_data.py"))
        
        # Extract dataset names by removing '_data.py' from filenames
        datasets = [f.stem.replace('_data', '') for f in dataset_files]
        logger.info(f"Found datasets: {datasets}")
        
        return sorted(datasets)
    
    def _get_dataset_class(self):
        """Dynamically import and return the dataset class."""
        try:
            # Format the module path
            module_name = f"{st.session_state.curr_state.modality}.{st.session_state.curr_state.questions_type}.datasets.{st.session_state.curr_state.questions_type.lower()}_dataset"
            logger.info(f"Attempting to import module: {module_name}")
            
            # Import the module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                st.error(f"Could not find module: {module_name}")
                logger.error(f"Module not found: {module_name}")
                return None
                
            module = importlib.import_module(module_name)
            
            # Get the dataset class (assumes class name is like 'MCQDataset' or 'OEQDataset')
            class_name = f"{st.session_state.curr_state.questions_type}Dataset"
            if hasattr(module, class_name):
                logger.info(f"Successfully imported class: {class_name}")
                return getattr(module, class_name)
                
            st.error(f"Could not find class {class_name} in module {module_name}")
            logger.error(f"Class not found: {class_name} in module {module_name}")
            return None
                
        except Exception as e:
            st.error(f"Error importing dataset class: {str(e)}")
            logger.exception("An exception occurred while importing the dataset class.")
            return None

    def _get_dataset_instance(self):
        """
        Create and return a dataset instance without loading data.
        
        Returns:
            The initialized dataset instance
            
        Raises:
            Exception: If dataset initialization fails
        """
        try:
            # Get the dataset class
            dataset_class = self._get_dataset_class()
            
            if dataset_class is None:
                st.error(f"Could not find dataset class for {st.session_state.curr_state.modality}/{st.session_state.curr_state.questions_type}")
                return None
                
            # Create an instance of the dataset
            dataset = dataset_class()
            logger.info(f"Successfully created dataset instance for: {st.session_state.curr_state.dataset_name}")
            return dataset
            
        except Exception as e:
            st.error(f"Error creating dataset {st.session_state.curr_state.dataset_name}: {str(e)}")
            logger.exception("An exception occurred while creating the dataset instance.")
            raise

    def _get_dataset_selection(self) -> Optional[str]:
        """
        Displays a selectbox for dataset selection and returns the user's choice.
        
        Returns:
            The name of the selected dataset or None if no datasets are available.
        """
        # Get available datasets for the selected modality and type
        available_datasets = self._get_available_datasets()
        
        if not available_datasets:
            st.error(f"No {st.session_state.curr_state.questions_type} datasets found for {st.session_state.curr_state.modality} modality.")
            return None
        
        # Get the previously selected dataset if it exists and is still available
        selected_dataset = st.session_state.curr_state.dataset_name
        default_index = 0
        
        if selected_dataset in available_datasets:
            default_index = available_datasets.index(selected_dataset)
        
        # Display dataset selector
        selected_dataset = st.selectbox(
            "Select Dataset:",
            available_datasets,
            index=default_index
        )
        
        return selected_dataset

    def _get_selected_dataset_instance(self):
        """
        Handles dataset selection and initializes the dataset instance.
        """
        selected_dataset = self._get_dataset_selection()

        if selected_dataset is None:
            st.stop()
            
        # Update the selection and check if it changed
        dataset_changed = self._update_selections(dataset_name=selected_dataset)
        
        # Check if we need to reload the dataset
        if dataset_changed or 'dataset' not in st.session_state:
            with st.spinner('Initializing dataset...'):
                try:
                    dataset = self._get_dataset_instance()
                    if dataset is None:
                        st.error(f"Failed to initialize dataset: {selected_dataset}")
                        logger.error(f"Failed to initialize dataset: {selected_dataset}")
                        st.stop()
                    
                    # Persist the dataset and a flag to avoid reloads
                    st.session_state.dataset = dataset
                except Exception as e:
                    st.error(f"Error initializing dataset: {str(e)}")
                    logger.exception("An exception occurred while getting the selected dataset instance.")
                    st.stop()
            
            st.rerun()

    def _get_subset(self) -> str:
        """
        Display a subset selector and handle subset selection.
        
        Returns:
            str: The selected subset
            
        Note:
            Updates session state and triggers a rerun if the subset changes
        """
        if not hasattr(st.session_state, 'dataset') or st.session_state.dataset is None:
            logger.warning("Dataset not available. Cannot get subsets.")
            return None
            
        try:
            available_subsets = st.session_state.dataset.get_subsets()
            
            if not available_subsets:
                logger.info("No subsets available for the selected dataset.")
                return None
                
            # Find the index of the current subset if it exists in available_subsets
            current_subset = st.session_state.curr_state.subset
            default_idx = 0
            
            if current_subset in available_subsets:
                default_idx = available_subsets.index(current_subset)
                
            # Display the subset selector
            selected_subset = st.selectbox(
                "Select Subset:",
                available_subsets,
                index=default_idx
            )
            
            # Update session state if subset changed
            if selected_subset != current_subset:
                logger.info(f"Subset changed to: {selected_subset}")
                # Save current state as previous
                st.session_state.prev_state = StateParams(**asdict(st.session_state.curr_state))
                
                # Update current state
                st.session_state.curr_state.subset = selected_subset
                
                # Clear dependent state
                st.session_state.curr_state.split = None
                
                # Reset dataset loaded flag to trigger reload
                st.session_state.dataset_loaded = False
                
                st.rerun()
                
            return selected_subset
            
        except Exception as e:
            st.error(f"Error getting subsets: {str(e)}")
            logger.exception("An exception occurred while getting subsets.")
            return None
    
    def _get_split(self):
        """
        Display a split selector and handle split selection.
        
        Returns:
            str: The selected split
            
        Note:
            Updates session state and triggers a rerun if the split changes
        """
        if not hasattr(st.session_state.dataset, 'get_splits'):
            logger.warning("get_splits method not found. Defaulting to 'train' split.")
            return 'train'  # Default to 'train' if get_splits is not available
            
        try:
            available_splits = st.session_state.dataset.get_splits(subset=st.session_state.curr_state.subset)
            if not available_splits:
                logger.info("No splits available for the selected dataset and subset.")
                return 'train'  # Default to 'train' if no splits available
                
            # Find the index of the current split if it exists in available_splits
            default_idx = 0
            if st.session_state.curr_state.split in available_splits:
                default_idx = available_splits.index(st.session_state.curr_state.split)
                
            # Display the split selector
            selected_split = st.selectbox(
                "Select Split:",
                available_splits,
                index=default_idx
            )
            
            # Update session state if split changed
            if selected_split != st.session_state.curr_state.split:
                logger.info(f"Split changed to: {selected_split}")
                st.session_state.selected_split = selected_split
                st.rerun()
                
            return selected_split
            
        except Exception as e:
            st.error(f"Error getting splits: {str(e)}")
            logger.exception("An exception occurred while getting splits.")
            return 'train'  # Default to 'train' on error
    
    def _load_dataset(self, dataset_obj, subset: str = None, split: str = 'train'):
        """Load data into the provided dataset object with the specified subset and split."""
        logger.info(f"Attempting to load dataset with subset: {subset}, split: {split}")
        try:
            with st.spinner("Loading dataset..."):
                dataset_loaded = dataset_obj.load(subset=subset, split=split)
            logger.info("Dataset loaded successfully.")
            return dataset_loaded
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            logger.exception("An exception occurred while loading the dataset.")
            return None
    
    def _load_selected_dataset(self) -> bool:
        """
        Load the selected dataset based on current state.
        
        Returns:
            bool: True if the dataset was loaded successfully, False otherwise
        """
        # Get current state
        curr_state = st.session_state.curr_state
        
        # Check if all required fields are set
        required_fields = ['modality', 'questions_type', 'dataset_name']
        if not all(getattr(curr_state, field) for field in required_fields):
            st.error("Please select all required fields (modality, question type, and dataset)")
            logger.warning("Required fields for dataset loading are missing.")
            return False
            
        # Check if we already have the dataset loaded with the same parameters
        if (st.session_state.get('dataset_loaded') and 
            st.session_state.dataset is not None and 
            st.session_state.prev_state == curr_state):
            logger.info("Dataset is already loaded with the same parameters. Skipping reload.")
            return True
            
        with st.spinner('Loading dataset...'):
            try:
                # Get the dataset instance
                dataset = self._get_dataset_instance()
                if dataset is None:
                    st.error("Failed to initialize dataset")
                    logger.error("Failed to initialize dataset instance.")
                    return False
                    
                # Load the dataset with the selected subset and split
                subset = curr_state.subset
                split = curr_state.split or 'train'  # Default to 'train' if not specified
                
                # Load the data
                dataset = self._load_dataset(dataset, subset=subset, split=split)
                
                # Update session state
                st.session_state.dataset = dataset
                st.session_state.dataset_loaded = True
                
                # Update dataset type for rendering
                st.session_state.dataset_type = curr_state.questions_type
                
                logger.info("Dataset loading process completed successfully.")
                return True
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                logger.exception("An exception occurred during the dataset loading process.")
                return False

    def _render_mcq_interface(self, dataset):
        """Render interface for Multiple Choice Question datasets."""
        try:
            # Get total number of questions
            total_questions = len(dataset)
            
            if total_questions == 0:
                st.warning("No questions found in the dataset.")
                logger.warning("No questions found in the MCQ dataset.")
                return
            
            # Initialize session state for question navigation if needed
            if 'current_question' not in st.session_state:
                st.session_state.current_question = 0
            
            # Navigation controls
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("← Previous") and st.session_state.current_question > 0:
                    st.session_state.current_question -= 1
                    logger.info(f"Navigating to previous question: {st.session_state.current_question}")
                    st.rerun()
            
            with col2:
                question_num = st.slider(
                    "Question",
                    1, total_questions,
                    st.session_state.current_question + 1,
                    key="question_slider"
                )
                # Update current question if slider changed
                if question_num - 1 != st.session_state.current_question:
                    st.session_state.current_question = question_num - 1
                    logger.info(f"Navigating to question via slider: {st.session_state.current_question}")
                    st.rerun()
            
            with col3:
                if st.button("Next →") and st.session_state.current_question < total_questions - 1:
                    st.session_state.current_question += 1
                    logger.info(f"Navigating to next question: {st.session_state.current_question}")
                    st.rerun()
            
            # Display the current question
            question_idx = st.session_state.current_question
            question_data = dataset[question_idx]
            
            # Display question data
            self._display_question_data(question_data)
            
        except Exception as e:
            st.error(f"Error rendering MCQ interface: {str(e)}")
            logger.exception("An exception occurred while rendering the MCQ interface.")
    
    def _render_oeq_interface(self, dataset):
        """Render interface for Open-Ended Question datasets with navigation controls."""
        try:
            # Get total number of questions
            total_questions = len(dataset)
            
            if total_questions == 0:
                st.warning("No questions found in the dataset.")
                logger.warning("No questions found in the OEQ dataset.")
                return
            
            # Initialize session state for question navigation if needed
            if 'current_question' not in st.session_state:
                st.session_state.current_question = 0
            
            # Navigation controls
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("← Previous") and st.session_state.current_question > 0:
                    st.session_state.current_question -= 1
                    logger.info(f"Navigating to previous question: {st.session_state.current_question}")
                    st.rerun()
            
            with col2:
                question_num = st.slider(
                    "Question",
                    1, total_questions,
                    st.session_state.current_question + 1,
                    key="question_slider"
                )
                # Update current question if slider changed
                if question_num - 1 != st.session_state.current_question:
                    st.session_state.current_question = question_num - 1
                    logger.info(f"Navigating to question via slider: {st.session_state.current_question}")
                    st.rerun()
            
            with col3:
                if st.button("Next →") and st.session_state.current_question < total_questions - 1:
                    st.session_state.current_question += 1
                    logger.info(f"Navigating to next question: {st.session_state.current_question}")
                    st.rerun()
            
            # Display the current question
            question_idx = st.session_state.current_question
            question_data = dataset[question_idx]
            
            # Display question data
            self._display_question_data(question_data)
            
            # Display additional metadata if available
            if hasattr(question_data, 'metadata'):
                with st.expander("View Metadata"):
                    st.json(question_data.metadata)
            
        except Exception as e:
            st.error(f"Error rendering OEQ interface: {str(e)}")
            logger.exception("An exception occurred while rendering the OEQ interface.")

    def _display_image(self, image_data, caption=None):
        """
        Display an image from various input formats.
        
        Args:
            image_data: Image data (bytes, file path, or URL)
            caption: Optional caption for the image
        """
        try:
            if image_data is None:
                st.warning("No image data provided")
                logger.warning("Attempted to display a None image.")
                return
                
            if isinstance(image_data, bytes):
                # Display from bytes
                st.image(image_data, caption=caption, use_column_width=True)
            elif isinstance(image_data, str):
                if image_data.startswith(('http://', 'https://')):
                    # Display from URL
                    st.image(image_data, caption=caption, use_column_width=True)
                else:
                    # Display from file path
                    if os.path.exists(image_data):
                        st.image(image_data, caption=caption, use_column_width=True)
                    else:
                        st.warning(f"Image file not found: {image_data}")
                        logger.warning(f"Image file not found at path: {image_data}")
            else:
                st.warning(f"Unsupported image data type: {type(image_data)}")
                logger.warning(f"Unsupported image data type: {type(image_data)}")
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            logger.exception("An exception occurred while displaying an image.")
    
    def _display_question_metadata(self, context, question):
        """Display question metadata (context and question text)."""
        if context:
            st.markdown(f"**Context:** {context}")
        if question:
            st.markdown(f"**Question:** {question}")
    
    def _display_options(self, options):
        """Display question options if available."""
        if options and len(options) > 0:
            st.markdown("**Options:**")
            for i, option in enumerate(options, 1):
                st.markdown(f"{i}. {option}")
    
    def _display_question_data(self, row_data):
        """
        Display question data including context, question, image, options, and answer.
        
        Args:
            row_data: An object containing question data with attributes like key, context, 
                      question, options, answer, explanation, and images.
        """
        if hasattr(row_data, 'key'):
            st.subheader(f"Question {row_data.key}")
        
        # Display images if available
        if hasattr(row_data, 'images') and row_data.images:
            if isinstance(row_data.images, (list, tuple)):
                for img in row_data.images:
                    self._display_image(img)
            else:
                self._display_image(row_data.images)
        
        # Display context and question
        context = getattr(row_data, 'context', '')
        question = getattr(row_data, 'question', '')
        self._display_question_metadata(context, question)
        
        # Display options if available
        if hasattr(row_data, 'options'):
            self._display_options(row_data.options)
        
        # Display answer if available
        if hasattr(row_data, 'answer'):
            answer = row_data.answer
            
            with st.expander("View Answer"):
                st.markdown(f"**Answer:** {answer}")
                
                # Display explanation if available
                if hasattr(row_data, 'explanation') and row_data.explanation:
                    st.markdown(f"**Explanation:** {row_data.explanation}")

def main():
    """Main entry point for the Streamlit application."""
    logger.info("Starting the Dataset Viewer application.")
    # Initialize and run the DataViewer
    viewer = DataViewer()
    viewer.run()
    logger.info("Application finished running.")


if __name__ == "__main__":
    main()

