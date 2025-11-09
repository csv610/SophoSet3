from typing import Dict, Any, List, Optional
import re

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class MathLightEvalDataset(BaseHFDataset):
    """A class to handle loading and managing the MATH_lighteval dataset."""
    
    DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval"

    def __init__(self):
        """Initialize the base class ."""
        super().__init__(self.DATASET_NAME)

    def extract_boxed_value(self, text):
        """
        Extracts the value inside a LaTeX \boxed{} command.

        Args:
            text (str): The input string containing the LaTeX code.

        Returns:
            str or None: The extracted value if found, otherwise None.
        """
        # The regular expression looks for the word "boxed" followed by curly braces
        # The content inside the curly braces is captured using parentheses
        pattern = r"\\boxed{(.*?)}"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from the dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and solution
        question = row.get('problem', '')
        explanation   = row.get('solution', '')
        answer  = self.extract_boxed_value(explanation)
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,   
            explanation=explanation
        )

if __name__ == "__main__":
    dset = MathLightEvalDataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
    # DatasetExporter.save(dset, format='lmdb', output_dir='../../../../datasets')
