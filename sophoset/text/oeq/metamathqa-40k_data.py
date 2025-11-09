from typing import Dict, Any, Optional
import re

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class MetaMathQA40KDataset(BaseHFDataset):
    """A class to handle loading and managing the MetaMathQA-40K dataset."""
    
    DATASET_NAME = "meta-math/MetaMathQA-40K"
    
    def __init__(self):
        """Initialize the MetaMathQA-40K dataset handler."""
        super().__init__(self.DATASET_NAME)

    def extract_final_answer(self, text: str) -> str | None:
        """
        Extracts the complete string after 'The answer is: ' in the text.

        Args:
            text: The input string containing the final answer.

        Returns:
            The extracted answer string after 'The answer is: ', or None if not found.
        """
        # Find the position of 'The answer is: ' in the text
        prefix = "The answer is: "
        start_idx = text.find(prefix)
        
        # If the prefix is found, return everything after it
        if start_idx != -1:
            return text[start_idx + len(prefix):].strip()
        
        # Return None if the prefix is not found
        return None
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MetaMathQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and answer
        question = row.get('query', '')
        explanation = row.get('response', '')
        answer = self.extract_final_answer(explanation)
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            explanation=explanation
        )

if __name__ == "__main__":
    dset = MetaMathQA40KDataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
    # DatasetExporter.save(dset, format='lmdb', output_dir='../../../../datasets')
