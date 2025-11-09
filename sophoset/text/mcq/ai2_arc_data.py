from typing import Dict, Any, Optional, List

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class Ai2ArcDataset(BaseHFDataset):
    """A class to handle loading and managing the AI2 ARC dataset without UI dependencies."""
    
    DATASET_NAME = "allenai/ai2_arc"
    
    def __init__(self):
        """Initialize the AI2 ARC dataset handler."""
        super().__init__(self.DATASET_NAME)
        
    def extract_row_data(self, row: Dict[str, Any], index: int) -> 'QAData':
        """Extract and format data from an AI2 ARC dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', '')
        choices = row.get('choices', {})
        options_list = choices.get('text', []) if isinstance(choices, dict) else []
        answer  = row.get('answerKey', '')

        # Format options as dict with letter keys (A, B, C, D, E, etc.)
        formatted_options = {}
        if options_list:
            letters = [chr(65 + i) for i in range(26)]
            for i, opt in enumerate(options_list):
                if i < len(letters):
                    formatted_options[letters[i]] = opt

        return QAData(
            key=self.get_key(index),
            question=question,
            options=formatted_options,
            answer=answer
        )

if __name__ == "__main__":
    dset = Ai2ArcDataset()
#   DatasetExporter.save(dset, format='lmdb', output_dir='../../../datasets')
    explorer = DatasetExplorer(dset)

    # Print each question using the explorer's print_question method
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)



