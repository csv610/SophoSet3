from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class MedicalConceptsQADataset(BaseHFDataset):
    """A class to handle loading and managing the MedicalConceptsQA dataset."""
    
    DATASET_NAME = "ofir408/MedConceptsQA"
    
    def __init__(self):
        """Initialize the MedicalConceptsQA dataset handler."""
        super().__init__(self.DATASET_NAME)
        
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MedicalConceptsQA dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and clean it
        question = row.get('question', '')

        # Extract options
        options_list = []
        for i in range(1, 6):  # Assuming there are up to 5 options (A-E)
            option = row.get(f'option_{i}', '')
            if option:
                options_list.append(option)

        # Extract answer
        answer = row.get('answer_id', '')

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
    dset = MedicalConceptsQADataset()

    # DatasetExporter.save(dset, format='lmdb', output_dir='../../../datasets')
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
