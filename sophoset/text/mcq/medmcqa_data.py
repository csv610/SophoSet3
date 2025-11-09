from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class MedMCQADataset(BaseHFDataset):
    """A class to handle loading and managing the MedMCQA dataset."""
    
    DATASET_NAME = "openlifescienceai/medmcqa"
    
    def __init__(self):
        """Initialize the MedMCQA dataset handler."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MedMCQA dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', '')

        options_list = [row.get('opa', ''), row.get('opb', ''), row.get('opc', ''), row.get('opd', '')]

        # Get correct answer (0-3 maps to A-D)
        correct_idx = row.get('cop', -1)
        answer = chr(65 + correct_idx) if 0 <= correct_idx <= 3 else '?'

        # Format options as dict with letter keys (A, B, C, D, E, etc.)
        formatted_options = {}
        if options_list:
            letters = [chr(65 + i) for i in range(26)]
            for i, opt in enumerate(options_list):
                if i < len(letters):
                    formatted_options[letters[i]] = opt

        return QAData(
            question=question,
            options=formatted_options,
            answer=answer,
            key=self.get_key(index)
        )


if __name__ == "__main__":
    dset = MedMCQADataset()

    # DatasetExporter.save(dset, format='lmdb', output_dir='../../../datasets')
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
