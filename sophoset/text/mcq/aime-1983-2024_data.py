from typing import Dict, Any, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class AimeDataset(BaseHFDataset):
    DATASET_NAME = "gneubig/aime-1983-2024"
    
    def __init__(self):
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from an AIME dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        # Extract question text
        question = row.get('Question', '')

        # Extract answer if it exists
        answer = row.get('Answer', '')

        # Extract options if they exist
        options_raw = row.get('Options', row.get('options', None))

        # Format options as dict with letter keys
        formatted_options = {}
        if options_raw:
            if isinstance(options_raw, dict):
                # If options is already a dict, use it
                formatted_options = options_raw
            elif isinstance(options_raw, list):
                # If options is a list, format with letter labels (A to Z)
                letters = [chr(65 + i) for i in range(26)]
                for i, opt in enumerate(options_raw):
                    if i < len(letters):
                        formatted_options[letters[i]] = opt

        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            options=formatted_options
        )

if __name__ == "__main__":
    dset = AimeDataset()
#   DatasetExporter.save(dset, format='lmdb', output_dir='../../../datasets')
    explorer = DatasetExplorer(dset)

    # Print each question using the explorer's print_question method
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)


