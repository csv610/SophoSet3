from typing import Dict, Any, List, Optional, ClassVar
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class BlinkDataset(BaseHFDataset):
    """A class to handle loading and interacting with the BLINK dataset."""

    DATASET_NAME = "BLINK-Benchmark/BLINK"

    def __init__(self):
        """
        Initialize the dataset handler.

        """
        super().__init__(self.DATASET_NAME)

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """
        Extract and format data from a dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and options
        question = row.get('question', '')
        options_list = row.get('options', [])

        # Extract raw images (context and candidate images)
        images = []
        if 'context' in row and 'image' in row['context']:
            images.append(row['context']['image'])
        if 'candidate' in row and 'image' in row['candidate']:
            images.append(row['candidate']['image'])

        # Get the correct answer
        answer = row.get('answer', "")

        # Format options to dict with letter keys
        formatted_options = self.get_formatted_options(options_list)

        return QAData(
            key=self.get_key(index),
            question=question,
            options=formatted_options,
            images=images,
            answer=answer
        )

if __name__ == "__main__":
    dset = BlinkDataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
