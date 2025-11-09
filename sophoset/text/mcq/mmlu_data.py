"""MMLU Dataset handler for Hugging Face Hub.

This module provides a dataset handler for the MMLU (Massive Multitask Language Understanding)
dataset from Hugging Face Hub.
"""

from typing import Any, Dict

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData


class MMLUDataset(BaseHFDataset):
    """Handler for MMLU dataset from Hugging Face Hub."""

    DATASET_NAME = "cais/mmlu"

    def __init__(self):
        """Initialize the MMLU dataset handler."""
        super().__init__(self.DATASET_NAME)

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from an MMLU dataset row.

        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset

        Returns:
            QAData object containing the formatted row data
        """
        # Extract question, options, and answer
        question = row.get('question', '')
        options_list = row.get('choices', [])
        answer = row.get('answer', '')

        # Convert numeric answer to letter
        try:
            answer = chr(ord('A') + int(answer))
        except (ValueError, TypeError):
            answer = "NA"

        # Format options using base class method
        formatted_options = self.get_formatted_options(options_list)

        return QAData(
            key=self.get_key(index),
            question=question,
            options=formatted_options,
            answer=answer,
        )


if __name__ == "__main__":
    dset = MMLUDataset()
    # ROOT_DIR = "../../../datasets"
    # DatasetExporter.save(dset, format='lmdb', output_dir=ROOT_DIR)
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
