from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    """
    Preference Dataset for loading preference pairs.

    Each item in the dataset is a tuple containing:
    - instruction_text: The input prompt or question.
    - chosen_response: The preferred model output.
    - rejected_response: The less preferred model output.
    """

    def __init__(self, preference_data, tokenizer, max_length=4096) -> None:
        """
        Initializes the dataset with preference pairs.

        Args:
            preference_data (list): List of tuples (instruction_text, chosen_response, rejected_response).
            tokenizer: Tokenizer to convert text to tokens.
            max_length: Maximum sequence length for tokenization.
        """
        self.data = preference_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing tokenized instructions, chosen, and rejected ids.
        """
        # Unpack the preference pair
        instruction_text, chosen_response, rejected_response = self.data[idx]

        # Tokenize the input text
        instruction_ids = self.tokenizer.encode(
            instruction_text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        ).squeeze(0)

        # Tokenize the chosen response
        chosen_ids = self.tokenizer.encode(
            chosen_response,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        ).squeeze(0)

        # Tokenize the rejected response
        rejected_ids = self.tokenizer.encode(
            rejected_response,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        ).squeeze(0)

        return {
            'instruction_ids': instruction_ids,
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids
        }
