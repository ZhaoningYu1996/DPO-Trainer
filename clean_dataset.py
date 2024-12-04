import json
import random
from alpaca_template import AlpacaInstructTemplate
from typing import List, Tuple

def clean_raw_data(raw_path, train_path, test_path, train_split=0.8) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Processes raw data from a JSON file, generates prompts using the AlpacaInstructTemplate,
    splits the data into training and testing sets, and saves them into separate files.

    Args:
        raw_path: Path to the input raw JSON file.
        train_path: Path to save the training data file.
        test_path: Path to save the testing data file.
        train_split: Proportion of data to use for training.

    Returns:
        train_data: The training data as a list of tuples (prompt, chosen, rejected).
        test_data: The testing data as a list of tuples (prompt, chosen, rejected).
    """

    # Read the JSON file
    with open(raw_path, "r") as f:
        data = json.load(f)

    # Generate tuples (prompt, chosen, rejected)
    data_tuples = []
    for obj in data:
        sample = {
            "instruction": obj["instruction"],
            "input": obj["input"], 
        }
        prompt = AlpacaInstructTemplate.format(sample)
        data_tuples.append((prompt, obj["chosen"], obj["rejected"]))

    # Split into training and testing sets
    random.shuffle(data_tuples)
    split_index = int(train_split * len(data_tuples))
    train_data = data_tuples[:split_index]
    test_data = data_tuples[split_index:]

    # Save to files
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=0)

    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=0)

    print(f"Data processing complete. Training data saved to {train_path}, testing data saved to {test_path}.")
    return train_data, test_data
