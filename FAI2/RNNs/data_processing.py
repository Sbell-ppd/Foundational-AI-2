import os
import json
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch


class TextDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for tokenized text data.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Ensure tensors are of type LongTensor
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)


def combine_text_files(input_dir, output_file):
    """
    Combine all .txt files in the input directory into a single file.
    :param input_dir: Directory containing raw .txt files
    :param output_file: Path to the combined output file
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read() + '\n')
    print(f"Combined text files into {output_file}")


def train_tokenizer(combined_file, model_prefix, vocab_size=10000):
    """
    Train a SentencePiece tokenizer using the combined text file.
    :param combined_file: Path to the combined text file
    :param model_prefix: Prefix for the tokenizer model files
    :param vocab_size: Vocabulary size for the tokenizer
    """
    spm.SentencePieceTrainer.Train(
        input=combined_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        bos_id=0,  # Beginning of sequence token
        eos_id=1,  # End of sequence token
        unk_id=2,  # Unknown token
        pad_id=3,  # Padding token
        user_defined_symbols=["<bos>", "<eos>", "<pad>"]  # Custom symbols
    )
    print(f"Trained tokenizer saved as {model_prefix}.model")


def tokenize_dataset(tokenizer_model, combined_file, output_dir, test_size=0.2):
    """
    Tokenize the dataset and split it into training and testing sets.
    Save the tokenized datasets in JSONL format.
    :param tokenizer_model: Path to the trained SentencePiece tokenizer model
    :param combined_file: Path to the combined text file
    :param output_dir: Directory to save tokenized datasets
    :param test_size: Proportion of the dataset to include in the test split
    """
    # Load the tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)

    # Read the combined text file
    with open(combined_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Tokenize each line
    tokenized_lines = []
    for line in lines:
        if line.strip():
            try:
                tokens = sp.Encode(line.strip(), out_type=int)
                if len(tokens) > 0:  # Ensure non-empty tokenized lines
                    tokenized_lines.append(tokens)
            except Exception as e:
                print(f"Error tokenizing line: {line.strip()} - {e}")

    # Check if tokenized_lines is empty
    if len(tokenized_lines) == 0:
        raise ValueError("No valid tokenized data found. Please check the input file and tokenizer.")

    # Split into training and testing sets
    train_data, test_data = train_test_split(tokenized_lines, test_size=test_size, random_state=42)

    # Save tokenized datasets in JSONL format
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'train.jsonl')
    test_file = os.path.join(output_dir, 'test.jsonl')

    with open(train_file, 'w', encoding='utf-8') as f:
        for sequence in train_data:
            f.write(json.dumps(sequence) + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for sequence in test_data:
            f.write(json.dumps(sequence) + '\n')

    print(f"Tokenized datasets saved as {train_file} and {test_file}")


def collate_fn(batch):
    """
    Custom collate function to pad and truncate sequences in a batch.
    :param batch: List of (input_sequence, target_sequence) tuples
    :return: Padded and truncated input and target tensors
    """
    inputs, targets = zip(*batch)
    max_seq_length = 50  # Ensure sequences do not exceed max_seq_length
    inputs_truncated = [seq[:max_seq_length] for seq in inputs]
    targets_truncated = [seq[:max_seq_length] for seq in targets]
    inputs_padded = pad_sequence(inputs_truncated, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets_truncated, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded


if __name__ == "__main__":
    # Paths and parameters
    raw_data_dir = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\raw"  # Directory containing raw .txt files
    combined_file = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\combined.txt"  # Combined text file
    tokenizer_model_prefix = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\tokenizer"  # Prefix for tokenizer model files
    processed_data_dir = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\processed"  # Directory to save processed datasets
    vocab_size = 10000  # Vocabulary size for the tokenizer

    # Step 1: Combine all .txt files into a single file
    combine_text_files(raw_data_dir, combined_file)

    # Step 2: Train the SentencePiece tokenizer
    train_tokenizer(combined_file, tokenizer_model_prefix, vocab_size)

    # Step 3: Tokenize the dataset and split into train/test sets
    tokenizer_model = f"{tokenizer_model_prefix}.model"
    tokenize_dataset(tokenizer_model, combined_file, processed_data_dir)