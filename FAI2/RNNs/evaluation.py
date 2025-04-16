import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformer_module import TransformerModel
import torch.nn as nn
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence


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


def compute_perplexity(model, data_loader, criterion, device, is_rnn=False):
    """
    Compute perplexity (PPL) on the test dataset.
    :param model: Trained model
    :param data_loader: DataLoader for the test dataset
    :param criterion: Loss function (e.g., CrossEntropyLoss)
    :param device: Device to run the evaluation (CPU or GPU)
    :param is_rnn: Whether the model is an RNN/LSTM that requires a hidden state
    :return: Perplexity (PPL)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Evaluating Perplexity"):
            x, y = x.to(device), y.to(device)

            if is_rnn:
                # Initialize hidden state (and cell state for LSTM)
                if hasattr(model, 'init_hidden_cell'):  # For LSTM
                    hidden, cell = model.init_hidden_cell(x.size(0))
                    hidden, cell = hidden.to(device), cell.to(device)
                    output, _, _ = model(x, hidden, cell)
                elif hasattr(model, 'init_hidden'):  # For RNN
                    hidden = model.init_hidden(x.size(0)).to(device)
                    output, _ = model(x, hidden)
                else:
                    raise AttributeError("Model does not have a method to initialize hidden state.")
            else:
                output = model(x)

            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def compute_bleu_score(model, data_loader, tokenizer, device, max_seq_length=50, is_rnn=False):
    """
    Compute BLEU score for the model's generated text.
    :param model: Trained model
    :param data_loader: DataLoader for the test dataset
    :param tokenizer: Tokenizer to decode token IDs into text
    :param device: Device to run the evaluation (CPU or GPU)
    :param max_seq_length: Maximum sequence length for generation
    :param is_rnn: Whether the model is an RNN/LSTM that requires a hidden state
    :return: Average BLEU score
    """
    model.eval()
    bleu_scores = []

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Evaluating BLEU Score"):
            x, y = x.to(device), y.to(device)

            # Generate text using the model
            generated_tokens = []
            for i in range(x.size(0)):  # Iterate over the batch
                input_seq = x[i].unsqueeze(0)  # Add batch dimension
                generated_seq = []

                # Initialize hidden state (and cell state for LSTM)
                if is_rnn:
                    if hasattr(model, 'init_hidden_cell'):  # For LSTM
                        hidden, cell = model.init_hidden_cell(1)  # Batch size = 1
                        hidden, cell = hidden.to(device), cell.to(device)
                    elif hasattr(model, 'init_hidden'):  # For RNN
                        hidden = model.init_hidden(1).to(device)
                    else:
                        raise AttributeError("Model does not have a method to initialize hidden state.")

                for _ in range(max_seq_length):
                    # Ensure input sequence does not exceed max_seq_length
                    if input_seq.size(1) > max_seq_length:
                        break

                    if is_rnn:
                        if hasattr(model, 'init_hidden_cell'):  # For LSTM
                            output, hidden, cell = model(input_seq, hidden, cell)
                        elif hasattr(model, 'init_hidden'):  # For RNN
                            output, hidden = model(input_seq, hidden)
                    else:
                        output = model(input_seq)

                    next_token = output[:, -1, :].argmax(dim=-1).item()
                    if next_token == tokenizer.eos_id:  # Stop if EOS token is generated
                        break
                    generated_seq.append(next_token)
                    input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)

                generated_tokens.append(generated_seq)

            # Compute BLEU score for each sequence
            for i in range(len(generated_tokens)):
                reference = [tokenizer.decode(y[i].tolist()).split()]
                candidate = tokenizer.decode(generated_tokens[i]).split()
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score

