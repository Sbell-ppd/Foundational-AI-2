import os
from prettytable import PrettyTable
from evaluation import compute_perplexity, compute_bleu_score
from rnn_module import RNNModule
from lstm_module import LSTM
from transformer_module import TransformerModel
from data_processing import TextDataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import sentencepiece as spm
import json

# Paths and parameters
processed_data_dir = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\processed"
model_save_dir = "C://Users//Sideeq Bello//CSC7809_FoundationModels//Project2//models//saved"
tokenizer_model = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\tokenizer.model"
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset from JSONL file
test_file = os.path.join(processed_data_dir, "test.jsonl")
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = [json.loads(line.strip()) for line in f]

# Create test dataset and DataLoader
test_dataset = TextDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(tokenizer_model)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Initialize models
rnn_model = RNNModule(vocab_size=10000, hidden_size=256, output_size=10000).to(device)
lstm_model = LSTM(vocab_size=10000, hidden_size=256, output_size=10000, embedding_dim=128).to(device)
transformer_model = TransformerModel(
    vocab_size=10000,
    embed_size=128,
    num_heads=8,
    num_layers=4,
    hidden_dim=256,
    max_seq_length=50
).to(device)

# Load trained model weights
rnn_model.load_state_dict(torch.load(os.path.join(model_save_dir, "rnn_model.pth")))
lstm_model.load_state_dict(torch.load(os.path.join(model_save_dir, "lstm_model.pth")))
transformer_model.load_state_dict(torch.load(os.path.join(model_save_dir, "transformer_model.pth")))

# Compute metrics
metrics = []
for model, name, is_rnn in [
    (rnn_model, "RNN", True),
    (lstm_model, "LSTM", True),
    (transformer_model, "Transformer", False),
]:
    perplexity = compute_perplexity(model, test_loader, criterion, device, is_rnn=is_rnn)
    bleu_score = compute_bleu_score(model, test_loader, tokenizer, device, is_rnn=is_rnn)
    metrics.append((name, perplexity, bleu_score))

# Display metrics in a table
table = PrettyTable(["Model", "Perplexity (PPL)", "BLEU Score"])
for name, perplexity, bleu_score in metrics:
    table.add_row([name, f"{perplexity:.4f}", f"{bleu_score:.4f}"])
print(table)