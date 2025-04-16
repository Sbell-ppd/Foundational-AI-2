import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing import TextDataset, collate_fn
from transformer_module import TransformerModel
from rnn_module import RNNModule
from lstm_module import LSTM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\RNNs")))
import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



def train_model(model, train_loader, val_loader, device, num_epochs, optimizer, scheduler, criterion, model_save_path, is_rnn=False):
    """
    Train the given model and save training/validation losses.
    """
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 5  # Early stopping patience

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if is_rnn:
                # Initialize hidden and cell states for RNN/LSTM
                if hasattr(model, 'init_hidden_cell'):  # For LSTM
                    hidden, cell = model.init_hidden_cell(x.size(0))
                    hidden, cell = hidden.to(device), cell.to(device)
                    output, hidden, cell = model(x, hidden, cell)
                elif hasattr(model, 'init_hidden'):  # For RNN
                    hidden = model.init_hidden(x.size(0)).to(device)
                    output, hidden = model(x, hidden)
                else:
                    raise AttributeError("Model does not have a method to initialize hidden state.")
            else:
                output = model(x)

            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                x, y = x.to(device), y.to(device)

                if is_rnn:
                    if hasattr(model, 'init_hidden_cell'):  # For LSTM
                        hidden, cell = model.init_hidden_cell(x.size(0))
                        hidden, cell = hidden.to(device), cell.to(device)
                        output, hidden, cell = model(x, hidden, cell)
                    elif hasattr(model, 'init_hidden'):  # For RNN
                        hidden = model.init_hidden(x.size(0)).to(device)
                        output, hidden = model(x, hidden)
                else:
                    output = model(x)

                loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        # Step the scheduler
        scheduler.step(val_loss)

    return train_losses, val_losses


if __name__ == "__main__":
    # Paths and parameters
    processed_data_dir = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\processed"
    model_save_dir = "C://Users//Sideeq Bello//CSC7809_FoundationModels//Project2//models//saved"
    os.makedirs(model_save_dir, exist_ok=True)

    batch_size = 128
    embed_size = 128
    hidden_size = 256
    num_heads = 8
    num_layers = 4
    vocab_size = 10000
    max_seq_length = 50
    num_epochs = 30
    learning_rate = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenized datasets from JSONL files
    train_file = os.path.join(processed_data_dir, "train.jsonl")
    test_file = os.path.join(processed_data_dir, "test.jsonl")

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line.strip()) for line in f]

    with open(test_file, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line.strip()) for line in f]

    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Initialize models
    rnn_model = RNNModule(vocab_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size).to(device)
    lstm_model = LSTM(vocab_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size, embedding_dim=embed_size).to(device)
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_size,
        max_seq_length=max_seq_length
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.AdamW(rnn_model.parameters(), lr=learning_rate)
    optimizer_lstm = optim.AdamW(lstm_model.parameters(), lr=learning_rate)
    optimizer_transformer = optim.AdamW(transformer_model.parameters(), lr=learning_rate)

    # Define learning rate scheduler
    scheduler_rnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_rnn, mode='min', patience=2, factor=0.5)
    scheduler_lstm = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode='min', patience=2, factor=0.5)
    scheduler_transformer = optim.lr_scheduler.ReduceLROnPlateau(optimizer_transformer, mode='min', patience=2, factor=0.5)

    # Train models and save losses
    losses = {}

    print("Training RNN model...")
    losses["rnn_train_losses"], losses["rnn_val_losses"] = train_model(
        rnn_model, train_loader, val_loader, device, num_epochs, optimizer_rnn, scheduler_rnn, criterion,
        os.path.join(model_save_dir, "rnn_model.pth"), is_rnn=True
    )

    print("Training LSTM model...")
    losses["lstm_train_losses"], losses["lstm_val_losses"] = train_model(
        lstm_model, train_loader, val_loader, device, num_epochs, optimizer_lstm, scheduler_lstm, criterion,
        os.path.join(model_save_dir, "lstm_model.pth"), is_rnn=True
    )

    print("Training Transformer model...")
    losses["transformer_train_losses"], losses["transformer_val_losses"] = train_model(
        transformer_model, train_loader, val_loader, device, num_epochs, optimizer_transformer, scheduler_transformer, criterion,
        os.path.join(model_save_dir, "transformer_model.pth"), is_rnn=False
    )

    # Save losses to a file
    torch.save(losses, os.path.join(model_save_dir, "losses.pth"))
    print("Losses saved to losses.pth")