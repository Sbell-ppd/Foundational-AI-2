import os
import torch
from rnn_module import RNNModule
from lstm_module import LSTM
from transformer_module import TransformerModel
import sentencepiece as spm

# Paths and parameters
model_save_dir = "C://Users//Sideeq Bello//CSC7809_FoundationModels//Project2//models//saved"
tokenizer_model = "C:\\Users\\Sideeq Bello\\CSC7809_FoundationModels\\Project2\\data\\tokenizer.model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(tokenizer_model)

# Initialize models
vocab_size = 10000
hidden_size = 256
embed_size = 128
num_heads = 8
num_layers = 4
max_seq_length = 50

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

# Load trained model weights
rnn_model.load_state_dict(torch.load(os.path.join(model_save_dir, "rnn_model.pth")))
lstm_model.load_state_dict(torch.load(os.path.join(model_save_dir, "lstm_model.pth")))
transformer_model.load_state_dict(torch.load(os.path.join(model_save_dir, "transformer_model.pth")))

# Generate responses
def generate_response(model, prompt, max_seq_length=50, is_rnn=False, temperature=1.0):
    """
    Generate a response for a given prompt using the specified model.
    :param model: The trained model (RNN, LSTM, or Transformer)
    :param prompt: The input prompt (string)
    :param max_seq_length: Maximum sequence length for generation
    :param is_rnn: Whether the model is an RNN/LSTM that requires hidden states
    :param temperature: Sampling temperature for randomness in generation
    :return: Generated response (string)
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the input prompt
        input_ids = torch.tensor([tokenizer.Encode(prompt, out_type=int)], device=device)
        generated_ids = input_ids.clone()

        # Initialize hidden states for RNN/LSTM
        if is_rnn:
            if hasattr(model, 'init_hidden_cell'):  # For LSTM
                hidden, cell = model.init_hidden_cell(1)  # Batch size = 1
                hidden, cell = hidden.to(device), cell.to(device)
            elif hasattr(model, 'init_hidden'):  # For RNN
                hidden = model.init_hidden(1).to(device)

        for _ in range(max_seq_length):
            # Ensure input sequence does not exceed max_seq_length
            if generated_ids.size(1) > max_seq_length:
                break

            if is_rnn:
                if hasattr(model, 'init_hidden_cell'):  # For LSTM
                    output, hidden, cell = model(generated_ids[:, -1:], hidden, cell)
                elif hasattr(model, 'init_hidden'):  # For RNN
                    output, hidden = model(generated_ids[:, -1:], hidden)
            else:
                output = model(generated_ids)

            # Apply temperature sampling
            logits = output[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            if next_token == tokenizer.eos_id:  # Stop if EOS token is generated
                break

            # Append the next token to the generated sequence
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]], device=device)], dim=1)

        # Decode the generated token IDs into text
        return tokenizer.Decode(generated_ids.squeeze(0).tolist())
# Prompts
prompts = [
    "Which do you prefer? Dogs or cats?",  # The Given prompt
    "What is your favorite programming language?"  # My Chosen prompt
]

# Generate and print responses for each model
for prompt in prompts:
    print(f"Prompt: {prompt}")
    for model, name, is_rnn in [
        (rnn_model, "RNN", True),
        (lstm_model, "LSTM", True),
        (transformer_model, "Transformer", False),
    ]:
        response = generate_response(model, prompt, max_seq_length=max_seq_length, is_rnn=is_rnn, temperature=1.0)
        print(f"{name} Response: {response}")
    print()
