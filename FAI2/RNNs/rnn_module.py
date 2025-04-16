import torch
import torch.nn as nn

class RNNModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer to convert token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Setup the weights
        self.Wxh = nn.Linear(hidden_size, hidden_size)  # Connect the input to the hidden state
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)  # Connect the hidden state to itself
        self.Who = nn.Linear(hidden_size, output_size)  # Connect the hidden state to the output

        # Activation function
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        """
        Predict the hidden state and output.
        :param x: Input token IDs (batch_size, seq_length)
        :param hidden: Previous hidden state (batch_size, hidden_size)
        :return: Output and new hidden state
        """
        # Convert token IDs to embeddings
        x = self.embedding(x)  # (batch_size, seq_length, hidden_size)

        outputs = []
        for t in range(x.size(1)):  # Iterate over the sequence length
            x_t = x[:, t, :]  # Get the embedding for the current timestep (batch_size, hidden_size)
            hidden = self.tanh(self.Wxh(x_t) + self.Whh(hidden))  # Recurrent computation
            output = self.Who(hidden)  # Compute output
            outputs.append(output)

        # Stack outputs to form the output sequence (batch_size, seq_length, output_size)
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state with zeros.
        :param batch_size: Number of samples in the batch
        :return: Initial hidden state (batch_size, hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size)