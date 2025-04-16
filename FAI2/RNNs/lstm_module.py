import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, embedding_dim=128):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Input, Forget, Cell, and Output gate weight matrices
        self.Wxi = nn.Linear(embedding_dim, hidden_size)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxf = nn.Linear(embedding_dim, hidden_size)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxc = nn.Linear(embedding_dim, hidden_size)
        self.Whc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxo = nn.Linear(embedding_dim, hidden_size)
        self.Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # Final output layer
        self.Why = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state, cell_state):
        """
        Compute the hidden state, memory cell internal state, and output of the LSTM module.
        :param x: Input token IDs (batch_size, seq_length)
        :param hidden_state: Previous hidden state (batch_size, hidden_size)
        :param cell_state: Previous cell state (batch_size, hidden_size)
        :return: Output, new hidden state, new cell state
        """
        # Pass input through embedding layer
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # Ensure input is of type float
        x = x.float()

        outputs = []
        seq_length = x.size(1)

        for t in range(seq_length):
            x_t = x[:, t, :]  # Extract the embedding for the current timestep (batch_size, embedding_dim)

            # Compute gate outputs
            input_gate = self.sigmoid(self.Wxi(x_t) + self.Whi(hidden_state))
            forget_gate = self.sigmoid(self.Wxf(x_t) + self.Whf(hidden_state))
            output_gate = self.sigmoid(self.Wxo(x_t) + self.Who(hidden_state))
            cell_node = self.tanh(self.Wxc(x_t) + self.Whc(hidden_state))

            # Update cell state
            cell_state = forget_gate * cell_state + input_gate * cell_node

            # Update hidden state
            hidden_state = output_gate * self.tanh(cell_state)

            # Compute final output
            output = self.Why(hidden_state)
            outputs.append(output)

        # Stack outputs to form the output sequence (batch_size, seq_length, output_size)
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden_state, cell_state

    def init_hidden_cell(self, batch_size):
        """
        Initializes hidden and cell states to zeros.
        :param batch_size: Number of samples in the batch
        :return: Initial hidden state, Initial cell state (both of shape: batch_size x hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)