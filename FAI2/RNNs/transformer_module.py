import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, max_seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length

        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # Set to True for batch-first input
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to predict the next token
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Forward pass for the Transformer model.
        :param x: Input token IDs (batch_size, seq_length)
        :param src_mask: Source mask for attention (optional)
        :return: Logits for the next token prediction (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = x.size()

        # Ensure the input sequence length does not exceed max_seq_length
        if seq_length > self.max_seq_length:
            raise ValueError(f"Input sequence length ({seq_length}) exceeds max_seq_length ({self.max_seq_length}).")

        # Token embedding and positional encoding
        token_embeddings = self.token_embedding(x)  # (batch_size, seq_length, embed_size)
        position_embeddings = self.positional_encoding[:, :seq_length, :]  # Dynamically adjust positional encoding
        embeddings = self.dropout(token_embeddings + position_embeddings)

        # Transformer encoder
        transformer_output = self.transformer_encoder(embeddings.permute(1, 0, 2), src_mask)  # (seq_length, batch_size, embed_size)

        # Output layer
        logits = self.fc_out(transformer_output.permute(1, 0, 2))  # (batch_size, seq_length, vocab_size)
        return logits