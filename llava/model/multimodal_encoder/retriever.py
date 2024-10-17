import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a learnable positional encoding parameter
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)  # You can adjust initialization strategy

    def forward(self, x):
        # Use only the necessary part of the positional encoding
        return x + self.positional_encoding[:x.size(1), :]

class Retriever(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout_rate=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True  # This configures the layer to accept input as [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Pooling layer to reduce dimensions from [batch, seq, feature] to [batch, feature]
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Output layer to convert [batch, feature] to [batch, 1] and apply sigmoid activation
        self.output_layer = nn.Linear(input_dim, 1)

        self.pos_encoder = PositionalEncoding(input_dim)
        self.text_projector = torch.nn.Linear(input_dim, input_dim)

    def forward(self, img_embed, text_embed, y=None):
        # text_embed = self.text_projector(text_emged)
        split_size = [len(x_i) for x_i in img_embed]
        img_embed = torch.cat(img_embed, dim=0)
        text_embed = torch.cat(text_embed, dim=0)
        x = torch.cat([img_embed, text_embed], dim=1)
        # add positional encoding

        x = self.pos_encoder(x)  # Add learnable positional encoding
        
        # x: [batch_size, seq_len, input_dim]
        x = self.transformer_encoder(x)  # Transformer encoder processes the data
        x = x.transpose(1, 2)  # Transpose for pooling, changes x to [batch, feature, seq]
        x = self.pooling(x).squeeze(-1)  # Pool and remove the last dimension, resulting in [batch, feature]
        logits = self.output_layer(x)  # Apply the output linear layer
        outputs = torch.sigmoid(logits)  # Apply sigmoid activation to get outputs in the range [0, 1]
        outputs = outputs.split(split_size, dim=0)
        ret = {'logits': logits, 'outputs': outputs}

        if y is not None:
            # y should have shape [batch], and be the same type as outputs
            y = torch.cat(y, dim=0).unsqueeze(1).bfloat16()
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor([5.0]).to(logits.device))
            ret['loss'] = loss

        return ret