# Import libraries
import torch
from torch import nn

import numpy as np

from ..components.layers import FnetCNNEncoder, Decoder

class UFnetCNNModel(nn.Module):
    def __init__(self,
                 model_dim: int,
                 extend_dim: int,
                 num_layers: int,
                 document_vocab_size: int,
                 max_document_sentences: int,
                 doc_sequence_len: int,
                 num_heads: int,
                 summary_vocab_size: int,
                 max_summary_sentences: int,
                 summary_sequence_len: int,
                 fnet_cnn_kernel_size: int = 3,
                 dropout: float = 0.5) -> nn.Module:
        super().__init__()

        # Create the FnetCNNEncoder
        self.encoder = FnetCNNEncoder(
            num_layers=num_layers,
            model_dim=model_dim,
            extend_dim=extend_dim,
            num_word_embeddings=document_vocab_size,
            num_type_embeddings=max_document_sentences,
            sequence_len=doc_sequence_len,
            fnet_cnn_kernel_size=fnet_cnn_kernel_size,
            dropout=dropout
        )
        
        # Create the Decoder
        self.decoder = Decoder(num_layers=num_layers,
                               model_dim=model_dim,
                               num_heads=num_heads,
                               extend_dim=extend_dim,
                               num_word_embeddings=summary_vocab_size,
                               num_type_embeddings=max_summary_sentences,
                               sequence_len=summary_sequence_len,
                               dropout=dropout)

        # Create the output layer
        self.output_layer = nn.Linear(in_features=model_dim,
                                      out_features=summary_vocab_size)

    def forward(self,
                doc_tokens: torch.tensor,
                doc_token_types: torch.tensor,
                sum_tokens: torch.tensor,
                sum_token_types: torch.tensor) -> torch.tensor:
        """Forward pass of the FnetCNNTransformer model.

        Args:
            document_tokens (torch.tensor): Tokens of the input documents in
            shape: [batch_size, doc_num_tokens]
            document_token_types (torch.tensor): Token types of the input
            documents in shape: [batch_size, doc_num_tokens]
            summary_tokens (torch.tensor): Tokens of the input summaries in
            shape: [batch_size, sum_num_tokens]
            summary_token_types (torch.tensor): Token types of the input
            summaries in shape: [batch_size, sum_num_tokens]

        Returns:
            torch.tensor: Model output
        """
        # Pass documents through our encoder
        encoder_output = self.encoder(tokens=doc_tokens,
                                      token_types=doc_token_types,
                                      return_output_list=True)
        
        # Pass summaries and encoder_outputs to our decoder
        attn_mask = self.get_attn_mask(sum_tokens.shape[-1],
                                       sum_tokens.device)
        decoder_outputs = self.decoder(sum_tokens,
                                       sum_token_types,
                                       encoder_output,
                                       attn_mask)
        
        # Generate our predictions.
        return self.output_layer(decoder_outputs)
    
    @staticmethod
    def get_attn_mask(summary_tokens_shape, device) -> torch.tensor:
        attn_mask = np.triu(
            m=np.ones((summary_tokens_shape,
                       summary_tokens_shape), dtype=bool),
            k=1
        )
        return torch.tensor(attn_mask).to(device)
