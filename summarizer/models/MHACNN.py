# Import libraries
import torch
from torch import nn

import numpy as np

from ..components.layers import MHACNNEncoder
from ..components.blocks import TripleEmbeddingBlock

class MHACNNModel(nn.Module):
    def __init__(self,
                 model_dim: int,
                 extend_dim: int,
                 enc_num_layers: int,
                 document_vocab_size: int,
                 max_document_sentences: int,
                 doc_sequence_len: int,
                 dec_num_layers: int,
                 num_heads: int,
                 summary_vocab_size: int,
                 max_summary_sentences: int,
                 summary_sequence_len: int,
                 cnn_kernel_size: int = 3,
                 dropout: float = 0.5) -> nn.Module:
        super().__init__()

        # Create the FnetCNNEncoder
        self.encoder = MHACNNEncoder(
            num_layers=enc_num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            extend_dim=extend_dim,
            num_word_embeddings=document_vocab_size,
            num_type_embeddings=max_document_sentences,
            sequence_len=doc_sequence_len,
            cnn_kernel_size=cnn_kernel_size,
            dropout=dropout
        )
        
        # Create the Decoder
        self.decoder_embedding = TripleEmbeddingBlock(
            num_word_embeddings=summary_vocab_size,
            num_type_embeddings=max_summary_sentences,
            embedding_dim=model_dim,
            sequence_len=summary_sequence_len
        )
        
        decoder = nn.TransformerDecoderLayer(d_model=model_dim,
                                             nhead=num_heads,
                                             dim_feedforward=extend_dim,
                                             dropout=dropout,
                                             batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder,
                                             num_layers=dec_num_layers)

        # Create the output layer
        self.output_layer = nn.Linear(in_features=model_dim,
                                      out_features=summary_vocab_size)

    def forward(self,
                doc_tokens: torch.tensor,
                doc_token_types: torch.tensor,
                sum_tokens: torch.tensor,
                sum_token_types: torch.tensor) -> torch.tensor:
        # Pass documents through our encoder
        encoder_output = self.encoder(tokens=doc_tokens,
                                      token_types=doc_token_types)
        
        # Pass summaries and encoder_outputs to our decoder
        decoder_embeddings = self.decoder_embedding(sum_tokens,
                                                    sum_token_types)
        attn_mask = self.get_attn_mask(sum_tokens.shape[-1],
                                       sum_tokens.device)
        decoder_outputs = self.decoder(decoder_embeddings,
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
