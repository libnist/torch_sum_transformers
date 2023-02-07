import torch
from torch import nn
import torch.nn.functional as F

from ..components.light_conv_layers import (
    EncoderLayer,
    DecoderLayer
)

from ..components.blocks import TripleEmbeddingBlock

class LightConvModel(nn.Module):
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 source_max_sentences: int,
                 target_max_sentences: int,
                 source_max_seq_len: int,
                 target_max_seq_len: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 padding_index: int = None,
                 dropout: float = 0.3,
                 encoder_kernels: list = [3, 7, 15, 31, 31, 31, 31],
                 decoder_kernels: list = [3, 7, 15, 31, 31, 31],
                 encoder_dilations: list = None,
                 decoder_dilations: list = None):
        
        super().__init__()
        
        self.enc_embeddig = TripleEmbeddingBlock(
            num_word_embeddings=source_vocab_size,
            num_type_embeddings=source_max_sentences,
            embedding_dim=d_model,
            sequence_len=source_max_seq_len,
            padding_index=padding_index
        )
        
        self.dec_embeddig = TripleEmbeddingBlock(
            num_word_embeddings=target_vocab_size,
            num_type_embeddings=target_max_sentences,
            embedding_dim=d_model,
            sequence_len=target_max_seq_len,
            padding_index=padding_index
        )
        
        if encoder_dilations:
            self.encoder = nn.Sequential(
                *[EncoderLayer(d_model=d_model,
                               n_heads=n_heads,
                               dim_feedforward=dim_feedforward,
                               dropout=dropout,
                               dilation=dilation)
                  for dilation in encoder_dilations]
            )
        else:
            self.encoder = nn.Sequential(
                *[EncoderLayer(d_model=d_model,
                               n_heads=n_heads,
                               dim_feedforward=dim_feedforward,
                               kernel_size=kernel,
                               dropout=dropout)
                  for kernel in encoder_kernels]
            )
            
        if decoder_dilations:
            self.decoder = nn.ModuleList(
                [DecoderLayer(d_model=d_model,
                              n_heads=n_heads,
                              dim_feedforward=dim_feedforward,
                              dropout=dropout,
                              dilation=dilation)
                 for dilation in decoder_dilations]
            )
        else:
            self.decoder = nn.ModuleList(
                [DecoderLayer(d_model=d_model,
                              n_heads=n_heads,
                              dim_feedforward=dim_feedforward,
                              kernel_size=kernel,
                              dropout=dropout)
                 for kernel in decoder_kernels]
            )
            
        self.classifier = nn.Linear(in_features=d_model,
                                    out_features=target_vocab_size)
            
    def forward(self,
                source_tokens: torch.tensor,
                source_token_types: torch.tensor,
                target_tokens: torch.tensor,
                target_token_types: torch.tensor):
        enc_embeddings = self.enc_embeddig(source_tokens,
                                           source_token_types)
        
        enc_output = self.encoder(enc_embeddings)
        
        output = self.dec_embeddig(target_tokens,
                                   target_token_types)
        
        for decoder in self.decoder:
            output = decoder(output,
                             enc_output)
            
        return self.classifier(output)