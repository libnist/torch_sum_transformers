import torch
from torch import nn

import numpy as np

from ..components.blocks import TripleEmbeddingBlock

class Transformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_layers: int,
                 doc_vocab_size: int,
                 doc_max_num_sentences: int,
                 doc_seq_len: int,
                 sum_vocab_size: int,
                 sum_max_num_sentences: int,
                 sum_seq_len: int,
                 dropout: int = .5):
        """Return a VanillaTransformer.

        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of heads used in MHA block.
            dim_feedforward (int): Dimenstion of the first layer of the mlp 
            block.
            num_layers (int): Number of stacked layers.
            doc_vocab_size (int): Documents vocabulary size.
            doc_max_num_sentences (int): Maximum number of sentences in a 
            document.
            doc_seq_len (int): Maximum length of the tokens in input documents.
            sum_vocab_size (int): Summary vocabulary size.
            sum_max_num_sentences (int): Maximum number of sentences in a
            summary.
            sum_seq_len (int): Maximum length of the tokens in input summaries.
            dropout (int, optional): Dropout rate. Defaults to .5.
        """
        super().__init__()
        
        # Create embedding blocks
        self.enc_embedding = TripleEmbeddingBlock(
            num_word_embeddings=doc_vocab_size,
            num_type_embeddings=doc_max_num_sentences,
            embedding_dim=d_model,
            sequence_len=doc_seq_len
        )
        
        self.dec_embedding = TripleEmbeddingBlock(
            num_word_embeddings=sum_vocab_size,
            num_type_embeddings=sum_max_num_sentences,
            embedding_dim=d_model,
            sequence_len=sum_seq_len
        )
        
        # Create a torch TransformerEncoderLayer
        encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                             nhead=nhead,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout,
                                             activation="relu", 
                                             norm_first=False,
                                             batch_first=True)
        
        # Create a torch TransformerDecoderLayer
        decoder = nn.TransformerDecoderLayer(d_model=d_model,
                                             nhead=nhead,
                                             dim_feedforward=dim_feedforward,
                                             activation="relu",
                                             dropout=dropout,
                                             norm_first=False,
                                             batch_first=True)
        
        # Create Encoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder,
                                             num_layers=num_layers)
        
        # Create Decoder
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder,
                                             num_layers=num_layers)
        
        self.final = nn.Linear(in_features=d_model,
                               out_features=sum_vocab_size)
        
    def forward(self,
                doc_tokens: torch.tensor,
                doc_token_types: torch.tensor,
                sum_tokens: torch.tensor,
                sum_token_types: torch.tensor):
        
        enc_embedding = self.enc_embedding(tokens=doc_tokens,
                                           token_types=doc_token_types)
        dec_embedding = self.dec_embedding(tokens=sum_tokens,
                                           token_types=sum_token_types)
        
        encoder_out = self.encoder(enc_embedding)
        
        attn_mask = self.get_attn_mask(sum_tokens.shape[-1],
                                       sum_tokens.device)
        decoder_out = self.decoder(dec_embedding, encoder_out, attn_mask)
        
        return self.final(decoder_out)
        
        
    @staticmethod
    def get_attn_mask(summary_tokens_shape, device) -> torch.tensor:
        attn_mask = np.triu(
            m=np.ones((summary_tokens_shape,
                       summary_tokens_shape), dtype=bool),
            k=1
        )
        return torch.tensor(attn_mask).to(device)