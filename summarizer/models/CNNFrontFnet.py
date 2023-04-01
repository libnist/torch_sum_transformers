# import libs
import torch
from torch import nn

from ..components.blocks import (
    TripleEmbeddingBlock,
    CNNBlock
)

from ..components.layers import FnetEncoderLayer
from ..utils import get_attn_mask


class CNNFrontFnetModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_layers: int,
                 num_cnn_blocks: int,
                 dim_feedforward: int,
                 num_heads: int,
                 doc_vocab_size: int,
                 doc_max_num_sentences: int,
                 doc_seq_len: int,
                 sum_vocab_size: int,
                 sum_max_num_sentences: int,
                 sum_seq_len: int,
                 cnn_kernel_size: int = 3,
                 dropout: float = .5):
        """Return a CNNFrontFnetModel

        Args:
            d_model (int): Dimension of the model.
            num_layers (int): Number of encoder and decoder layers.
            num_cnn_blocks (int): Number of cnn blocks.
            dim_feedforward (int): Dimension of the feedforward block.
            num_heads (int): Number of attention heads.
            doc_vocab_size (int): Vocabulary size of the inputs.
            doc_max_num_sentences (int): Maximum number of sentences appeared
            in a document.
            doc_seq_len (int): Sequence length of the input documents.
            sum_vocab_size (int): Vocabulary size of the summary.
            sum_max_num_sentences (int): Maximum number of sentences appeared
            in a document.
            sum_seq_len (int): Maximum number of sentences appeared in a
            summary.
            cnn_kernel_size (int, optional): Kernel size of the CNN blocks.
            . Defaults to 3.
            dropout (float, optional): Dropout rate. Defaults to .5.
        """
        super().__init__()

        # The embedding below is used for input documents
        self.enc_embedding = TripleEmbeddingBlock(
            num_word_embeddings=doc_vocab_size,
            num_type_embeddings=doc_max_num_sentences,
            embedding_dim=d_model,
            sequence_len=doc_seq_len
        )

        # The embedding below is used for output documents
        self.dec_embedding = TripleEmbeddingBlock(
            num_word_embeddings=sum_vocab_size,
            num_type_embeddings=sum_max_num_sentences,
            embedding_dim=d_model,
            sequence_len=sum_seq_len
        )

        # The cnn block that is used for compressing input document
        self.cnn_block = CNNBlock(num_cnn_blocks=num_cnn_blocks,
                                  model_dim=d_model,
                                  kernel_size=cnn_kernel_size,
                                  dropout=dropout)

        # Create a list of FnetCNNEncoderLayer Module.
        self.encoder = nn.Sequential(
            *[FnetEncoderLayer(model_dim=d_model,
                               extend_dim=dim_feedforward,
                               dropout=dropout)
              for _ in range(num_layers)]
        )

        decoder = nn.TransformerDecoderLayer(d_model=d_model,
                                             nhead=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout,
                                             activation="relu",
                                             batch_first=True)

        self.decoder = nn.TransformerDecoder(decoder_layer=decoder,
                                             num_layers=num_layers)
        
        self.final = nn.Linear(in_features=d_model,
                               out_features=sum_vocab_size)
        
    def forward(self, 
                doc_tokens: torch.tensor,
                doc_token_types: torch.tensor,
                sum_tokens: torch.tensor,
                sum_token_types: torch.tensor) -> torch.tensor:
        
        embedded_inputs = self.enc_embedding(doc_tokens, doc_token_types)
        embedded_outputs = self.dec_embedding(sum_tokens, sum_token_types)
        
        compressed_inputs = self.cnn_block(embedded_inputs)
        
        encoder_output = self.encoder(compressed_inputs)
        
        attn_mask = get_attn_mask(sum_tokens.shape[-1],
                                  device=sum_tokens.device)
        decoder_output = self.decoder(embedded_outputs,
                                      encoder_output,
                                      attn_mask)
        return self.final(decoder_output)
        
        
