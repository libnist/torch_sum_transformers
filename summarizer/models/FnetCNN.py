# Import libraries
import torch
from torch import nn

import numpy as np

from ..components.layers import FnetCNNEncoder
from ..components.blocks import TripleEmbeddingBlock

class FnetCNNModel(nn.Module):
    def __init__(self,
                 model_dim: int,
                 extend_dim: int,
                 enc_num_layers: int,
                 document_vocab_size: int,
                 dec_num_layers: int,
                 num_heads: int,
                 summary_vocab_size: int,
                 max_document_sentences: int = None,
                 max_summary_sentences: int = None,
                 padding_index: int = None,
                 fnet_cnn_kernel_size: int = 3,
                 dropout: float = 0.5) -> nn.Module:
        """Creates an FnetCNNTransformer.

        Args:
            model_dim (int): Dimension of the model.
            extend_dim (int): Dimension used in the first linear layer of
            the mlp block.
            enc_num_layers (int): Number of FnetCNNEncoder layers. these layers
            compress tokens in half in each layer.
            document_vocab_size (int): Vocabulary size of the documents.
            max_document_sentences (int): Maximum number of sentences
            comprising a document.
            dec_num_layers (int): Number of Decoder layers.
            num_heads (int): Number of heads used in MHA blocks.
            summary_vocab_size (int): Vocabulary size of the summaries.
            max_summary_sentences (int): Maximum number of sentences
            comprising a summary.
            fnet_cnn_kernel_size (int, optional): Kernel size of the CNN used
            in the FnetCNNEncoder (won't be effecting compression size of the 
            documents). Defaults to 3.
            dropout (float, optional): Dropout rate. Defaults to 0.5.

        Returns:
            nn.Module: PyTorch Model.
        """
        super().__init__()

        # Create the FnetCNNEncoder
        self.encoder = FnetCNNEncoder(
            num_layers=enc_num_layers,
            model_dim=model_dim,
            extend_dim=extend_dim,
            num_word_embeddings=document_vocab_size,
            num_type_embeddings=max_document_sentences,
            fnet_cnn_kernel_size=fnet_cnn_kernel_size,
            dropout=dropout,
            padding_index=padding_index
        )
        
        # Create the Decoder
        self.decoder_embedding = TripleEmbeddingBlock(
            num_word_embeddings=summary_vocab_size,
            num_type_embeddings=max_summary_sentences,
            embedding_dim=model_dim,
            padding_index=padding_index
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
                sum_tokens: torch.tensor,
                doc_token_types: torch.tensor = None,
                sum_token_types: torch.tensor = None) -> torch.tensor:
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
