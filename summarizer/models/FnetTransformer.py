# Import libs
import torch
from torch import nn

from ..components.blocks import TripleEmbeddingBlock
from ..components.layers import FnetEncoderLayer
from .input_utils import get_attn_mask


class FnetTransformer(nn.Module):

    def __init__(self,
                 d_model: int,
                 dim_feedforward: int,
                 nhead: int,
                 num_layers: int,
                 doc_vocab_size: int,
                 max_num_doc_sentences: int,
                 doc_seq_len: int,
                 sum_vocab_size: int,
                 max_num_sum_sentences: int,
                 sum_seq_len: int,
                 dropout: float = .5):
        """Returns a Fnet Transformer model.

        Args:
            d_model (int): Dimension of the model.
            dim_feedforward (int): Dimension used in the first layer of the mlp
            block.
            nhead (int): Number of heads used in MHA block.
            num_layers (int): Number of layers for encoder and decoder layers.
            doc_vocab_size (int): Documents vocabulary size.
            max_num_doc_sentences (int): Maximum number of sentences in a
            document.
            doc_seq_len (int): Length of the input document tokens.
            sum_vocab_size (int): Summaries vocabulary size.
            max_num_sum_sentences (int): Maximum number of sentensces in a
            summary.
            sum_seq_len (int): Length of the inpu summary tokens.
            dropout (float, optional): Dropout rate. Defaults to .5.
        """
        super().__init__()

        # Create embedding blocs
        self.enc_embedding = TripleEmbeddingBlock(
            num_word_embeddings=doc_vocab_size,
            num_type_embeddings=max_num_doc_sentences,
            embedding_dim=d_model,
            sequence_len=doc_seq_len
        )

        self.dec_embedding = TripleEmbeddingBlock(
            num_word_embeddings=sum_vocab_size,
            num_type_embeddings=max_num_sum_sentences,
            embedding_dim=d_model,
            sequence_len=sum_seq_len
        )

        # Create FnetEncoderLayer
        encoders = [FnetEncoderLayer(model_dim=d_model,
                                     extend_dim=dim_feedforward,
                                     dropout=dropout)
                    for _ in range(num_layers)]

        # Create a TransformerDecoderLayer
        decoder = nn.TransformerDecoderLayer(d_model=d_model,
                                             nhead=nhead,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout,
                                             activation="relu",
                                             batch_first=True,)
        # Create the Encoder
        self.encoder = nn.Sequential(*encoders)

        # Create the Decoder
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder,
                                             num_layers=num_layers)

        # The final output layer
        self.final = nn.Linear(in_features=d_model,
                               out_features=sum_vocab_size)

    def forward(self,
                doc_tokens: torch.tensor,
                doc_token_types: torch.tensor,
                sum_tokens: torch.tensor,
                sum_token_types: torch.tensor):

        # First we embed our inputs
        encoder_input = self.enc_embedding(doc_tokens, doc_token_types)
        decoder_input = self.dec_embedding(sum_tokens, sum_token_types)

        # Now we pass our embedded inputs throw our layers
        encoder_output = self.encoder(encoder_input)

        attn_mask = get_attn_mask(sum_tokens.shape[-1],
                                  sum_tokens.device)

        decoder_output = self.decoder(decoder_input,
                                      encoder_output,
                                      attn_mask)

        return self.final(decoder_output)
