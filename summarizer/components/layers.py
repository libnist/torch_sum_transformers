# Import libraries
import torch
from torch import nn
import torch.nn.functional as F

from .blocks import (
    FnetCNNBlock,
    MHABlock,
    MLPBlock,
    TripleEmbeddingBlock,
    FnetBlock
)


class FnetCNNEncoderLayer(nn.Module):
    def __init__(self,
                 model_dim: int,
                 extend_dim: int,
                 fnet_cnn_kernel_size: int,
                 dropout: float = .5) -> torch.nn.Module:
        """Returns an encoder layer comprised of FnetCNNBlock in order to shrink
        the input tokens.

        Args:
            model_dim (int): Dimension of the model.
            extend_dim (int): Dimension of the first linear layer in MLP block.
            fnet_cnn_kernel_size (int, optional): Kernel size of the CNN used in
            FnetCNN block.
            dropout (float, optional): Dropout rate. Defaults to .5.
        Returns:
            FnetCNNEncoderLayer: torch.nn.Module
        """
        super().__init__()

        self.fnet_cnn_block = FnetCNNBlock(
            output_dim=model_dim,
            cnn_kernel_size=fnet_cnn_kernel_size,
            dropout=dropout)

        self.mlp_block = MLPBlock(extend_dim=extend_dim,
                                  output_dim=model_dim,
                                  dropout=dropout)

    def forward(self,
                x: torch.tensor) -> torch.tensor:

        # Pass the input through FnetCNNBlock.
        output = self.fnet_cnn_block(x)

        # Pass the input through MLP block.
        return self.mlp_block(output)

class FnetEncoderLayer(nn.Module):
    def __init__(self,
                 model_dim: int,
                 extend_dim: int,
                 dropout: float = .5) -> torch.nn.Module:
        """Returns an encoder layer comprised of FnetBlock.

        Args:
            model_dim (int): Dimension of the model.
            extend_dim (int): Dimension of the first linear layer in MLP block.
            dropout (float, optional): Dropout rate. Defaults to .5.
        Returns:
            FnetEncoderLayer: torch.nn.Module
        """
        super().__init__()

        self.fnet_block = FnetBlock(
            output_dim=model_dim,
            dropout=dropout)

        self.mlp_block = MLPBlock(extend_dim=extend_dim,
                                  output_dim=model_dim,
                                  dropout=dropout)

    def forward(self,
                x: torch.tensor) -> torch.tensor:

        # Pass the input through FnetBlock.
        output = self.fnet_block(x)

        # Pass the input through MLP block.
        return self.mlp_block(output)

class DecoderLayer(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 extend_dim: int,
                 dropout: float = .5) -> torch.nn.Module:
        """Return a VanillaDecoder Layer.

        Args:
            model_dim (int): Dimension of the model.
            num_heads (int): Num heads used in MHA block.
            extend_dim (int): Dimenstion used in the first layer if mlp block.
            dropout (float, optional): Dropout rate. Default to .5.
            
        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super().__init__()

        # Creating the MHA block that perfroms on decoder input.
        self.mha_1_block = MHABlock(embed_dim=model_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)

        # Creating the MHA blpcl that perfroms both on past MHA
        # output and encoders output
        self.mha_2_block = MHABlock(embed_dim=model_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)

        # Creating MLP block
        self.mlp_block = MLPBlock(extend_dim=extend_dim,
                                  output_dim=model_dim,
                                  dropout=dropout)

    def forward(self,
                x: torch.tensor,
                encoder_output: torch.tensor,
                attn_mask: torch.tensor) -> torch.tensor:

        # Perform first MHA block, this block is only related
        # to the decoders input.
        mha_1_block_output = self.mha_1_block(query=x,
                                              key=x,
                                              value=x,
                                              attn_mask=attn_mask)

        # Perform second MHA block, this block maps decoders input
        # to encoders output
        mha_2_block_output = self.mha_2_block(query=mha_1_block_output,
                                              key=encoder_output,
                                              value=encoder_output)

        # Perform the MLP Block.
        return self.mlp_block(mha_2_block_output)


class FnetCNNEncoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 model_dim: int,
                 extend_dim: int,
                 num_word_embeddings: int,
                 num_type_embeddings: int,
                 sequence_len: int,
                 fnet_cnn_kernel_size: int = 3,
                 dropout: float = 0.5) -> torch.nn.Module:
        """Return an FnetCNNEncoder with `num_layers` of FnetEncoderLayer.

        Args:
            num_layers (int): Number of FnetEncoderLayers.
            model_dim (int): Dimension of the model.
            extend_dim (int): Dimension of the firt linear layer in MLP block.
            num_word_embeddings (int): Input documents vocabulary size.
            num_type_embeddings (int): Max number of sentences of documents.
            sequence_len (int): Length of the documents tokens.
            fnet_cnn_kernel_size (int, optional): Kernel size used in
            FnetCNN block. Defaults to 3.
            dropout (float, optional): Dropout rate. Defaults to 0.5.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super().__init__()

        # Set the num_layers as an instance attr.
        self.num_layers = num_layers

        # Create the embedding block.
        self.embedding = TripleEmbeddingBlock(
            num_word_embeddings=num_word_embeddings,
            num_type_embeddings=num_type_embeddings,
            embedding_dim=model_dim,
            sequence_len=sequence_len
        )

        # Create a list of FnetCNNEncoderLayer Module.
        self.encoder_layers = nn.Sequential(
            *[FnetCNNEncoderLayer(model_dim=model_dim,
                                  extend_dim=extend_dim,
                                  fnet_cnn_kernel_size=fnet_cnn_kernel_size,
                                  dropout=dropout)
              for _ in range(num_layers)]
        )

    def forward(self,
                tokens: torch.tensor,
                token_types: torch.tensor) -> torch.tensor:
        # Perform the embeddings
        x = self.embedding(tokens, token_types)

        # Go through all FnetCNNEncoderLayers.
        return self.encoder_layers(x)


class Decoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 model_dim: int,
                 num_heads: int,
                 extend_dim: int,
                 num_word_embeddings: int,
                 num_type_embeddings: int,
                 sequence_len: int,
                 dropout: float = 0.5) -> torch.nn.Module:
        """Return a Decoder stacked with DecoderLayers.

        Args:
            num_layers (int): Number of DecoderLayers.
            model_dim (int): Dimension of the model.
            num_heads (int): Number of attention heads in MHA block.
            extend_dim (int): Dimension used in the first linear layer of the
            mlp block.
            num_word_embedding (int): Vocabulary size of summaries.
            num_type_embedding (int): Max number of sentencs in summaries.
            sequence_len (int): Number of input tokens.
            dropout (float, optional): Dropout rate. Defaults to 0.5.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super().__init__()

        # Setting number of layers as an instance attr.
        self.num_layers = num_layers

        # Creating the embedding layer
        self.embedding = TripleEmbeddingBlock(
            num_word_embeddings=num_word_embeddings,
            num_type_embeddings=num_type_embeddings,
            embedding_dim=model_dim,
            sequence_len=sequence_len
        )

        # Create a list of DecoderLayers.
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim=model_dim,
                          num_heads=num_heads,
                          extend_dim=extend_dim,
                          dropout=dropout)
             for _ in range(num_layers)]
        )

    def forward(self,
                tokens: torch.tensor,
                token_types: torch.tensor,
                encoder_output: torch.tensor,
                attn_mask: torch.tensor) -> torch.tensor:
        """Forward pass of the Decoder.

        Args:
            tokens (torch.tensor): Tokens of input summaries.
            token_types (torch.tensor): Type tokens of input summaries.
            encoder_output (torch.tensor): The output from the last layer of 
            the encoder.
            attn_mask (torch.tensor): 2D attention mask to prevent look ahead
            bias.

        Returns:
            torch.tensro: Output of Decoder.
        """

        # First we embedd the tokens
        x = self.embedding(tokens, token_types)

        # Now we pass the embedded tokens through our stack of DecoderLayers
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x=x,
                                       encoder_output=encoder_output,
                                       attn_mask=attn_mask)
        return x
