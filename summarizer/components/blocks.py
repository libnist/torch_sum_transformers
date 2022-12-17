# Import libraries
import torch
from torch import nn
import torch.nn.functional as F


class TripleEmbeddingBlock(nn.Module):
    def __init__(self,
                 num_word_embeddings: int,
                 num_type_embeddings: int,
                 embedding_dim: int,
                 sequence_len: int) -> torch.nn.Module:
        """Return an embedding block that also uses positional and
        type embedding.

        Args:
            num_word_embeddings (int): Size of vocabulary.
            num_type_embeddings (int): Number of type embeddings.
            embedding_dim (int): Model dimensions.
            sequence_len (int): Length of the input sequence.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super(TripleEmbeddingBlock, self).__init__()
        
        # Create word embedding layer.
        self.word_embedding = nn.Embedding(num_embeddings=num_word_embeddings,
                                           embedding_dim=embedding_dim)
        
        # Create type embedding layer.
        self.type_embedding = nn.Embedding(num_embeddings=num_type_embeddings,
                                           embedding_dim=embedding_dim)
        
        # Create positional embedding layer.
        self.positional_embedding = nn.Parameter(
            torch.rand((1, sequence_len, embedding_dim),
                       requires_grad=True)
        )

    def forward(self,
                tokens: torch.tensor,
                token_types: torch.tensor) -> torch.tensor:
        
        # Getting the length of the input
        token_length = tokens.shape[-1]
        
        # Perform word embeddings.
        word_embedding = self.word_embedding(tokens)
        
        # Perform type embeddings.
        type_embedding = self.type_embedding(token_types)
        
        # Add all the embeddings to produce the output tensor
        return (word_embedding + 
                type_embedding + 
                self.positional_embedding[:, :token_length, :])


class MLPBlock(nn.Module):
    def __init__(self,
                 extend_dim: int,
                 output_dim: int,
                 dropout: float) -> torch.nn.Module:
        """Return the MLP block.

        Args:
            extend_dim (int): Dimension of first linear layer.
            output_dim (int): Dimension of the model.
            dropout (float): Dropout rate.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super(MLPBlock, self).__init__()

        # Creating the first linear layer.
        self.extend_layer = nn.Sequential(
            nn.Linear(in_features=output_dim,
                      out_features=extend_dim),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        # Creating the output linear layer.
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=extend_dim, 
                      out_features=output_dim),
            nn.Dropout(p=dropout)
        )

        # Creating the layer norm module.
        self.layer_norm = nn.LayerNorm(normalized_shape=output_dim)

    def forward(self,
                x: torch.tensor) -> torch.tensor:
        # Performing the first linear layer and it's dropout.
        output = self.extend_layer(x)

        # Performing output linear layer and it's dropout.
        output = self.output_layer(output)

        # Performing the residual connection and layer normalization.
        return self.layer_norm(output + x)


class FnetBlock(nn.Module):
    def __init__(self,
                 output_dim: int,
                 dropout: float) -> torch.nn.Module:
        """Returns the Fnet Block.

        Args:
            output_dim (int): Dimension of the model.
            dropout (float): Dropout rate.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super(FnetBlock, self).__init__()
        
        # Create a dropoutlayer
        self.dropout = nn.Dropout(p=dropout)

        # Creating the output linear layer.
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=output_dim, 
                      out_features=output_dim),
            nn.Dropout(p=dropout)
        )

        # Creating the layer norm module.
        self.layer_norm = nn.LayerNorm(normalized_shape=output_dim)

    def forward(self,
                x: torch.tensor) -> torch.tensor:
        # Performing the fft2d and it's dropout.
        output = self.dropout(torch.real(torch.fft.fft2(x)))

        # Perfroming the output layer and it's dropout.
        output = self.output_layer(output)

        # Performing the residual connection and layer normalization.
        return self.layer_norm(output + x)


class FnetCNNBlock(nn.Module):
    def __init__(self,
                 output_dim: int,
                 cnn_kernel_size: int,
                 dropout: float) -> torch.nn.Module:
        """Return the fnet conv block to shrink the input tokens.

        Args:
            output_dim (int): Dimension of the model.
            cnn_kernel_size (int): Kernel size of the conv layer.
            dropout (float): Dropout rate.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super(FnetCNNBlock, self).__init__()

        # Set the dropout rate as an instance attr.
        self.dropout = nn.Dropout(p=dropout)

        # Creating the output linear layer.
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=output_dim, 
                      out_features=output_dim),
            nn.Dropout(p=dropout)
        )

        # Creating the conv block. this block is responsable to
        # shrink tokens to half.
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=output_dim,
                      out_channels=output_dim,
                      kernel_size=cnn_kernel_size,
                      stride=1,
                      padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2),
            nn.Dropout(p=dropout)
        )

        # Creating the layer norm module.
        self.layer_norm = nn.LayerNorm(normalized_shape=output_dim)

    def forward(self,
                x: torch.tensor) -> torch.tensor:
        # Performing the fft2 and it's dropout.
        output = self.dropout(torch.real(torch.fft.fft2(x)))

        # Permute the fft2 output to put it into
        # shape: [batch, channel, tokens] -> [batch, model_dim, num_tokens]
        output = output.permute(0, 2, 1)

        # Performing the CNN block and it's dropout.
        output = self.cnn_block(output)

        # Permute the CNN block output to put it into
        # sahpe: [batch, tokens, channel] -> [batch, num_tokens, model_dim]
        output = output.permute(0, 2, 1)

        # Perform the output linear layer and it's dropout
        output = self.output_layer(output)

        # Perform a layer normalizatoin, residual connection cand be doen
        # cause the input and output shaps are different.
        return self.layer_norm(output)


class MHABlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float) -> torch.nn.Module:
        """Return the vanilla MultiheadSelfAttention block.

        Args:
            embed_dim (int): Dimension of query matris.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            kdim (int, optional): Dimension of the key matris.
            Defaults to None.
            vdim (int, optional): Dimension of the value matris.
            Defaults to None.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super(MHABlock, self).__init__()

        # Creating the MultiheadSelfAttention module.
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True)

        # Creating the layer norm module.
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self,
                query: torch.tensor,
                key: torch.tensor,
                value: torch.tensor,
                attn_mask: torch.tensor = None) -> torch.tensor:

        # Performing MHSA.
        output, _ = self.mha(query=query,
                             key=key,
                             value=value,
                             attn_mask=attn_mask)

        # Performing residual connection and layer normalization.
        return self.layer_norm(output + query)
