import torch
from torch import nn
import torch.nn.functional as F


class LightConvBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 kernel_size: int = 3,
                 dropout: float = 0.3,
                 dilation: int = None,
                 maxpool: bool = False):
        super().__init__()
        
        self.maxpool = maxpool

        assert not d_model % n_heads, f"{d_model} is not divisable by {n_heads}"

        self.d_model = d_model
        self.dropout = dropout
        self.channels = d_model // n_heads

        self.in_linear = nn.Linear(in_features=d_model,
                                   out_features=2 * d_model)

        self.glu = nn.GLU()

        if dilation:
            self.weight = nn.Parameter(
                torch.rand(size=(self.channels, 1, 3)),
                requires_grad=True
            )
            self.bias = nn.Parameter(
                torch.rand(size=(self.channels, )),
                requires_grad=True
            )
            self.dilation = dilation
        else:
            self.weight = nn.Parameter(
                torch.rand(size=(self.channels, 1, kernel_size)),
                requires_grad=True
            )
            self.bias = nn.Parameter(
                torch.rand(size=(self.channels, )),
                requires_grad=True
            )
            self.dilation = 1

        self.out_linear = nn.Linear(in_features=d_model,
                                    out_features=d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.tensor):
        conv_in = self.in_linear(x)
        conv_in = self.glu(conv_in)
        conv_in = conv_in.permute(0, 2, 1)
        
        output = F.conv1d(
                input=conv_in[:, :self.channels, :],
                weight=F.dropout(F.softmax(self.weight, dim=-1),
                                 p=self.dropout),
                bias=self.bias,
                padding="same",
                groups=self.channels,
                dilation=self.dilation
        )

        for i in range(self.channels, self.d_model, self.channels):
            temp = F.conv1d(
                input=conv_in[:, i:i+self.channels, :],
                weight=F.dropout(F.softmax(self.weight, dim=-1),
                                 p=self.dropout),
                bias=self.bias,
                padding="same",
                groups=self.channels,
                dilation=self.dilation
            )
            output = torch.concat((output, temp), dim=1)
        if self.maxpool:
            output = F.max_pool1d(
                output,
                kernel_size=2,
                stride=2
            )
        output = self.out_linear(output.permute(0, 2, 1))
        if self.maxpool:
            return self.layer_norm(output)
        return self.layer_norm(output + x)


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
            nn.ReLU(),
        )

        # Creating the output linear layer.
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=extend_dim,
                      out_features=output_dim),
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
                key: torch.tensor = None,
                value: torch.tensor = None,
                attn_mask: torch.tensor = None) -> torch.tensor:
        # Performing MHSA.
        
        if key is None and value is None:
            key, value = query, query
        
        output, _ = self.mha(query=query,
                             key=key,
                             value=value,
                             attn_mask=attn_mask)

        # Performing residual connection and layer normalization.
        return self.layer_norm(output + query)
