# Import libs
import torch
from torch import nn

from transformers import FNetForPreTraining
from ..components.layers import FnetCNNEncoderLayer
from ..utils import get_attn_mask


class FnetPretrainedModel(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 num_heads: int,
                 cnn_kernel: int = 3):
        super().__init__()

        FnetPretrained = FNetForPreTraining.from_pretrained("google/fnet-base")

        self.embedding = FnetPretrained.fnet.embeddings
        self.requires_grad_off(self.embedding)

        d_model = self.embedding.word_embeddings.embedding_dim
        output_dim = self.embedding.word_embeddings.num_embeddings

        self.pretrained_encoder = FnetPretrained.fnet.encoder
        self.requires_grad_off(self.pretrained_encoder)

        dim_feedforward = (self.pretrained_encoder
                           .layer[0].output.dense.in_features)

        dropout = self.pretrained_encoder.layer[0].output.dropout.p

        self.compress_encoder = nn.Sequential(
            *[FnetCNNEncoderLayer(model_dim=d_model,
                                  extend_dim=dim_feedforward,
                                  fnet_cnn_kernel_size=cnn_kernel,
                                  dropout=dropout)
              for _ in range(num_encoder_layers)]
        )

        decoder = nn.TransformerDecoderLayer(d_model=d_model,
                                             nhead=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout,
                                             activation="relu",
                                             batch_first=True)

        self.decoder = nn.TransformerDecoder(decoder_layer=decoder,
                                             num_layers=num_decoder_layers)
        
        self.final = nn.Linear(in_features=d_model, out_features=output_dim)

    def forward(self,
                doc_tokens: torch.tensor,
                sum_tokens: torch.tensor):

        x = self.embedding(doc_tokens, None)
        x = self.pretrained_encoder(x).last_hidden_state
        x = self.compress_encoder(x)
        
        y = self.embedding(sum_tokens, None)
        attn_mask = get_attn_mask(sum_tokens.shape[-1], sum_tokens.device)
        y = self.decoder(y, x, attn_mask)
        
        return self.final(y)
    
    def requires_grad_off(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False
