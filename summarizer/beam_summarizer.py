import torch
from torch import nn

from .utils.tokenizer import CustomTokenizer


class BeamSummarizer(nn.Module):
    
    def __init__(self,
                 model: torch.nn.Module,
                 document_tokenizer: CustomTokenizer,
                 summary_tokenizer:CustomTokenizer,
                 max_out_len: int,
                 sum_max_num_sentences: int,
                 beam_width):
        
        self.model = model
        self.model.eval()
        
        self._doc_tokenizer = document_tokenizer
        self._sum_tokenizer = summary_tokenizer
        
        self.max_out_len = max_out_len
        self._sum_max_num_sent = sum_max_num_sentences
        self.beam_width = beam_width
        
        self.device = next(self.model.parameters()).device()
        self.vocab_size = summary_tokenizer.vocab_size
        
    def torch_tensor(self, x):
        return torch.LongTenosr(x).to(self.device)
        
    def next_candidates(self,
                        encoder_input_tokens,
                        encoder_input_token_ids,
                        decoder_input_tokens,
                        decoder_input_token_ids,
                        log_probs = torch.tensor([[0.0]], requires_grad=True)):
        
        encoder_batch_size = encoder_input_tokens.shape[0]
        decoder_batch_size = decoder_input_tokens.shape[0]
        
        if encoder_batch_size != decoder_batch_size:
            encoder_input_tokens = encoder_input_tokens.repeat(decoder_batch_size, 1)
            encoder_input_token_ids = encoder_input_token_ids.repeat(decoder_batch_size, 1)
        
        log_probs_shape = log_probs.shape[0]
        
        error_msg = (f"Shape missmatch of previous log_probs: "+
                     f"number of output samples= {encoder_batch_size}"+
                     f"number of previous_probs= {log_probs_shape}")
        
        assert encoder_batch_size == log_probs_shape, error_msg
        
        logits = self.model(encoder_input_tokens, encoder_input_token_ids,
                            decoder_input_tokens, decoder_input_token_ids)
        
        logits += log_probs
        
        logits = logits[:, -1, :].view(1, -1)
        
        topk = logits.topk(k=self.beam_width, dim=-1)
        
        topk_indices = topk.indices
        log_probs = topk.values.view(-1, 1)
        
        which_candid = topk_indices // self.vocab_size
        what_token = (topk_indices % self.vocab_size).view(-1, 1)
        
        new_candidates = decoder_input_tokens[which_candid[0]]
        new_candidates = torch.concat((new_candidates, what_token), dim=-1)
        
        
        
        return new_candidates, log_probs
        
        
        
        
        
        
        
        
        