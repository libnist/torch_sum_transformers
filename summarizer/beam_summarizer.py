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
        super().__init__()
        self.model = model
        self.model.eval()
        
        self._doc_tokenizer = document_tokenizer
        self._sum_tokenizer = summary_tokenizer
        
        self.max_out_len = max_out_len
        self._sum_max_num_sent = sum_max_num_sentences
        self.beam_width = beam_width
        
        self.device = next(self.model.parameters()).device
        self.vocab_size = summary_tokenizer.vocab_size
        
        marks = ".?!"
        self.type_changers = [summary_tokenizer.token_to_id(mark)
                              for mark in marks]
        
    def forward(self, input_str):
        
        error_msg = "input should be a string."
        assert isinstance(input_str, str), error_msg
        
        encoder_in, encoder_in_types = self._doc_tokenizer.encode(input_str)
        
        encoder_in = self.torch_tensor(encoder_in)
        encoder_in_types = self.torch_tensor(encoder_in_types)
        
        decoder_in, decoder_in_types = self._sum_tokenizer.encode("")
        decoder_in = self.torch_tensor(decoder_in[:-1])
        decoder_in_types = self.torch_tensor(decoder_in_types[:-1])
        
        initial_types = torch.zeros(size=(self.beam_width, 1))
        
        log_probs = torch.tensor([[0.0]], requires_grad=True)
        
        decoder_in, log_probs = self.next_candidates(
                encoder_input_tokens=encoder_in,
                encoder_input_token_ids=encoder_in_types,
                decoder_input_tokens=decoder_in,
                decoder_input_token_ids=decoder_in_types,
                log_probs=log_probs
            )
        
        new_types = self.next_types(new_candidates=decoder_in,
                                    current_types=initial_types)
        decoder_in_types = torch.concat((initial_types, new_types), dim=-1)
        
        for _ in range(self.max_out_len - 1):
            decoder_in, log_probs = self.next_candidates(
                encoder_input_tokens=encoder_in,
                encoder_input_token_ids=encoder_in_types,
                decoder_input_tokens=decoder_in,
                decoder_input_token_ids=decoder_in_types,
                log_probs=log_probs
            )
            new_types = self.next_types(new_candidates=decoder_in,
                                        current_types=new_types)
            decoder_in_types = torch.concat((decoder_in_types, new_types),
                                            dim=-1)
        return self._sum_tokenizer.decode(decoder_in.tolist()[0])
        
    def torch_tensor(self, x):
        return torch.LongTensor(x).unsqueeze(0).to(self.device)
        
    def next_candidates(self,
                        encoder_input_tokens,
                        encoder_input_token_ids,
                        decoder_input_tokens,
                        decoder_input_token_ids,
                        log_probs):
        
        encoder_batch_size = encoder_input_tokens.shape[0]
        decoder_batch_size = decoder_input_tokens.shape[0]
        
        if encoder_batch_size != decoder_batch_size:
            encoder_input_tokens = encoder_input_tokens.repeat(decoder_batch_size, 1)
            encoder_input_token_ids = encoder_input_token_ids.repeat(decoder_batch_size, 1)
        
        log_probs_shape = log_probs.shape[1]
        
        error_msg = (f"Shape missmatch of previous log_probs: "+
                     f"number of output samples= {encoder_batch_size}, "+
                     f"number of previous_probs= {log_probs_shape}")
        
        assert encoder_batch_size == log_probs_shape, error_msg
        
        logits = self.model(encoder_input_tokens, encoder_input_token_ids,
                            decoder_input_tokens, decoder_input_token_ids)
        
        logits = logits[:, -1, :].contiguous()
        
        logits += log_probs
        
        logits = logits.view(1, -1)
        
        topk = logits.topk(k=self.beam_width, dim=-1)
        
        topk_indices = topk.indices
        log_probs = topk.values.view(-1, 1)
        
        which_candid = topk_indices // self.vocab_size
        what_token = (topk_indices % self.vocab_size).view(-1, 1)
        
        new_candidates = decoder_input_tokens[which_candid[0]]
        new_candidates = torch.concat((new_candidates, what_token), dim=-1)
        
        
        
        return new_candidates, log_probs
    
    def next_types(self, new_candidates, current_types):
        new_candidates = new_candidates[:, -1:]
        
        new_types = torch.zeros_like(new_candidates, dtype=torch.bool)
        
        for type_changer in self.type_changers:
            new_types = new_types | (new_candidates == type_changer)
            
        return current_types + new_types
        
        
        
        
        
        
        
        
        
        