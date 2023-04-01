# import libs
import torch
from torch import nn

import numpy as np

from .utils.tokenizer import CustomTokenizer

class GreedySummarizer(nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 document_tokenizer: CustomTokenizer,
                 summary_tokenizer: CustomTokenizer,
                 max_input_length: int,
                 max_output_length: int,
                 summary_max_num_sentences: int,
                 with_token_types: int = True):
        super().__init__()
        
        self.with_token_types = with_token_types
        self.model = model
        self.model.eval()

        self.doc_tokenizer = document_tokenizer
        self.sum_tokenizer = summary_tokenizer

        self._fill_token_id = document_tokenizer.token_to_id("[FILL]")

        self.device = next(self.model.parameters()).device

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.sum_max_num_sent = summary_max_num_sentences

        self.start = self.sum_tokenizer.token_to_id("[START]")
        end_marks = ["[END]", "[FILL]"]
        self.end = [self.sum_tokenizer.token_to_id(mark)
                    for mark in end_marks]

        marks = ".?!"
        self.type_changers = [self.sum_tokenizer.token_to_id(mark)
                              for mark in marks]

    def forward(self,
                document: str):
        assert isinstance(document, str), "input document must be str."

        # Preparing Encoders input

        # 1. Tokenize the input sentence
        encoder_input = self.doc_tokenizer.encode(document)
        encoder_input = np.array(encoder_input)

        # 2. Make the input longer or shorter in order for
        # it to be in the desirable shape
        shape = encoder_input.shape
        if shape[-1] < self.max_input_length:
            array = np.ones((2, self.max_input_length)) * self._fill_token_id
            array[:, :shape[-1]] = encoder_input[:, :shape[-1]]
            encoder_input = array
        elif shape[-1] > self.max_input_length:
            encoder_input = encoder_input[:, :self.max_input_length]

        # 3. Turn our input to `torch.tensor` so it can be fed into the model.
        encoder_input = self.torch_tensor(encoder_input)

        # 4. Divide the input into tokens and token types
        input_tokens = encoder_input[0]
        input_token_types = encoder_input[1]

        output_tokens_list = [self.start]
        current_type = 0
        output_token_types_list = [current_type]

        for _ in range(self.max_output_length):
            output_tokens_tensor = self.torch_tensor(output_tokens_list)
            output_token_types_tensor = self.torch_tensor(
                output_token_types_list
            )
            
            if self.with_token_types:
                prediction = self.model(input_tokens,
                                        input_token_types,
                                        output_tokens_tensor,
                                        output_token_types_tensor)
            else:
                prediction = self.model(input_tokens,
                                        output_tokens_tensor)

            prediction = prediction[0, -1, :].log_softmax(dim=-1).argmax(dim=-1)
            prediction_id = prediction.item()
            output_tokens_list.append(prediction_id)
            
            if self.with_token_types:
                if (prediction_id in self.type_changers
                    and current_type < self.sum_max_num_sent):
                    current_type += 1

                output_token_types_list.append(current_type)

            if prediction_id in self.end:
                break
        return self.sum_tokenizer.decode(output_tokens_list)

    def torch_tensor(self, x):
        x = torch.tensor(x, dtype=torch.int64).to(self.device)
        return x
