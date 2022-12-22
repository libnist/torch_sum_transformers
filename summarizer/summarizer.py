# import libs
import torch
from torch import nn

import numpy as np

from .components.tokenizer import CustomTokenizer

MAX_LENGTH = 64


class Summarizer(nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 document_tokenizer: CustomTokenizer,
                 summary_tokenizer: CustomTokenizer,
                 max_input_length: int,
                 max_output_length: int):
        super().__init__()

        self.model = model
        self.doc_tokenizer = document_tokenizer
        self.sum_tokenizer = summary_tokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

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
        input_tokens = encoder_input[:, 0, :]
        input_token_types = encoder_input[:, 1, :]

        start = self.sum_tokenizer.token_to_id("[START]")
        end_marks = ["[END]", "[FILL]"]
        end = [self.sum_tokenizer.token_to_id(mark)
               for mark in end_marks]
        

        marks = ".?!"
        type_changers = [self.sum_tokenizer.token_to_id(mark) 
                         for mark in marks]

        output_tokens_list = [start]
        current_type = 0
        output_token_types_list = [current_type]

        for _ in range(self.sum_tokenizer):
            output_tokens_tensor = self.torch_tensor(output_tokens_list)
            output_token_types_tensor = self.torch_tensor(
                output_token_types_list
            )

            prediction = self.model(input_tokens,
                                    input_token_types,
                                    output_tokens_tensor,
                                    output_token_types_tensor)

            prediction = prediction[:, -1, :]
            prediction_id = torch.argmax(
                torch.softmax(prediction),
                dim=-1).item()
            output_tokens_list.append(prediction_id)

            if prediction_id in type_changers:
                current_type += 1

            output_token_types_list.append(current_type)
            
            if prediction_id in end:
                break
        return self.sum_tokenizer.decode(output_tokens_list)

    def torch_tensor(self,
                     x):
        return (torch.tensor(x, dtype=torch.int64)
                .unsqueeze(0)
                .to(self.model.device))
