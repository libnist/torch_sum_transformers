# import libs
import torch
from torch import nn

import numpy as np

from .utils.tokenizer import CustomTokenizer

MAX_LENGTH = 64


class Summarizer(nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 document_tokenizer: CustomTokenizer,
                 summary_tokenizer: CustomTokenizer,
                 max_input_length: int,
                 max_output_length: int,
                 summary_max_num_sentences: int,
                 k: int):
        super().__init__()

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

        self.k = k

    # def forward(self,
    #             document: str):
    #     assert isinstance(document, str), "input document must be str."

    #     # Preparing Encoders input

    #     # 1. Tokenize the input sentence
    #     encoder_input = self.doc_tokenizer.encode(document)
    #     encoder_input = np.array(encoder_input)

    #     # 2. Make the input longer or shorter in order for
    #     # it to be in the desirable shape
    #     shape = encoder_input.shape
    #     if shape[-1] < self.max_input_length:
    #         array = np.ones((2, self.max_input_length)) * self._fill_token_id
    #         array[:, :shape[-1]] = encoder_input[:, :shape[-1]]
    #         encoder_input = array
    #     elif shape[-1] > self.max_input_length:
    #         encoder_input = encoder_input[:, :self.max_input_length]

    #     # 3. Turn our input to `torch.tensor` so it can be fed into the model.
    #     encoder_input = self.torch_tensor(encoder_input)

    #     # 4. Divide the input into tokens and token types
    #     input_tokens = encoder_input[:, 0, :]
    #     input_token_types = encoder_input[:, 1, :]

    #     output_tokens_list = [self.start]
    #     current_type = 0
    #     output_token_types_list = [current_type]

    #     for _ in range(self.max_output_length):
    #         output_tokens_tensor = self.torch_tensor(output_tokens_list)
    #         output_token_types_tensor = self.torch_tensor(
    #             output_token_types_list
    #         )

    #         prediction = self.model(input_tokens,
    #                                 input_token_types,
    #                                 output_tokens_tensor,
    #                                 output_token_types_tensor)

    #         prediction = prediction[:, -1, :]
    #         prediction_id = torch.argmax(
    #             torch.softmax(prediction, dim=-1)
    #         ).item()
    #         output_tokens_list.append(prediction_id)

    #         if (prediction_id in self.type_changers
    #             and current_type < self.sum_max_num_sent):
    #             current_type += 1

    #         output_token_types_list.append(current_type)

    #         if prediction_id in self.end:
    #             break
    #     return self.sum_tokenizer.decode(output_tokens_list)

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
        encoder_input = self.torch_tensor(encoder_input).unsqueeze(0)

        # 4. Divide the input into tokens and token types
        input_tokens = encoder_input[:, 0, :]
        input_token_types = encoder_input[:, 1, :]

        (output_tokens_list,
         output_token_types_list,
         output_likes) = self.first_k(
            input_tokens=input_tokens,
            input_token_types=input_token_types,
            k=self.k
        )

        for _ in range(self.max_output_length - 1):
            (output_tokens_list,
             output_token_types_list,
             output_likes) = self.search(
                input_tokens=input_tokens,
                input_token_types=input_token_types,
                output_tokens=output_tokens_list,
                output_token_types=output_token_types_list,
                output_likes=output_likes,
                k=self.k
            )

        return self.sum_tokenizer.decode(output_tokens_list[-1])

    def torch_tensor(self,
                     x):
        x = torch.tensor(x, dtype=torch.int64).to(self.device)
        return x

    def first_k(self,
                input_tokens,
                input_token_types,
                k):

        initial_token = self.start
        initial_type = 0
        output_tokens = [[initial_token]]
        output_token_types = [[initial_type]]
        tokens = self.torch_tensor(output_tokens)
        token_types = self.torch_tensor(output_token_types)

        with torch.inference_mode():
            predictions = self.model(input_tokens,
                                     input_token_types,
                                     tokens,
                                     token_types)

            top_k = torch.topk(torch.log_softmax(predictions[:, -1, :],
                                                 dim=-1),
                               k=k,
                               dim=-1)

            top_k_log_soft = top_k.values.squeeze().tolist()
            top_k_ind = top_k.indices.squeeze().tolist()

        temp_output_tokens = []
        temp_output_token_types = []
        temp_output_likes = []

        temp = []

        for i in range(k):
            token = top_k_ind[i]
            temp_output_tokens.append(output_tokens[0] + [token])
            type_ = initial_type
            if (token in self.type_changers
                    and token < self.sum_max_num_sent):
                type_ = initial_type + 1
            temp_output_token_types.append(
                output_token_types[0] + [type_])
            temp_output_likes.append(top_k_log_soft[i])

            temp.append((temp_output_tokens[-1],
                         temp_output_token_types[-1],
                         temp_output_likes[-1]))

        output_tokens = [row[0] for row in temp]
        output_token_types = [row[1] for row in temp]
        output_likes = [row[2] for row in temp]
        return (output_tokens,
                output_token_types,
                output_likes)

    def search(self,
               input_tokens,
               input_token_types,
               output_tokens,
               output_token_types,
               output_likes,
               k):

        tokens = self.torch_tensor(output_tokens)
        token_types = self.torch_tensor(output_token_types)

        input_tokens = input_tokens.repeat((k, 1))
        input_token_types = input_token_types.repeat((k, 1))

        assert tokens.shape[0] == k

        with torch.inference_mode():
            predictions = self.model(input_tokens,
                                     input_token_types,
                                     tokens,
                                     token_types)

            top_k = torch.topk(torch.log_softmax(predictions[:, -1, :],
                                                 dim=-1),
                               k=k,
                               dim=-1)

            top_k_log_soft = top_k.values.squeeze().tolist()
            top_k_ind = top_k.indices.squeeze().tolist()

        temp_output_tokens = []
        temp_output_token_types = []
        temp_output_likes = []

        temp = []

        for i in range(k):
            summary = output_tokens[i]
            summary_types = output_token_types[i]
            like = output_likes[i]
            last_type = summary_types[-1]
            last_token = summary[-1]
            for j in range(k):
                if last_token in self.end:
                    temp_output_tokens.append(summary)
                    temp_output_token_types.append(summary_types)
                    temp_output_likes.append(like)
                else:
                    token = top_k_ind[i][j]
                    temp_output_tokens.append(summary + [token])
                    type_ = last_type
                    if (token in self.type_changers
                            and token < self.sum_max_num_sent):
                        type_ = last_type + 1
                    temp_output_token_types.append(summary_types + [type_])
                    temp_output_likes.append(like + top_k_log_soft[i][j])

                temp.append((temp_output_tokens[-1],
                            temp_output_token_types[-1],
                            temp_output_likes[-1]))

        temp = sorted(temp, key=lambda x: x[-1])

        output_tokens = [row[0] for row in temp[-k:]]
        output_token_types = [row[1] for row in temp[-k:]]
        output_likes = [row[2] for row in temp[-k:]]
        return (output_tokens,
                output_token_types,
                output_likes)
