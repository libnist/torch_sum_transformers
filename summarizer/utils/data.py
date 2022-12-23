# Import libraries
import datasets
from datasets import load_dataset, load_dataset_builder

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from . import tokenizer
import numpy as np
import os

def description(name: str,
                version: str = "3.0.0") -> None:
    """Prints out the description of a given dataset name and version.

    Args:
        name (str): dataset's name (e.g. "cnn_dailymail")
        version (str, optional): version of the dataset. Defaults to "3.0.0".
    """
    builder = load_dataset_builder(path=name, version=version)
    print(builder.info.description)


def get_data(path: str,
             split: str,
             *args, 
             version: str = "3.0.0",
             **kwargs) -> datasets.arrow_dataset.Dataset:
    """Downloads and returns a dataset in a specified version.

    Args:
        path (str): path or name of the dataset.
        split (str): which split to download (e.g. "train", "test", "val")
        version (str, optional): version of the dataset. Defaults to "3.0.0".

    Returns:
        datasets.arrow_dataset.Dataset: A hugging face dataset.
    """
    return load_dataset(path=path,
                        split=split,
                        *args,
                        version=version,
                        **kwargs)


class DocumentSummaryDataset(Dataset):
    def __init__(self,
                 documents: List[str],
                 summaries: List[str],
                 document_tokenizer: tokenizer,
                 summary_tokenizer: tokenizer,
                 document_max_tokens: int,
                 summary_max_tokens: int,
                 device: str) -> torch.utils.data.Dataset:
        """Creates a dataset instance containing document and summaries
        w/ their corresponding type_ids.

        Args:
            documents (List[str]): A list of strings containing documents.
            summaries (List[str]): A list of strings containing summaries.
            document_tokenizer (tokenizer): A tokenizer trained on documents.
            summary_tokenizer (tokenizer): A tokenizer trained on summaries.
            document_max_tokens (int): Maximum number of document tokens to
            return.
            summary_max_tokens (int): Maximum number of summary tokens to
            return. 
            device (str): The device in which token tensors will settle on.
        """
        self._document_max_tokens = document_max_tokens
        self._summary_max_tokens = summary_max_tokens

        error_message = "[ERROR] Shape missmatch of documents and summaries"
        self._len_documents = len(documents)
        self._len_summaries = len(summaries)
        assert self._len_documents == self._len_documents, error_message

        self._documents = documents
        self._summaries = summaries

        self._document_tokenizer = document_tokenizer
        self._summary_tokenizer = summary_tokenizer

        self._fill_token_id = document_tokenizer.token_to_id("[FILL]")

        self._device = device

    def __len__(self) -> int:
        return self._len_documents

    def encode_input(self,
                     input: str,
                     tokenizer: tokenizer,
                     max_tokens: int) -> torch.tensor:
        """Return tokenized input with it's given tokenizer.

        Args:
            input (str): An string input.
            tokenizer (tokenizer): Tokenizer used to tokenize the input.
            max_tokens (int): Maximum number of tokens to return. if the
            tokens length are longer than max_tokens it cuts them, and if
            they're shorter in fills them with [FILL] special token. 

        Returns:
            torch.tensor: A tokenized input.
        """
        encoded = tokenizer.encode(input)
        encoded = np.array(encoded)

        shape = encoded.shape
        if shape[-1] < max_tokens:
            array = np.ones((2, max_tokens)) * self._fill_token_id
            array[:, :shape[-1]] = encoded[:, :shape[-1]]
            encoded = array
        elif shape[-1] > max_tokens:
            encoded = encoded[:, :max_tokens]

        return torch.tensor(encoded, dtype=torch.int64).to(self._device)

    def __getitem__(self,
                    index: int) -> Tuple[torch.tensor, torch.tensor]:
        """Return the document and it's corresponing summary at location
        `index`.

        Args:
            index (int): index of the input document and it's summary.

        Returns:
            Tuple[torch.tensor, torch.tensor]: One Document and it's summary.
        """
        encoded_document = self.encode_input(self._documents[index],
                                             self._document_tokenizer,
                                             self._document_max_tokens)
        
        encoded_summary = self.encode_input(self._summaries[index],
                                            self._summary_tokenizer,
                                            self._summary_max_tokens)
        return encoded_document, encoded_summary


def get_dataloader(documents: List[str],
                   summaries: List[str],
                   document_tokenizer: tokenizer,
                   summary_tokenizer: tokenizer,
                   document_max_tokens: int,
                   summary_max_tokens: int,
                   batch_size: int, device: str,
                   num_workers: int = os.cpu_count(),
                   shuffle: bool = False) -> torch.utils.data.DataLoader:
    """Creates a torch DataLoader from input documents and summaries.

    Args:
        documents (List[str]): A list containing input documents.
        summaries (List[str]): A list containing input summaries.
        document_tokenizer (tokenizer): Tokenizer used to tokenize documents.
        summary_tokenizer (tokenizer): Tokenizer used to tokenize summaries.
        document_max_tokens (int): Maximum number of tokens to return.
        summary_max_tokens (int): Maximum number of tokens to return.
        batch_size (int): Batch size.
        device (str): Device.
        shuffle (bool, optional): Shuffle. Defaults to False.
        num_workers (int, optional): Num workers. Defaults to os.cpu_count().

    Returns:
        torch.utils.data.DataLoader: A torch DataLoader.
    """
    dataset = DocumentSummaryDataset(documents=documents,
                                     summaries=summaries,
                                     document_tokenizer=document_tokenizer,
                                     summary_tokenizer=summary_tokenizer,
                                     document_max_tokens=document_max_tokens,
                                     summary_max_tokens=summary_max_tokens,
                                     device=device)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader
