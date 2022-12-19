import torch
from torch import nn

from tqdm.auto import tqdm

from typing import Tuple, Dict
from types import FunctionType

from collections import defaultdict

import pathlib

from .components.save_and_load import save_model

# Train step


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               accuracy_function: FunctionType,
               optimizer: torch.optim.Optimizer,
               batch_verbose: int = 100) -> Tuple[float, float]:
    """Performs one epoch.

    Args:
        model (torch.nn.Module): A summarization model.
        dataloader (torch.data.utils.DataLoader): A torch.utils.data.DataLoader
        which return data in shape: [batch_size, 2, sequence_len] for documents
        and summaries.
        loss_functionn (torch.nn.Module): A loss function which returns 
        a torch.tensor.
        accuracy_function (FunctionType): An accuracy function which returns
        a torch.tensor.
        optimizer (torch.optim.Optimizers): Optimizer.
        batch_verbose (int, optional): This number specifies how many times
        your going to print in-epoch results. Defaults to 100.

    Returns:
        Tuple[float, float]: (Loss, Accuracy)
    """

    # Putting the model in train mode
    model.train()

    # Initializing loss and accuracy values
    train_loss, train_acc = 0, 0

    num_batches = len(dataloader)

    # device of the dataloader is already set.
    for batch, (X, y) in enumerate(dataloader):

        # Getting data in the form of the models input
        # Device is set w/ dataset initialization

        # Document input
        doc_tokens = X[:, 0, :]  # [batch_size, doc_seq_len]
        doc_token_types = X[:, 1, :]  # [batch_size, doc_seq_len]

        # Summary input
        sum_input_tokens = y[:, 0, :-1]  # [batch_size, sum_seq_len]
        sum_input_token_types = y[:, 1, :-1]  # [batch_size, sum_seq_len]

        # Summary targets
        sum_target_tokens = y[:, 0, 1:]  # [batch_size, sum_seq_len]

        # Forward pass
        # The output is in shape:
        # [batch_size, summary_token_length, sum_vocab_size]
        sum_pred_outputs = model(doc_tokens,
                                 doc_token_types,
                                 sum_input_tokens,
                                 sum_input_token_types)

        # Calculating probs, output_shape:
        # [batch_size, summary_token_length, sum_vocab_size]
        sum_pred_probs = torch.softmax(sum_pred_outputs, dim=-1)

        # Calculating prediction tokens, output shpae:
        # [batch_size, summary_token_lenght]
        sum_preds = torch.argmax(sum_pred_probs, dim=-1)

        # Calculating loss
        loss = loss_function(sum_pred_outputs, sum_target_tokens)
        train_loss += loss.item()

        # Bacward pass and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = accuracy_function(sum_preds, sum_target_tokens)
        train_acc += accuracy.item()

        if (((not batch % batch_verbose) and batch != 0) 
            or (batch == num_batches)):
            loss_to_print = train_loss / batch
            accuracy_to_print = train_acc / batch
            
            print(f"\tBatch {batch} of {num_batches}" +
                  f"'{batch/num_batches*100:.4.2f}%':\t", end="")
            
            print(f"Train Loss: {loss_to_print:8.4f} |  " + 
                  f"Train Accuracy: {accuracy_to_print:8.4f}")

    train_loss /= num_batches
    train_acc /= num_batches
    return (train_loss, train_acc)


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              accuracy_function: FunctionType) -> Tuple[float, float]:
    """Performs one validation step on validation or test dataset.

    Args:
        model (torch.nn.Module): A summarization model.
        dataloader (torch.data.utils.DataLoader): A torch.data.utils.DataLoader
        which return data in shape: [batch_size, 2, sequence len] for documents
        and summaries.
        loss_function (torch.nn.Module): A loss function which returns a 
        torch.tensor.
        accuracy_function (FunctionType): An accuracy function which returns a
        torch.tensor.

    Returns:
        Tuple[float, float]: (Loss, Accuracy)
    """

    # Putting the model in eval mode
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            # Getting the inputs in proper shape

            # document tokens
            doc_tokens = X[:, 0, :]  # [batch_size, document_tokens]
            doc_token_types = X[:, 1, :]  # [batch_size, document_tokens]

            # summary tokens
            sum_input_tokens = y[:, 0, :-1]
            sum_input_token_types = y[:, 1, :-1]

            sum_target_tokens = y[:, 0, 1:]

            # Forward pass
            sum_pred_outputs = model(doc_tokens,
                                     doc_token_types,
                                     sum_input_tokens,
                                     sum_input_token_types)

            # Calculating probs, output_shape:
            # [batch_size, summary_token_length, sum_vocab_size]
            sum_pred_probs = torch.softmax(sum_pred_outputs, dim=-1)

            # Calculating prediction tokens, output shpae:
            # [batch_size, summary_token_lenght]
            sum_preds = torch.argmax(sum_pred_probs, dim=-1)

            # Calculating loss
            loss = loss_function(sum_pred_outputs, sum_target_tokens)
            test_loss += loss.item()

            accuracy = accuracy_function(sum_preds, sum_target_tokens)
            test_acc += accuracy.item()
    
    num_batches = len(dataloader)
    test_loss /= num_batches
    test_acc /= num_batches

    print(f"{'-'*10}>Validation Loss: {test_loss:4.4f} | " + 
          f"Validation Acc: {test_acc:4.4f}")

    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          accuracy_function: FunctionType,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: str,
          lr_scheduler = None,
          path: pathlib.Path = None ,
          model_name: str = None) -> Dict[str, list]:
    """Performs the whole training procces given the inputs.

    Args:
        model (torch.nn.Module): A summarization model.
        train_dataloader (torch.data.utils.DataLoader): Train dataloader
        which returns documents and summaries in shape:
        [batch_size, 2, sequence_len]
        test_dataloader (torch.data.utils.DataLoader): Validation dataloader
        which returns documents and summaries in shape:
        [batch_size, 2, sequence_len]
        loss_function (torch.nn.Module): A loss function which returns a 
        torch.tensor.
        accuracy_function (FunctionType): An accuracy function which returns a
        torch.tensor.
        optimizer (torch.optim.Optimzier): Optimizer.
        epochs (int): Number of epochs.
        device (str): The device in which we need to put our tensors in.

    Returns:
        Dict[str, list]: A dictionary contaning training results.
    """
    
    if path:
        assert model_name is not None, "Define model_name parameter."
    
    # The dictionary below will containg all the loss and accuracy
    # reports from the training proccess
    results = defaultdict(list)
    
    # Putting our model into the predefined device
    model.to(device)
    
    # Iterating as many epochs we need and updating our model weights.
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch+1} of {epochs}")
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           accuracy_function=accuracy_function,
                                           optimizer=optimizer)
        
        results["train_losses"].append(train_loss)
        results["train_accuracies"].append(train_acc)
        
        if path:
            save_model(model=model,
                       path=path,
                       name=model_name,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_function=loss_function,
                                        accuracy_function=accuracy_function)
        
        results["test_losses"].append(test_loss)
        results["test_accuracies"].append(test_acc)
        
    return results
        
        
