# Import libraries
import torch

from tqdm.auto import tqdm

from typing import Tuple, Dict
from types import FunctionType

from collections import defaultdict

import pathlib

from .utils.save_and_load import save_model

import wandb

# Train step: Performs one epoch on the training data


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               accuracy_function: FunctionType,
               optimizer: torch.optim.Optimizer,
               log_per_epoch: int = 100) -> Tuple[float, float]:
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
        log_per_epoch (int, optional): This number specifies how many times
        your going to print in-epoch results. Defaults to 100.

    Returns:
        Tuple[float, float]: (Loss, Accuracy)
    """

    # Putting the model in train mode. Activates things like dropout layers.
    model.train()

    # Initializing loss and accuracy values
    train_loss, train_acc = 0, 0

    # Get the number of batches in order to avoid recalculating it each
    # time we want it again.
    num_batches = len(dataloader)

    # The next step is to train the model with our dataloader.
    for batch, (X, y) in enumerate(dataloader):
        # Device of the input tensors are already set in the
        # dataloader initialization.

        batch += 1

        # Getting data in the form of the models input

        # Document inputs
        doc_tokens = X[:, 0, :]  #[batch_size, doc_seq_len]
        doc_token_types = X[:, 1, :]  #[batch_size, doc_seq_len]

        # Summary input
        sum_input_tokens = y[:, 0, :-1]  #[batch_size, sum_seq_len-1]
        sum_input_token_types = y[:, 1, :-1]  #[batch_size, sum_seq_len-1]

        # Summary targets
        sum_target_tokens = y[:, 0, 1:]  #[batch_size, sum_seq_len-1]

        # Forward pass
        # The output is in shape:
        # [batch_size, summary_token_length, sum_vocab_size]
        sum_pred_logits = model(doc_tokens,
                                doc_token_types,
                                sum_input_tokens,
                                sum_input_token_types)

        # Calculating loss
        loss = loss_function(sum_pred_logits, sum_target_tokens)
        train_loss += loss.item()

        # Backward pass and updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating the accuracy of the model

        # Calculating probs, output_shape:
        # [batch_size, summary_token_length, sum_vocab_size]
        sum_pred_probs = torch.softmax(sum_pred_logits, dim=-1)

        # Calculating prediction tokens, output shpae:
        # [batch_size, summary_token_lenght]
        sum_preds = torch.argmax(sum_pred_probs, dim=-1)

        accuracy = accuracy_function(sum_preds, sum_target_tokens)
        train_acc += accuracy.item()

        if ((not batch % log_per_epoch)
            or (batch == num_batches)):
            loss_to_print = train_loss / batch
            accuracy_to_print = train_acc / batch

            print(f"\tBatch {batch} of {num_batches}: " +
                  f"'{batch/num_batches*100:4.2f}%':\t", end="")

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
    eval_loss, eval_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:

            # Getting the inputs in proper shape

            # Document tokens
            doc_tokens = X[:, 0, :]  #[batch_size, document_tokens]
            doc_token_types = X[:, 1, :]  #[batch_size, document_tokens]

            # Summary tokens
            sum_input_tokens = y[:, 0, :-1] #[batch_size, summary_tokens-1]
            sum_input_token_types = y[:, 1, :-1] #[batch_size, summary_tokens-1]
            
            # Summary target token
            sum_target_tokens = y[:, 0, 1:] #[batch_size, summary_tokens-1]

            # Making predictions
            sum_pred_logits = model(doc_tokens,
                                    doc_token_types,
                                    sum_input_tokens,
                                    sum_input_token_types)
            
            # Calculating loss
            loss = loss_function(sum_pred_logits, sum_target_tokens)
            eval_loss += loss.item()

            # Calculating probs, output_shape:
            # [batch_size, summary_token_length, sum_vocab_size]
            sum_pred_probs = torch.softmax(sum_pred_logits, dim=-1)

            # Calculating prediction tokens, output shpae:
            # [batch_size, summary_token_lenght]
            sum_preds = torch.argmax(sum_pred_probs, dim=-1)

            accuracy = accuracy_function(sum_preds, sum_target_tokens)
            eval_acc += accuracy.item()

    num_batches = len(dataloader)
    eval_loss /= num_batches
    eval_acc /= num_batches

    print(f"{'-'*10}>Validation Loss: {eval_loss:4.4f} | " +
          f"Validation Acc: {eval_acc:4.4f}")

    return eval_loss, eval_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          accuracy_function: FunctionType,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: str,
          initial_epoch: int = None,
          val_dataloader: torch.utils.data.DataLoader = None,
          lr_scheduler=None,
          path: pathlib.Path = None,
          model_name: str = None,
          log_per_epoch: int = 100,
          wandb_config: dict = None) -> Dict[str, list]:
    """Performs the whole training procces given the inputs.

    Args:
        model (torch.nn.Module): A summarization model.
        train_dataloader (torch.data.utils.DataLoader): Train dataloader
        which returns documents and summaries in shape:
        [batch_size, 2, sequence_len]
        loss_function (torch.nn.Module): A loss function which returns a 
        torch.tensor.
        accuracy_function (FunctionType): An accuracy function which returns a
        torch.tensor.
        optimizer (torch.optim.Optimzier): Optimizer.
        epochs (int): Number of epochs.
        device (str): The device in which we need to put our tensors in.
        val_dataloader (torch.data.utils.DataLoader, Optional):
        Validation dataloader which returns documents and summaries in shape:
        [batch_size, 2, sequence_len]
        lr_scheduler (torch.optim.lr_scheduler, Optional): Learning rate
        scheduler.
        path (pathlib.Path, Optional): Saves the model, optimizer, and 
        lr_scheduler in the given path.
        model_name (str, Optional): Model will be saved under this name.
        (if path is defined model_name should be defined too.)
        log_per_epoch (int, Optional): Prints models log after each 
        `log_per_epoch` time.
        wandb_config (dict, Optional): In case of setting this parameter as
        the training information. it will log our metrics into WandB.

    Returns:
        Dict[str, list]: A dictionary contaning training results.
    """
    
    # If a path is defined in order for the model to be saved
    # a model_name also should be defined, it will be used to save
    # the model.
    if path:
        assert model_name is not None, "Define model_name parameter."
        
    # If wandb_config is defined, will be initializing it, so we can
    # make logs in Weights and Biases.
    if wandb_config:
        wandb.init(project=model_name,
                   config=wandb_config)

    # The dictionary below will containg all the loss and accuracy
    # reports from the training proccess
    results = defaultdict(list)

    # Putting our model into the predefined device
    model.to(device)
    
    tqdm_iterator = tqdm(range(epochs))
    if initial_epoch:
        tqdm_iterator = tqdm(range(initial_epoch, epochs))

    # Iterating as many epochs we need and updating our model weights.
    for epoch in tqdm_iterator:
        epoch += 1
        print(f"Epoch {epoch} of {epochs}")
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           accuracy_function=accuracy_function,
                                           optimizer=optimizer,
                                           log_per_epoch=log_per_epoch)
        
        # If there is a learning rate scheduler defined, after each train_step
        # will call it's .step() in order to update our optimizers lr.
        if lr_scheduler:
            lr_scheduler.step()
            
        # Append the results of the current finished epoch.
        results["train_losses"].append(train_loss)
        results["train_accuracies"].append(train_acc)
        
        # Save the model in case of having a path to save it.
        if path:
            save_model(model=model,
                       path=path,
                       name=model_name,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler)
            
        # We'll be evaluate our model in case of having an validation
        # dataset.
        if val_dataloader:
            test_loss, test_acc = test_step(model=model,
                                            dataloader=val_dataloader,
                                            loss_function=loss_function,
                                            accuracy_function=accuracy_function)
            # Append the results of the current finished validation epoch.
            results["val_losses"].append(test_loss)
            results["val_accuracies"].append(test_acc)
            
        # Report our results to wandb
        if wandb_config:
            log = {"train_loss": train_loss,
                   "train_accuracy": train_acc}
            if val_dataloader:
                log.update({"val_loss": test_loss,
                            "val_accuracy": test_acc})
            wandb.log(log, step=epoch)

    return results
