# Import libraries
import torch

from tqdm.auto import tqdm, trange

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
               device: str,
               log_per_batch: int = 100,
               pretrained: bool = False) -> Tuple[float, float]:
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
        log_per_batch (int, optional): This number specifies how many times
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

    t_range = trange(num_batches)
    t_range.set_description("Train Batch")
    
    # The next step is to train the model with our dataloader.
    for batch, (X, y, z) in zip(t_range, dataloader):
        # Device of the input tensors are already set in the
        # dataloader initialization.

        batch += 1

        # Getting data in the form of the models input

        # Preparing inputs
        inputs = []
        if pretrained:
            # add input docs
            inputs.append(X.to(device))
            # add input sums
            inputs.append(y.to(device))
        else:
            # add input docs
            inputs.append(X[:, 0, :].to(device))
            inputs.append(X[:, 1, :].to(device))
            # add input sums
            inputs.append(y[:, 0, :].to(device))
            inputs.append(y[:, 1, :].to(device))
        z = z.to(device)

        # Forward pass
        # The output is in shape:
        # [batch_size, summary_token_length, sum_vocab_size]
        sum_pred_logits = model(*inputs)

        # Calculating loss
        loss = loss_function(sum_pred_logits, z)
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

        accuracy = accuracy_function(sum_preds, z)
        train_acc += accuracy.item()

        if ((not batch % log_per_batch)
                or (batch == num_batches)):
            loss_to_print = train_loss / batch
            accuracy_to_print = train_acc / batch
            
            t_range.set_postfix(Train_Loss=f"{loss_to_print:8.4f}",
                                Train_Accuracy=f"{accuracy_to_print:8.4f}")
    print()
    train_loss /= num_batches
    train_acc /= num_batches
    return (train_loss, train_acc)


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              accuracy_function: FunctionType,
              device: str,
              pretrained: bool = False) -> Tuple[float, float]:
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
    
    num_batches = len(dataloader)
    t_range = trange(num_batches)
    t_range.set_description("Test Batch:")

    with torch.inference_mode():
        for _, (X, y, z) in zip(t_range, dataloader):
            # Getting the inputs in proper shape

            # Preparing inputs
            inputs = []
            if pretrained:
                # add input docs
                inputs.append(X.to(device))
                # add input sums
                inputs.append(y.to(device))
            else:
                # add input docs
                inputs.append(X[:, 0, :].to(device))
                inputs.append(X[:, 1, :].to(device))
                # add input sums
                inputs.append(y[:, 0, :].to(device))
                inputs.append(y[:, 1, :].to(device))
            z = z.to(device)

            # Forward pass
            # The output is in shape:
            # [batch_size, summary_token_length, sum_vocab_size]
            sum_pred_logits = model(*inputs)

            # Calculating loss
            loss = loss_function(sum_pred_logits, z)
            eval_loss += loss.item()

            # Calculating probs, output_shape:
            # [batch_size, summary_token_length, sum_vocab_size]
            sum_pred_probs = torch.softmax(sum_pred_logits, dim=-1)

            # Calculating prediction tokens, output shpae:
            # [batch_size, summary_token_lenght]
            sum_preds = torch.argmax(sum_pred_probs, dim=-1)

            accuracy = accuracy_function(sum_preds, z)
            eval_acc += accuracy.item()

    eval_loss /= num_batches
    eval_acc /= num_batches
    
    t_range.set_postfix(Test_Loss=f"{eval_loss:8.4f}",
                        Test_Accuracy=f"{eval_acc:8.4f}")
    print()
    return eval_loss, eval_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          accuracy_function: FunctionType,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: str,
          pretrained: bool = False,
          initial_epoch: int = None,
          val_dataloader: torch.utils.data.DataLoader = None,
          lr_scheduler=None,
          path: pathlib.Path = None,
          model_name: str = None,
          log_per_batch: int = 100,
          wandb_config: dict = None,
          wandb_proj: str = None,
          wandb_id: str = None,
          tb_writer: torch.utils.tensorboard.SummaryWriter = None) -> Dict[str, list]:
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
        log_per_batch (int, Optional): Prints models log after each 
        `log_per_batch` time.
        wandb_config (dict, Optional): In case of setting this parameter as
        the training information. it will log our metrics into WandB.

    Returns:
        Dict[str, list]: A dictionary contaning training results.
    """

    # If a path is defined in order for the model to be saved
    # a model_name also should be defined, it will be used to save
    # the model.
    if path:
        assert model_name, "Define model_name parameter."

    # If wandb_config is defined, will be initializing it, so we can
    # make logs in Weights and Biases.
    if wandb_config:
        assert wandb_proj, "Define wandb_proj as the project name."
        if wandb_id:
            wandb.init(id=wandb_id,
                       project=wandb_proj,
                       config=wandb_config,
                       name=model_name,
                       resume="must")
        else:
            wandb.init(project=wandb_proj,
                       config=wandb_config,
                       name=model_name)

    # The dictionary below will containg all the loss and accuracy
    # reports from the training proccess
    results = defaultdict(list)

    # Putting our model into the predefined device
    model.to(device)

    if initial_epoch:
        range_iter = range(initial_epoch, epochs)
    else:
        range_iter = range(epochs)

    # Iterating as many epochs we need and updating our model weights.
    for epoch in range_iter:
        epoch += 1
        print(f"Epoch {epoch} of {epochs}: ", end="")
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           accuracy_function=accuracy_function,
                                           optimizer=optimizer,
                                           log_per_batch=log_per_batch,
                                           device=device,
                                           pretrained=pretrained)

        # Append the results of the current finished epoch.
        results["train_losses"].append(train_loss)
        results["train_accuracies"].append(train_acc)

        # We'll be evaluate our model in case of having an validation
        # dataset.
        if val_dataloader:
            test_loss, test_acc = test_step(model=model,
                                            dataloader=val_dataloader,
                                            loss_function=loss_function,
                                            accuracy_function=accuracy_function,
                                            device=device,
                                            pretrained=pretrained)
            # Append the results of the current finished validation epoch.
            results["val_losses"].append(test_loss)
            results["val_accuracies"].append(test_acc)

        # If there is a learning rate scheduler defined, after each train_step
        # will call it's .step() in order to update our optimizers lr.
        if lr_scheduler:
            lr_scheduler.step()

        # Save the model in case of having a path to save it.
        if path:
            save_model(model=model,
                       path=path,
                       name=model_name,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler)

        # Report our results to wandb
        if wandb_config:
            log = {"train_loss": train_loss,
                   "train_accuracy": train_acc}
            if val_dataloader:
                log.update({"val_loss": test_loss,
                            "val_accuracy": test_acc})
            wandb.log(log, step=epoch, commit=True)
            print(f"{'-'*10}> Wandb Logs are reported!")

        # Report our results to tensorboard
        if tb_writer:
            acc_log = {"train_acc": train_acc}
            loss_log = {"train_loss": train_loss}
            if val_dataloader:
                acc_log.update({"val_acc": test_acc})
                loss_log.update({"val_loss": test_loss})
            tb_writer.add_scalar(main_tag="Loss",
                                 tag_scaler_dict=loss_log,
                                 global_step=epoch)
            tb_writer.add_scalar(main_tag="Accuracy",
                                 tag_scaler_dict=acc_log,
                                 global_step=epoch)
            print(f"{'-'*10}> TensorBoared Logs are reported!")

    if wandb_config:
        wandb.finish()
    if tb_writer:
        tb_writer.close()

    return results
