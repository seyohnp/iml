from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import random
#import wandb
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Start a new wandb run to track this script.
# run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="cvaiac_LightningMcSpeed",
#     # Set the wandb project where this run will be logged.
#     project="IML_Task3",
#     # Track hyperparameters and run metadata.
#     config={
#         "learning_rate": 0.001,
#         "architecture": "CNN",
#         "dataset": "MNIST",
#         "epochs": 50,
#     },
# )



"""
README FIRST

The below code is a template for the solution. You can change the code according
to your preferences, but the test_model function has to save the output of your 
model on the test data as it does in this template. This output must be submitted.

Replace the dummy code with your own code in the TODO sections.

We also encourage you to use tensorboard or wandb to log the training process
and the performance of your model. This will help you to debug your model and
to understand how it is performing. But the template does not include this
functionality.
Link for wandb:
https://docs.wandb.ai/quickstart/
Link for tensorboard: 
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")

# If you have a Mac consult the following link:
# https://pytorch.org/docs/stable/notes/mps.html

# It is important that your model and all data are on the same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("FROOOOOOOOOOOOOOGS")

def get_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    where N_train_samples is the number of training samples, N_test_samples is
    the number of test samples, C is the number of channels (1 for grayscale),
    H is the height of the image, and W is the width of the image.
    """
    # Load the training data
    train_data = np.load("train_data.npz")["data"]

    # Make the training data a tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)

    # Load the test data
    test_data_input = np.load("test_data.npz")["data"]

    # Make the test data a tensor
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    ########################################
    # TODO: Given the original training images, create the input images and the
    # label images to train your model. 
    # Replace the two placholder lines below (which currently just copy the
    # training data) with your own implementation.
    # img is torch.Size([60000, 1, 28, 28]) 
    # to set the center 8x8 pixels to black, i.e. mask them -> image[10:18, 10:18] = 0
    train_data_label = train_data.clone()
    train_data_input = train_data.clone()
    train_data_input[:, :, 10:18, 10:18] = 0

    #print(train_data_input.shape)
    #dprint(train_data_input)


    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze().cpu().numpy(), cmap="gray")

            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input


def train_model(train_data_input, train_data_label, **kwargs):
    """
    Train the model. Fill in the details of the data loader, the loss function,
    the optimizer, and the training loop.

    Args:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - model: torch.nn.Module
    """
    model = Model()
    model.train()
    model.to(device)

    # TODO: Dummy criterion - change this to the correct loss function
    # https://pytorch.org/docs/stable/nn.html#loss-functions
    #criterion = lambda x, y: torch.mean((x))
    criterion = nn.MSELoss()
    # TODO: Dummy optimizer - change this to a more suitable optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TODO: Correctly setup the dataloader - the below is just a placeholder
    # Also consider that you might not want to use the entire dataset for
    # training alone
    # (batch_size needs to be changed)
    batch_size = 100
    full_dataset = TensorDataset(train_data_input, train_data_label)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # Consider the shuffle parameter and other parameters of the DataLoader
    # class (see
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    # TODO: Modify the training loop in case you need to

    # TODO: The value of n_epochs is just a placeholder and likely needs to be
    # changed
    n_epochs = 50

    for epoch in range(n_epochs):
        
        #training loop
        for x, y in tqdm(
            train_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        train_loss_item = loss.item()
        
        print(f"Epoch {epoch} loss: {train_loss_item}")

        
        #validation loop
        val_loss = 0
        for x, y in tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch} validation loss: {val_loss}")
        
        
        # Log metrics to wandb.
        #run.log({"val_loss": val_loss, "loss": train_loss_item})

    return model


# TODO: define a model. Here, a basic MLP model is defined. You can completely
# change this model - and are encouraged to do so.
class Model(nn.Module):
    """
    Implement your model here.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()


        #Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=1, padding=1) #27x27x16 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #13x13x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1) #12x12x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #6x6x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1) #5x5x64

        
        #Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 5 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5 * 5 * 64),
            nn.Unflatten(1, (64, 5, 5))
        )


        # Decoder
        self.dec1 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0)





    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        # Encoder
        x1 = F.relu(self.conv1(x)) #27x27x16
        
        x2 = self.pool1(x1)
        x2 = F.relu(self.conv2(x2)) #12x12x32

        x3 = self.pool2(x2)
        x3 = F.relu(self.conv3(x3)) #5x5x64


        # Bottleneck
        x = self.bottleneck(x3) #5x5x64


        # Decoder
        x = torch.cat((x, x3), dim=1) #5x5x96
        x = F.relu(self.dec1(x)) #12x12x32

        x = torch.cat((x, x2), dim=1) #12x12x64
        x = F.relu(self.dec2(x)) #27x27x16

        x = self.dec3(x) #28x28x1

        return x


def test_model(model, test_data_input):
    """
    Uses your model to predict the ouputs for the test data. Saves the outputs
    as a binary file. This file needs to be submitted. This function does not
    need to be modified except for setting the batch_size value. If you choose
    to modify it otherwise, please ensure that the generating and saving of the
    output data is not modified.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise to avoid memory issues
        test_data_output = []
        # TODO: You can increase or decrease this batch size depending on your
        # memory requirements of your computer / model
        # This will not affect the performance of the model and your score
        batch_size = 64
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # You don't need to change the code below
    # Load the data
    train_data_input, train_data_label, test_data_input = get_data()
    # Train the model
    model = train_model(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    # The name of the submission file is submit_this_test_data_output.npz
    test_model(model, test_data_input)

    return None


if __name__ == "__main__":
    main()
