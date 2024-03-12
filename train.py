"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # Transform image scale, Tensor and normalize
    transform = transforms.Compose([transforms.Resize((1024//4, 2048//4)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    target_transforms = transforms.Compose([transforms.Resize((1024//4, 2048//4),transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

    # data loading for local
    dataset = Cityscapes(root='./data', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transforms)

    # data loading for snellius
    #dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transforms)

    # split dataset in train and val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

    # plot image
    img,label = train_loader.dataset[1]
    plt.imshow(img[0,:,:])
    plt.show()

    # define model
    model = Model()

    # define optimizer and loss function (don't forget to ignore class index 255)
    learning_rate = 0.0001
    num_epochs = 10
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training/validation loop
    for epoch in range(10):
        train_loss = 0.0
        for inputs, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            masks = (masks * 255)
            loss = criterion(outputs, masks.long().squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        epoch_loss_train = train_loss / len(train_loader)

        val_loss = 0.0
        model.eval()
        with torch.inference_mode():
            for inputs, masks in val_loader:
                val_outputs = model(inputs)
                masks = (masks * 255)
                val_loss = criterion(val_outputs, masks.long().squeeze())
                val_loss += val_loss.item()
            epoch_loss_val = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss train: {epoch_loss_train:.4f}, Loss validation: {epoch_loss_val:.4f}')


    # save model
    torch.save(model,'.')

    # visualize some results


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
