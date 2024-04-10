"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
import utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import Model
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import wandb


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    scale = 4
    # Transform image scale, Tensor and normalize
    transform = transforms.Compose([transforms.Resize((1024//scale, 2048//scale)), transforms.ToTensor(), transforms.Normalize((0.2869, 0.3251, 0.2839), (0.1870, 0.1902, 0.1872))])
    target_transforms = transforms.Compose([transforms.Resize((1024//scale, 2048//scale),transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

    # data loading for snellius
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform, target_transform=target_transforms)

    # split dataset in train and val
    train_indices = range(0,int(0.8 * len(dataset)))
    val_indices = range(int(0.8 * len(dataset)),len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=12, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4,num_workers=12, shuffle=True)

    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    learning_rate = 0.01
    num_epochs = 30
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_classes = 19

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="segmentation_project",
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "dataset": "Cityscapes",
        "epochs": num_epochs,
        }
    )

    def dice_coefficient(y_true, y_pred, num_classes):
        """
        Function to calculate the Dice Coefficient for each class in a multi-class segmentation task.

        Parameters:
        y_true: numpy.array, true segmentation mask, where each pixel has an integer value representing the class
        y_pred: numpy.array, predicted segmentation mask, with the same dimensions and class representation as y_true
        num_classes: int, number of classes in the segmentation task

        Returns:
        dice_scores: list, Dice Coefficients for each class
        mean_dice_score: float, mean Dice Coefficient over all classes
        """
        
        dice_scores = []
        
        for class_id in range(num_classes):
            # Calculate intersection and union for the current class
            true_class = y_true == class_id
            pred_class = y_pred == class_id
            intersection = np.logical_and(true_class, pred_class)
            union = np.logical_or(true_class, pred_class)
            
            # Calculate Dice score for the current class
            if union.sum() == 0:  # to handle division by zero if there's no ground truth and no prediction for a class
                dice_score = 1.0 if intersection.sum() == 0 else 0.0
            else:
                dice_score = (2. * intersection.sum()) / (true_class.sum() + pred_class.sum())
            
            dice_scores.append(dice_score)
        
        # Calculate mean Dice score across all classes
        mean_dice_score = np.mean(dice_scores)
        
        return dice_scores, mean_dice_score

    # training/validation loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        dice_scores_epoch = []  # List to store dice scores for each batch
        val_loss = 0.0
        val_dice_scores_epoch = []
        
        model.train()
        for inputs, masks in train_loader:
            inputs = inputs.cuda()
            masks = masks.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            masks = (masks*255).squeeze(1).long()
            masks = utils.map_id_to_train_id(masks)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Convert model output to class predictions
            _, predicted_masks = torch.max(outputs, 1)
            dice_scores_batch, _ = dice_coefficient(masks.cpu().numpy(), predicted_masks.cpu().numpy(), num_classes)
            dice_scores_epoch.extend(dice_scores_batch)
        epoch_loss_train = train_loss / len(train_loader)
        mean_dice_score_epoch = np.mean(dice_scores_epoch)

        model.eval()
        with torch.inference_mode():
            for inputs, masks in val_loader:
                inputs = inputs.cuda()
                masks = masks.cuda()
                val_outputs = model(inputs)
                masks = (masks*255).squeeze(1).long()
                masks = utils.map_id_to_train_id(masks)
                val_loss_batch = criterion(val_outputs, masks)
                
                val_loss += val_loss_batch.item()

                _, val_predicted_masks = torch.max(val_outputs, 1)
                val_dice_scores_batch, _ = dice_coefficient(masks.cpu().numpy(), val_predicted_masks.cpu().numpy(), num_classes)
                val_dice_scores_epoch.extend(val_dice_scores_batch)
            


        # Calculate mean loss and mean dice score for the epoch
        epoch_loss_val = val_loss / len(val_loader)
        val_mean_dice_score_epoch = np.mean(val_dice_scores_epoch)  

        # Logging the metrics for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss train: {epoch_loss_train:.4f}, Loss val: {epoch_loss_val:.4f}')
        wandb.log({"loss_train": epoch_loss_train,"loss_val": epoch_loss_val,"Mean Dice Score train":mean_dice_score_epoch,"Mean Dice Score val":val_mean_dice_score_epoch})
        torch.save(model.state_dict(),'./first_model.pth')
    wandb.finish()

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
