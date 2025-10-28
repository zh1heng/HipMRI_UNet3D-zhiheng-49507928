
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from modules import UNet3D
from dataset import HipMRIDataset
from utils import dice_coef, dice_loss

# We will generate loss and dice curves after training.  Import matplotlib here.
import matplotlib.pyplot as plt

image_dir = "D:/RICH/data/HipMRI_study_complete_release_v1/semantic_MRs_anon_train"
mask_dir  = "D:/RICH/data/HipMRI_study_complete_release_v1/semantic_labels_anon_train"

image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
label_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

"""
Initialise the dataset with optional downsampling and patch extraction.

downsample_factor: Stride for subsampling the 3D volumes. A value of 2 halves
each spatial dimension (e.g. 256×256×128 → 128×128×64). This greatly reduces
computational cost and memory footprint.

patch_size: If provided, a random 3D patch of this size is extracted from each
volume during training. Smaller patches (e.g. 96×96×96) further reduce the
amount of data processed per step. For validation/testing, you can set
patch_size to None to process the entire volume.
"""
def main() -> None:
    """Train the 3D UNet model on the HipMRI dataset.

    The training code is wrapped in a function so that DataLoader workers
    spawned on Windows do not re-execute the module-level code. Mixed
    precision training is enabled via torch.amp when running on CUDA.
    """
    # Instantiate dataset with downsampling and patch extraction to reduce computation
    dataset = HipMRIDataset(
        image_paths,
        label_paths,
        downsample_factor=2,        # adjust to 1 for no downsampling
        patch_size=(96, 96, 96),    # set to None to use full volumes
    )

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Training parameters
    num_epochs = 20
    batch_size = 1
    learning_rate = 1e-4

    # DataLoader with multiple workers and pinned memory for faster data transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Use BCEWithLogitsLoss since the model outputs raw logits. This avoids
    # numerical instability with sigmoid+BCELoss in mixed precision.
    criterion = nn.BCEWithLogitsLoss()

    # Use new torch.amp API for mixed precision training when available
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Create lists to record the average loss and dice score per epoch.
    loss_history: list[float] = []
    dice_history: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_dice = 0.0, 0.0
        for i, batch in enumerate(train_loader):
            imgs = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    preds = model(imgs)
                    # BCEWithLogitsLoss expects raw logits; compute BCE on logits
                    loss_bce = criterion(preds, masks)
                    # Convert logits to probabilities for dice computation
                    prob_preds = torch.sigmoid(preds)
                    loss_dice = dice_loss(prob_preds, masks)
                    loss = loss_bce + loss_dice
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs)
                loss_bce = criterion(preds, masks)
                prob_preds = torch.sigmoid(preds)
                loss_dice = dice_loss(prob_preds, masks)
                loss = loss_bce + loss_dice
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_dice += dice_coef(torch.sigmoid(preds), masks)
            # print progress every few iterations
            if i % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}: loss={loss.item():.4f}"
                )
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        # total_dice is a tensor, so divide and then call .item() to get a Python float
        avg_dice_tensor = total_dice / len(train_loader) if len(train_loader) > 0 else torch.tensor(0.0)
        avg_dice = avg_dice_tensor.item() if isinstance(avg_dice_tensor, torch.Tensor) else float(avg_dice_tensor)
        # Append metrics to history lists as plain floats to avoid GPU tensors in plotting
        loss_history.append(float(avg_loss))
        dice_history.append(float(avg_dice))
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")

    # Save model weights
    torch.save(model.state_dict(), "unet_hipmri.pt")
    print("✅ Model training completed and saved!")

    # Plot and save training curves.  Use the recorded histories which are Python floats.
    try:
        epochs_range = range(1, len(loss_history) + 1)
        # Plot loss curve
        plt.figure()
        plt.plot(epochs_range, loss_history, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.savefig("loss_curve.png")
        plt.close()

        # Plot dice coefficient curve
        plt.figure()
        plt.plot(epochs_range, dice_history, label="Dice Coefficient")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Coefficient")
        plt.title("Dice Coefficient over Epochs")
        plt.savefig("dice_curve.png")
        plt.close()
        print("📈 Saved training curves as 'loss_curve.png' and 'dice_curve.png'")
    except Exception as e:
        # If plotting fails, print a warning with the error message
        print(f"Warning: failed to plot training curves due to {e}")


if __name__ == "__main__":
    # On Windows, calling freeze_support() is recommended when spawning
    # subprocesses via the DataLoader. This prevents issues when packaging the
    # script with PyInstaller. It is a no-op on other platforms.
    import multiprocessing as mp

    mp.freeze_support()
    main()
