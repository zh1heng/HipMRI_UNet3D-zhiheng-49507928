import os
import argparse
import torch
from torch.utils.data import DataLoader
from modules import UNet3D
from dataset import HipMRIDataset
from utils import dice_coef

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    # 加载测试集
    test_images = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)])
    test_masks  = sorted([os.path.join(args.mask_dir, f) for f in os.listdir(args.mask_dir)])
    dataset = HipMRIDataset(test_images, test_masks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Collect dice coefficients for each sample
    dices: list[float] = []
    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        preds = torch.sigmoid(model(imgs))
        dice_val = dice_coef(preds, masks).item()
        dices.append(dice_val)
        print(f"[{i+1}/{len(loader)}] Dice = {dice_val:.4f}")

    # Compute mean dice
    mean_dice = sum(dices) / len(dices) if len(dices) > 0 else 0.0
    print("\n✅ Dice scores per sample:")
    for idx, val in enumerate(dices, 1):
        print(f"  Sample {idx}: {val:.4f}")
    print(f"\n✅ Final Mean Dice: {mean_dice:.4f}")

    # Optionally save dice scores to a text file
    try:
        with open("dice_scores.txt", "w") as f:
            for val in dices:
                f.write(f"{val}\n")
            f.write(f"Mean: {mean_dice}\n")
        print("📄 Saved per-sample dice scores to 'dice_scores.txt'")
    except Exception as e:
        print(f"Warning: failed to save dice scores due to {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir",  required=True)
    parser.add_argument("--model_path", default="unet_hipmri.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
