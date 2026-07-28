"""Stage 2: jointly optimize PCD-Net with simulated LST labels."""

import argparse
from utils.data_loader import load_lst_data
from utils.model import PCDNet
from utils.training import get_device, load_weights, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/simulation_lst.csv")
    parser.add_argument("--checkpoint-dir", default="models/components")
    parser.add_argument("--output-dir", default="models/simulation_finetuned")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    model = load_weights(PCDNet(), args.checkpoint_dir, get_device())
    train_model(
        model,
        load_lst_data(args.data),
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
