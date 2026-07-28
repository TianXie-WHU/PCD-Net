"""Stage 1: initialize the four component branches with simulation labels."""

import argparse
from utils.data_loader import load_component_data
from utils.model import PCDNet
from utils.training import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/simulation_components.csv")
    parser.add_argument("--output-dir", default="models/components")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    train_model(
        PCDNet(),
        load_component_data(args.data),
        args.output_dir,
        component_training=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-5,
        residual_learning_rate=1e-3,
    )


if __name__ == "__main__":
    main()
