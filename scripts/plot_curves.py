import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics(csv_path, title="Training Curves"):
    
    df = pd.read_csv(csv_path)

    
    df.columns = df.columns.str.strip()

    
    print("Available columns:")
    print(df.columns.tolist())

    epochs = df['epoch']

    plt.figure(figsize=(12, 8))

    
    if 'metrics/mAP50(B)' in df.columns:
        plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP50')

    if 'metrics/mAP50-95(B)' in df.columns:
        plt.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95')


    if 'metrics/precision(B)' in df.columns:
        plt.plot(epochs, df['metrics/precision(B)'], label='Precision')

    if 'metrics/recall(B)' in df.columns:
        plt.plot(epochs, df['metrics/recall(B)'], label='Recall')

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_path = Path(csv_path).parent / "training_curves.png"
    plt.savefig(save_path)
    plt.show()

    print(f"\nPlot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--title", type=str, default="Training Curves")
    args = parser.parse_args()

    plot_metrics(args.csv, args.title)


if __name__ == "__main__":
    main()