from ultralytics import YOLO
import argparse
import yaml
import sys
import os

def load_config(config_path, name):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})
    experiments = cfg.get("experiments", {})

    if name not in experiments:
        print(f"Error: experiment '{name}' not found in config.")
        print("Available experiments:")
        for exp in experiments:
            print(f"  {exp}")
        sys.exit(1)

    # Merge defaults with experiment-specific overrides
    params = {**defaults, **experiments[name]}
    return params

def list_experiments(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print("Available experiments:")
    for name, vals in cfg.get("experiments", {}).items():
        model = vals.get("model", cfg["defaults"].get("model", ""))
        epochs = vals.get("epochs", cfg["defaults"].get("epochs", ""))
        imgsz = vals.get("imgsz", cfg["defaults"].get("imgsz", ""))
        print(f"  {name:<35} model={model:<12} epochs={epochs:<5} imgsz={imgsz}")

def main(args):
    # Resolve parameters: config file takes base values, CLI flags override
    if args.config:
        params = load_config(args.config, args.name)
    else:
        params = {}

    model_path  = args.model   or params.get("model")
    data        = args.data    or params.get("data")
    epochs      = args.epochs  or params.get("epochs", 100)
    imgsz       = args.imgsz   or params.get("imgsz", 640)
    batch       = args.batch   or params.get("batch", 16)
    patience    = args.patience or params.get("patience", 20)
    device      = params.get("device", 0)
    project     = args.project or params.get("project", "runs/experiments")
    workers     = params.get("workers", 8)
    amp         = params.get("amp", True)
    seed        = params.get("seed", 0)

    if not model_path:
        print("Error: --model is required when not using --config.")
        sys.exit(1)
    if not data:
        print("Error: --data is required when not using --config.")
        sys.exit(1)

    # Resolve data path relative to repo root when using config
    if args.config and not os.path.isabs(data):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
        data = os.path.join(repo_root, data)

    print(f"\nStarting experiment: {args.name}")
    print(f"  model   : {model_path}")
    print(f"  data    : {data}")
    print(f"  epochs  : {epochs}")
    print(f"  imgsz   : {imgsz}")
    print(f"  batch   : {batch}")
    print(f"  patience: {patience}\n")

    model = YOLO(model_path)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=project,
        name=args.name,
        workers=workers,
        amp=amp,
        seed=seed,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a YOLO model for PCB defect detection.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run a named experiment from config\n"
            "  python train.py --config ../configs/train_config.yaml --name yolov8x_640_100_epochs\n\n"
            "  # Override epochs on the fly\n"
            "  python train.py --config ../configs/train_config.yaml --name yolov8x_640_100_epochs --epochs 50\n\n"
            "  # List all experiments in the config\n"
            "  python train.py --config ../configs/train_config.yaml --list\n\n"
            "  # Run without a config file (original behaviour)\n"
            "  python train.py --model yolov8s.pt --data ../datasets/HRIPCB/data.yaml --name my_run\n"
        )
    )

    parser.add_argument("--config",   type=str, default=None,  help="Path to train_config.yaml")
    parser.add_argument("--name",     type=str, default=None,  help="Experiment name (required unless --list)")
    parser.add_argument("--list",     action="store_true",     help="List available experiments in the config and exit")
    parser.add_argument("--model",    type=str, default=None,  help="Model weights (overrides config)")
    parser.add_argument("--data",     type=str, default=None,  help="Path to data.yaml (overrides config)")
    parser.add_argument("--epochs",   type=int, default=None,  help="Number of epochs (overrides config)")
    parser.add_argument("--imgsz",    type=int, default=None,  help="Image size (overrides config)")
    parser.add_argument("--batch",    type=int, default=None,  help="Batch size (overrides config)")
    parser.add_argument("--patience", type=int, default=None,  help="Early stopping patience (overrides config)")
    parser.add_argument("--project",  type=str, default=None,  help="Output project directory (overrides config)")

    args = parser.parse_args()

    if args.list:
        if not args.config:
            print("Error: --list requires --config.")
            sys.exit(1)
        list_experiments(args.config)
        sys.exit(0)

    if not args.name:
        parser.error("--name is required (use --list to see available experiment names).")

    main(args)
