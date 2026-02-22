from ultralytics import YOLO
import argparse

def main(args):
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)

    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()
    main(args)
