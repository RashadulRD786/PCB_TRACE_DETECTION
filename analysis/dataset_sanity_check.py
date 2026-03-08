import os

# dataset path
dataset_path = "../datasets/HRIPCB"

splits = ["train", "val", "test"]

num_classes = 6

errors = []

for split in splits:

    images_path = os.path.join(dataset_path, split, "images")
    labels_path = os.path.join(dataset_path, split, "labels")

    image_files = sorted(os.listdir(images_path))
    label_files = sorted(os.listdir(labels_path))

    print(f"\nChecking {split} set")

    # check missing labels
    for img in image_files:

        label_name = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(labels_path, label_name)

        if not os.path.exists(label_path):
           continue

    # check label files
    for label in label_files:

        label_path = os.path.join(labels_path, label)

        with open(label_path, "r") as f:

            lines = f.readlines()

            if len(lines) == 0:
                errors.append(f"Empty label file: {label}")

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    errors.append(f"Invalid label format: {label}")

                else:

                    class_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    if class_id >= num_classes:
                        errors.append(f"Invalid class id in {label}")

                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        errors.append(f"Invalid bbox values in {label}")

print("\nDataset Check Complete")

if len(errors) == 0:
    print("No issues found ✅")
else:
    print(f"{len(errors)} issues found:")
    for e in errors[:20]:
        print(e)