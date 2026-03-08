import os
import matplotlib.pyplot as plt

# path to label folder
label_folder = "../datasets/HRIPCB/labels/train"

# class names
classes = [
    "missing hole",
    "mouse bite",
    "open circuit",
    "short",
    "spur",
    "spurious copper"
]

class_counts = [0] * len(classes)

for file in os.listdir(label_folder):
    if file.endswith(".txt"):
        with open(os.path.join(label_folder, file), "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                class_counts[class_id] += 1

# print counts
for i, name in enumerate(classes):
    print(f"{name}: {class_counts[i]}")

# plot distribution
plt.figure(figsize=(8,5))
plt.bar(classes, class_counts)
plt.xticks(rotation=45)
plt.title("PCB Defect Class Distribution")
plt.xlabel("Defect Type")
plt.ylabel("Number of Instances")
plt.tight_layout()

plt.savefig("class_distribution.png", dpi=300)
plt.show()