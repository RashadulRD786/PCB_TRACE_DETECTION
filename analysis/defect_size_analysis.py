import os
import numpy as np
import matplotlib.pyplot as plt


label_folder = "../datasets/HRIPCB/labels/train"


image_width = 600
image_height = 600

widths = []
heights = []
areas = []
aspect_ratios = []

for file in os.listdir(label_folder):
    if file.endswith(".txt"):
        with open(os.path.join(label_folder, file), "r") as f:
            for line in f.readlines():
                parts = line.strip().split()

                w = float(parts[3])
                h = float(parts[4])

                w_pixels = w * image_width
                h_pixels = h * image_height

                area = w_pixels * h_pixels
                aspect_ratio = w_pixels / (h_pixels + 1e-6)

                widths.append(w_pixels)
                heights.append(h_pixels)
                areas.append(area)
                aspect_ratios.append(aspect_ratio)

widths = np.array(widths)
heights = np.array(heights)
areas = np.array(areas)
aspect_ratios = np.array(aspect_ratios)

# ---------- Plot 1: Area Distribution ----------
plt.figure(figsize=(6,4))
plt.hist(areas, bins=50)
plt.title("Defect Area Distribution")
plt.xlabel("Bounding Box Area (pixels²)")
plt.ylabel("Frequency")
plt.savefig("area_distribution.png", dpi=300)
plt.show()

# ---------- Plot 2: Width vs Height Scatter ----------
plt.figure(figsize=(6,6))
plt.scatter(widths, heights, alpha=0.3)
plt.title("Defect Width vs Height")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.savefig("width_height_scatter.png", dpi=300)
plt.show()

# ---------- Plot 3: Aspect Ratio Distribution ----------
plt.figure(figsize=(6,4))
plt.hist(aspect_ratios, bins=50)
plt.title("Aspect Ratio Distribution")
plt.xlabel("Width / Height")
plt.ylabel("Frequency")
plt.savefig("aspect_ratio_distribution.png", dpi=300)
plt.show()

# ---------- Small Object Statistics ----------
small = np.sum(areas < 32*32)
medium = np.sum((areas >= 32*32) & (areas < 96*96))
large = np.sum(areas >= 96*96)

print("Total objects:", len(areas))
print("Small:", small)
print("Medium:", medium)
print("Large:", large)