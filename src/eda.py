import json
import os
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import *


def main() -> None:
    np.random.seed(42)

    os.chdir("..")
    # Set dataset paths
    ROOT_DIR = os.getcwd()
    print(ROOT_DIR)

    LABEL_DIR = os.path.join(
        ROOT_DIR, "data", "bdd100k_labels_release", "bdd100k", "labels"
    )
    IMAGE_DIR = os.path.join(
        ROOT_DIR, "data", "bdd100k_images_100k", "bdd100k", "images", "100k"
    )

    # --- Paths (assuming ROOT_DIR is set) ---
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    IMAGE_DIR_TRAIN = os.path.join(IMAGE_DIR, "train")
    IMAGE_DIR_VAL = os.path.join(IMAGE_DIR, "val")

    # Train
    with open(
        os.path.join(LABEL_DIR, "bdd100k_labels_images_train.json"), encoding="utf-8"
    ) as f:
        train_labels = json.load(f)

    # Validation
    with open(
        os.path.join(LABEL_DIR, "bdd100k_labels_images_val.json"), encoding="utf-8"
    ) as f:
        val_labels = json.load(f)

    all_train_images = [ann["name"] for ann in train_labels]
    all_val_images = [ann["name"] for ann in val_labels]

    # Pick a random train image and save it
    sample_annotation = np.random.choice(train_labels)
    sample_save_path = os.path.join(RESULTS_DIR, "sample_image_train.jpg")
    img = show_sample_with_boxes(
        sample_annotation, IMAGE_DIR, split="train", save_path=sample_save_path
    )

    print(f"Sample image saved to {sample_save_path}")

    # Pick a random train image and save it
    sample_annotation = np.random.choice(val_labels)
    sample_save_path = os.path.join(RESULTS_DIR, "sample_image_val.jpg")
    img = show_sample_with_boxes(
        sample_annotation, IMAGE_DIR, split="val", save_path=sample_save_path
    )

    print(f"Sample image saved to {sample_save_path}")

    # --- Sample a subset of images for fast stats ---
    sample_train_images = np.random.choice(
        all_train_images, size=min(1000, len(all_train_images)), replace=False
    )
    sample_val_images = np.random.choice(
        all_val_images, size=min(1000, len(all_val_images)), replace=False
    )

    # Width and Heights Analysis
    # train data
    widths, heights = [], []

    for img_name in sample_train_images:
        img_path = os.path.join(IMAGE_DIR_TRAIN, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    # --- Compute statistics ---
    stats = {
        "width_mean": widths.mean(),
        "width_std": widths.std(),
        "width_min": widths.min(),
        "width_max": widths.max(),
        "height_mean": heights.mean(),
        "height_std": heights.std(),
        "height_min": heights.min(),
        "height_max": heights.max(),
    }

    print("Image Dimension Stats:", stats)
    print(f"All images have the same size: {widths[0]} x {heights[0]}")

    aspect_ratios = widths / heights

    plt.figure(figsize=(8, 4))
    plt.hist(aspect_ratios, bins=10, color="skyblue", edgecolor="black")
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Width / Height")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "aspect_ratio_hist.png"))

    # Val data

    widths, heights = [], []

    for img_name in sample_val_images:
        img_path = os.path.join(IMAGE_DIR_VAL, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    # --- Compute statistics ---
    stats = {
        "width_mean": widths.mean(),
        "width_std": widths.std(),
        "width_min": widths.min(),
        "width_max": widths.max(),
        "height_mean": heights.mean(),
        "height_std": heights.std(),
        "height_min": heights.min(),
        "height_max": heights.max(),
    }

    print("Image Dimension Stats:", stats)
    print(f"All images have the same size: {widths[0]} x {heights[0]}")

    aspect_ratios = widths / heights

    plt.figure(figsize=(8, 4))
    plt.hist(aspect_ratios, bins=10, color="skyblue", edgecolor="black")
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Width / Height")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "aspect_ratio_hist_val.png"))

    # --- Sample image grid ---
    plt.figure(figsize=(12, 12))
    grid_sample = np.random.choice(
        all_train_images, size=min(12, len(all_train_images)), replace=False
    )
    for i, img_name in enumerate(grid_sample):
        img = Image.open(os.path.join(IMAGE_DIR_TRAIN, img_name))
        plt.subplot(4, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_images_grid.png"))

    # --- Sample image grid ---
    plt.figure(figsize=(12, 12))
    grid_sample = np.random.choice(
        all_val_images, size=min(12, len(all_train_images)), replace=False
    )
    for i, img_name in enumerate(grid_sample):
        img = Image.open(os.path.join(IMAGE_DIR_VAL, img_name))
        plt.subplot(4, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_images_grid_val.png"))

    # Sample subset of images

    sample_size = min(1000, len(all_train_images))
    sample_images = np.random.choice(all_train_images, size=sample_size, replace=False)

    brightness_list = []
    contrast_list = []

    # --- Compute brightness and contrast ---
    for img_name in sample_images:
        img_path = os.path.join(IMAGE_DIR_TRAIN, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img) / 255.0  # normalize 0-1
            # Brightness as mean intensity
            brightness = img_array.mean()
            # Contrast as std of intensity
            contrast = img_array.std()
            brightness_list.append(brightness)
            contrast_list.append(contrast)

    brightness_array = np.array(brightness_list)
    contrast_array = np.array(contrast_list)

    # --- Statistics ---
    stats = {
        "brightness_mean": brightness_array.mean(),
        "brightness_std": brightness_array.std(),
        "contrast_mean": contrast_array.mean(),
        "contrast_std": contrast_array.std(),
    }

    print("Brightness & Contrast Stats:", stats)

    # --- Plot Brightness Histogram ---
    plt.figure(figsize=(10, 5))
    plt.hist(brightness_array, bins=30, color="gold", alpha=0.7)
    plt.title("Brightness Distribution")
    plt.xlabel("Brightness (mean pixel intensity)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "brightness_hist.png"))

    # --- Plot Contrast Histogram ---
    plt.figure(figsize=(10, 5))
    plt.hist(contrast_array, bins=30, color="skyblue", alpha=0.7)
    plt.title("Contrast Distribution")
    plt.xlabel("Contrast (std of pixel intensity)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "contrast_hist.png"))

    # Sample subset of images

    sample_size = min(1000, len(all_val_images))
    sample_images = np.random.choice(all_val_images, size=sample_size, replace=False)

    brightness_list = []
    contrast_list = []

    # --- Compute brightness and contrast ---
    for img_name in sample_images:
        img_path = os.path.join(IMAGE_DIR_VAL, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img) / 255.0  # normalize 0-1
            # Brightness as mean intensity
            brightness = img_array.mean()
            # Contrast as std of intensity
            contrast = img_array.std()
            brightness_list.append(brightness)
            contrast_list.append(contrast)

    brightness_array = np.array(brightness_list)
    contrast_array = np.array(contrast_list)

    # --- Statistics ---
    stats = {
        "brightness_mean": brightness_array.mean(),
        "brightness_std": brightness_array.std(),
        "contrast_mean": contrast_array.mean(),
        "contrast_std": contrast_array.std(),
    }

    print("Brightness & Contrast Stats:", stats)

    # --- Plot Brightness Histogram ---
    plt.figure(figsize=(10, 5))
    plt.hist(brightness_array, bins=30, color="gold", alpha=0.7)
    plt.title("Brightness Distribution")
    plt.xlabel("Brightness (mean pixel intensity)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "brightness_hist_val.png"))

    # --- Plot Contrast Histogram ---
    plt.figure(figsize=(10, 5))
    plt.hist(contrast_array, bins=30, color="skyblue", alpha=0.7)
    plt.title("Contrast Distribution")
    plt.xlabel("Contrast (std of pixel intensity)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "contrast_hist_val.png"))

    # --- Arrays to store RGB means ---
    rgb_means = []
    sample_images = np.random.choice(
        all_train_images, size=min(1000, len(all_train_images)), replace=False
    )
    for img_name in sample_images:
        img_path = os.path.join(IMAGE_DIR_TRAIN, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img) / 255.0  # normalize to 0-1
            mean_rgb = arr.mean(axis=(0, 1))
            rgb_means.append(mean_rgb)

    rgb_means = np.array(rgb_means)
    rgb_mean = rgb_means.mean(axis=0)
    rgb_std = rgb_means.std(axis=0)

    print("RGB Mean:", rgb_mean)
    print("RGB Std:", rgb_std)

    # --- Plot RGB means ---
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Red", "Green", "Blue"],
        rgb_mean,
        yerr=rgb_std,
        color=["r", "g", "b"],
        alpha=0.7,
    )
    plt.ylabel("Mean Intensity")
    plt.title("RGB Mean ± Std per Channel")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rgb_mean_plot.png"))

    # --- Arrays to store RGB means ---
    rgb_means = []
    sample_images = np.random.choice(
        all_val_images, size=min(1000, len(all_train_images)), replace=False
    )
    for img_name in sample_images:
        img_path = os.path.join(IMAGE_DIR_VAL, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img) / 255.0  # normalize to 0-1
            mean_rgb = arr.mean(axis=(0, 1))
            rgb_means.append(mean_rgb)

    rgb_means = np.array(rgb_means)
    rgb_mean = rgb_means.mean(axis=0)
    rgb_std = rgb_means.std(axis=0)

    print("RGB Mean:", rgb_mean)
    print("RGB Std:", rgb_std)

    # --- Plot RGB means ---
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Red", "Green", "Blue"],
        rgb_mean,
        yerr=rgb_std,
        color=["r", "g", "b"],
        alpha=0.7,
    )
    plt.ylabel("Mean Intensity")
    plt.title("RGB Mean ± Std per Channel")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rgb_mean_plot_val.png"))

    # --- Preprocess train and val ---

    train_stats = preprocess_annotations(train_labels)
    val_stats = preprocess_annotations(val_labels)

    save_extreme_boxes(
        all_train_images,
        train_labels,
        IMAGE_DIR_TRAIN,
        save_dir="results/extreme_boxes",
        num_examples=5,
    )

    # Sort annotations by object count descending

    top5_idx = np.argsort(-train_stats["objects_per_image"])[:5]

    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(top5_idx):
        ann = train_labels[idx]
        img_path = os.path.join(IMAGE_DIR_TRAIN, ann["name"])
        with Image.open(img_path) as img:
            draw = ImageDraw.Draw(img)
            for obj in ann.get("labels", []):
                box = obj.get("box2d")
                if box:
                    draw.rectangle(
                        [box["x1"], box["y1"], box["x2"], box["y2"]],
                        outline="red",
                        width=2,
                    )

            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(
                f"{ann['name']}\nObjects: {train_stats['objects_per_image'][idx]}",
                fontsize=10,
            )

    plt.tight_layout()

    # ✅ Save the figure
    output_path = "results/top5_crowded_images_train.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Top 5 crowded images saved to {output_path}")

    timeofday_counter = train_stats["timeofday_counter"]

    total_images = sum(timeofday_counter.values())

    timeofday_table = []
    for time in ["daytime", "night", "dawn/dusk"]:
        count = timeofday_counter.get(time, 0)
        pct = (count / total_images) * 100 if total_images > 0 else 0
        timeofday_table.append((time.capitalize(), count, f"{pct:.1f}%"))

    # Print table
    print("| Time | Count | % |")
    print("|:-----|------:|--:|")
    for row in timeofday_table:
        print(f"| {row[0]} | {row[1]} | {row[2]} |")

    weather_counter = train_stats["weather_counter"]

    total_images = sum(weather_counter.values())

    weather_table = []
    for weather in ["clear", "rainy", "foggy", "snowy"]:
        count = weather_counter.get(weather, 0)
        pct = (count / total_images) * 100 if total_images > 0 else 0
        weather_table.append((weather.title(), count, f"{pct:.1f}%"))

    print("### 5.2 Weather Conditions")
    print("![Weather Distribution](results/weather_distribution_train.png)\n")
    print("| Weather | Count | % |")
    print("|:--------|------:|--:|")
    for row in weather_table:
        print(f"| {row[0]} | {row[1]} | {row[2]} |")

    # --- 5.3 Scene Types ---
    scene_counter = train_stats["scene_counter"]
    total_images = sum(scene_counter.values())

    scene_table = []
    for scene in ["city street", "highway", "residential"]:
        count = scene_counter.get(scene, 0)
        pct = (count / total_images) * 100 if total_images > 0 else 0
        scene_table.append((scene.title(), count, f"{pct:.1f}%"))

    print("### 5.3 Scene Types")
    print("![Scene Distribution](results/scene_distribution.png)\n")
    print("| Scene | Count | % |")
    print("|:------|------:|--:|")
    for row in scene_table:
        print(f"| {row[0]} | {row[1]} | {row[2]} |")

    print("\n---\n")

    # --- 5.4 Scene-Object Correlation ---
    scene_object_counter = train_stats["scene_object_counter"]

    print("### 5.4 Correlation between Scene & Object Types")
    print("![Scene-Object Correlation Heatmap](results/scene_object_corr.png)\n")
    print("> _Interpret patterns (e.g., trucks mainly appear on highways)._")
    print("\n---\n")

    # --- 6. Train-Test Split Validation ---
    train_images_count = len(train_labels)
    val_images_count = len(val_labels)
    # Sum total objects
    train_objects_count = sum(train_stats["objects_per_image"])
    val_objects_count = sum(val_stats["objects_per_image"])

    print("## 6. 🔍 Train-Test Split Validation\n")
    print("| Split | #Images | #Objects | Note |")
    print("|:------|---------:|---------:|:-----|")
    print(f"| Train | {train_images_count} | {train_objects_count} |  |")
    print(f"| Val   | {val_images_count} | {val_objects_count} |  |")

    # Example: t-SNE on first 200 images (can increase later)

    train_img_paths = [ann["name"] for ann in train_labels][:200]
    features = extract_image_features(train_img_paths, IMAGE_DIR_TRAIN)

    # PCA for initial dimensionality reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], s=10)
    plt.title("t-SNE Embedding of Train Images")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig("results/tsne_clusters.png")

    # Count co-occurrence of object pairs per image

    co_occurrence = Counter()
    for ann in train_labels:
        objs = [obj["category"] for obj in ann.get("labels", [])]
        for pair in combinations(sorted(set(objs)), 2):
            co_occurrence[pair] += 1

    # Create a matrix for top 10 objects
    top_objects = [obj for obj, _ in train_stats["class_counter"].most_common(10)]
    matrix = np.zeros((len(top_objects), len(top_objects)), dtype=int)
    for i, obj1 in enumerate(top_objects):
        for j, obj2 in enumerate(top_objects):
            if obj1 != obj2:
                matrix[i, j] = co_occurrence.get(tuple(sorted([obj1, obj2])), 0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=top_objects,
        yticklabels=top_objects,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title("Top 10 Object Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig("results/cooccurrence_matrix.png")

    iou_list = []

    for ann in train_labels:
        boxes = [obj["box2d"] for obj in ann.get("labels", []) if "box2d" in obj]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = compute_iou(boxes[i], boxes[j])
                iou_list.append(iou)

    plt.figure(figsize=(8, 6))
    sns.histplot(iou_list, bins=50, kde=True)
    plt.title("IoU Distribution of Object Pairs")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results/iou_density_plot.png")

    # Class Distribution

    plot_class_distribution(
        train_stats["class_counter"],
        os.path.join(RESULTS_DIR, "class_distribution_train.png"),
    )
    plot_class_distribution(
        val_stats["class_counter"],
        os.path.join(RESULTS_DIR, "class_distribution_val.png"),
    )

    # Box Area & Aspect Ratio

    plot_box_area_hist(
        train_stats["box_areas"], os.path.join(RESULTS_DIR, "box_area_hist_train.png")
    )
    plot_aspect_ratio_hist(
        train_stats["aspect_ratios"],
        os.path.join(RESULTS_DIR, "aspect_ratio_hist_train.png"),
    )

    # Objects per image
    plot_objects_per_image(
        train_stats["objects_per_image"],
        os.path.join(RESULTS_DIR, "objects_per_image_train.png"),
    )

    # Object centers heatmap
    plot_object_center_heatmap(
        train_stats["object_centers"],
        os.path.join(RESULTS_DIR, "object_heatmap_train.png"),
    )

    # Time of day / weather / scene
    plot_attribute_distribution(
        train_stats["timeofday_counter"],
        "Time of Day Distribution",
        os.path.join(RESULTS_DIR, "timeofday_distribution_train.png"),
    )
    plot_attribute_distribution(
        train_stats["weather_counter"],
        "Weather Distribution",
        os.path.join(RESULTS_DIR, "weather_distribution_train.png"),
    )
    plot_attribute_distribution(
        train_stats["scene_counter"],
        "Scene Distribution",
        os.path.join(RESULTS_DIR, "scene_distribution_train.png"),
    )

    # Scene-object correlation
    plot_scene_object_corr(
        train_stats["scene_object_counter"],
        os.path.join(RESULTS_DIR, "scene_object_corr_train.png"),
    )

    compare_class_distribution(
        train_stats["class_counter"],
        val_stats["class_counter"],
        os.path.join(RESULTS_DIR, "train_val_class_balance.png"),
    )


if __name__ == "__main__":
    main()
