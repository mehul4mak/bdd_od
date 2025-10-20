"""
EDA Utility Functions for plotting
"""

import os
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- Optional: Save one sample image with boxes ---
def show_sample_with_boxes(annotation, IMAGE_DIR, split="train", save_path=None):
    image_path = os.path.join(IMAGE_DIR, split, annotation["name"])
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for label in annotation.get("labels", []):
        if "box2d" in label:
            box = label["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            category = label.get("category", "N/A")
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), category, fill="yellow")

    if save_path:
        image.save(save_path)
    return image


# -------------------------
# --- Plotting Functions ---
# -------------------------


def plot_class_distribution(counter, filename):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counter.keys()), y=list(counter.values()))
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_box_area_hist(box_areas, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(box_areas, bins=50, alpha=0.7)
    plt.title("Box Area Distribution")
    plt.xlabel("Area (pixels^2)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_aspect_ratio_hist(aspect_ratios, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(aspect_ratios, bins=50, alpha=0.7)
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Width / Height")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_objects_per_image(objects_per_image, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(objects_per_image, bins=50, alpha=0.7)
    plt.title("Objects per Image")
    plt.xlabel("Number of Objects")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_object_center_heatmap(centers, filename, img_width=1280, img_height=720):
    plt.figure(figsize=(8, 6))
    if len(centers) > 0:
        x, y = centers[:, 0], centers[:, 1]
        plt.hist2d(
            x, y, bins=[50, 50], range=[[0, img_width], [0, img_height]], cmap="Reds"
        )
        plt.colorbar(label="Count")
        plt.gca().invert_yaxis()
    plt.title("Object Centers Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_attribute_distribution(counter, title, filename):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(counter.keys()), y=list(counter.values()))
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_scene_object_corr(scene_obj_counter, filename):
    scenes = list(scene_obj_counter.keys())
    classes = set()
    for c in scene_obj_counter.values():
        classes.update(c.keys())
    classes = sorted(classes)

    data = np.zeros((len(scenes), len(classes)))
    for i, s in enumerate(scenes):
        for j, c in enumerate(classes):
            data[i, j] = scene_obj_counter[s][c]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        data, xticklabels=classes, yticklabels=scenes, cmap="Reds", annot=True, fmt="g"
    )
    plt.title("Scene-Object Correlation")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def preprocess_annotations(labels):
    """Extracts per-image statistics from annotations."""
    data_stats = {
        "image_name": [],
        "class_counter": Counter(),
        "box_areas": [],
        "aspect_ratios": [],
        "objects_per_image": [],
        "object_centers": [],
        "timeofday_counter": Counter(),
        "weather_counter": Counter(),
        "scene_counter": Counter(),
        "scene_object_counter": defaultdict(Counter),
    }

    for ann in labels:
        objects = ann.get("labels", [])
        n_objects = len(objects)
        data_stats["objects_per_image"].append(n_objects)
        data_stats["image_name"].append(ann.get("name", []))

        for obj in objects:
            cls = obj.get("category")
            data_stats["class_counter"][cls] += 1

            # Bounding box stats
            box = obj.get("box2d")
            if box:
                x1, y1 = box["x1"], box["y1"]
                x2, y2 = box["x2"], box["y2"]
                w, h = x2 - x1, y2 - y1
                area = w * h
                aspect_ratio = w / h if h != 0 else 0
                data_stats["box_areas"].append(area)
                data_stats["aspect_ratios"].append(aspect_ratio)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                data_stats["object_centers"].append(center)

        # Scene-level attributes
        attrs = ann.get("attributes", {})
        timeofday = attrs.get("timeofday")
        weather = attrs.get("weather")
        scene = attrs.get("scene")

        if timeofday:
            data_stats["timeofday_counter"][timeofday] += 1
        if weather:
            data_stats["weather_counter"][weather] += 1
        if scene:
            data_stats["scene_counter"][scene] += 1
            for obj in objects:
                data_stats["scene_object_counter"][scene][obj.get("category")] += 1

    # Convert lists to numpy arrays for easy statistics
    data_stats["box_areas"] = np.array(data_stats["box_areas"])
    data_stats["aspect_ratios"] = np.array(data_stats["aspect_ratios"])
    data_stats["objects_per_image"] = np.array(data_stats["objects_per_image"])
    data_stats["object_centers"] = np.array(data_stats["object_centers"])

    return data_stats


def save_extreme_boxes(images, labels, image_dir, save_dir, num_examples=5):
    os.makedirs(save_dir, exist_ok=True)

    # Collect all boxes with their areas
    all_boxes = []
    for ann in labels:
        img_name = ann["name"]
        for obj in ann.get("labels", []):
            if "box2d" in obj:
                box = obj["box2d"]
                area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                all_boxes.append(
                    {
                        "img_name": img_name,
                        "box": box,
                        "area": area,
                        "category": obj["category"],
                    }
                )

    # Sort boxes by area
    sorted_boxes = sorted(all_boxes, key=lambda x: x["area"])

    # Smallest and largest
    extremes = {
        "smallest": sorted_boxes[:num_examples],
        "largest": sorted_boxes[-num_examples:],
    }

    for key, items in extremes.items():
        for idx, item in enumerate(items):
            img_path = os.path.join(image_dir, item["img_name"])
            with Image.open(img_path) as img:
                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)
                box = item["box"]
                color = "red" if key == "smallest" else "green"
                draw.rectangle(
                    [box["x1"], box["y1"], box["x2"], box["y2"]], outline=color, width=5
                )

                # Add text
                text = f"{item['category']} | Area: {item['area']:.2f}"
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except AttributeError as e:
                    font = ImageFont.load_default()
                    print(e)
                draw.text(
                    (box["x1"], max(box["y1"] - 25, 0)), text, fill=color, font=font
                )

                # Save
                save_path = os.path.join(
                    save_dir, f"{key}_{idx + 1}_{item['img_name']}"
                )
                draw_img.save(save_path)
                print(f"Saved: {save_path}")


def extract_image_features(image_paths, image_dir, size=(64, 64)):
    """Convert images to flattened grayscale vectors for clustering."""
    features = []
    for img_name in image_paths:
        img_path = os.path.join(image_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.convert("L").resize(size)
                features.append(np.array(img).flatten() / 255.0)
        except AttributeError:
            continue
    return np.array(features)


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def compare_class_distribution(train_counter, val_counter, filename):
    classes = sorted(set(train_counter.keys()) | set(val_counter.keys()))
    train_counts = [train_counter.get(c, 0) for c in classes]
    val_counts = [val_counter.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, train_counts, width, label="Train")
    plt.bar(x + width / 2, val_counts, width, label="Val")
    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Count")
    plt.title("Class Distribution: Train vs Val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
