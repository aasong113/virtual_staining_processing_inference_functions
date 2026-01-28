import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

def extract_xy(filename):
    """Extract X and Y values from the filename."""
    match = re.search(r'X=(\d+)_Y=(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def group_images_by_xy(image_files):
    """Group image file paths by (X, Y) tuple."""
    grouped = {}
    for f in image_files:
        key = extract_xy(f)
        if key:
            grouped.setdefault(key, []).append(f)
    return grouped

def select_images_gui(images, root_dir):
    """Display images with checkboxes, return subset to keep."""
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    check_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    labels = [f"Keep {os.path.basename(p)}" for p in images]
    visibility = [True] * len(images)
    checkboxes = widgets.CheckButtons(check_ax, labels, visibility)

    selected = {label: True for label in labels}

    def on_checkbox(label):
        selected[label] = not selected[label]

    checkboxes.on_clicked(on_checkbox)

    for ax, img_path in zip(axes, images):
        img = Image.open(os.path.join(root_dir, img_path))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(os.path.basename(img_path), fontsize=10)

    def on_submit(event):
        plt.close()

    submit_ax = fig.add_axes([0.45, 0.90, 0.1, 0.05])
    button = widgets.Button(submit_ax, 'Confirm')
    button.on_clicked(on_submit)

    plt.show()

    keep_paths = [img for img, label in zip(images, labels) if selected[label]]
    return keep_paths

def run_image_selector(input_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        return

    all_files = [f for f in os.listdir(input_dir) if 'BIT' in f and f.endswith('.tif')]
    grouped = group_images_by_xy(all_files)

    print(f"Found {len(all_files)} BIT images in {len(grouped)} (X,Y) groups.")

    for (x, y), group in sorted(grouped.items()):
        print(f"\nGroup (X={x}, Y={y}) with {len(group)} images...")
        keep = select_images_gui(group, input_dir)

        to_delete = set(group) - set(keep)
        for file in to_delete:
            os.remove(os.path.join(input_dir, file))
            print(f"Deleted: {file}")
        for file in keep:
            print(f"Kept:    {file}")

    print("✅ Done: All (X,Y) groups processed.")

# Example usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bit_image_selector.py /path/to/image_directory")
    else:
        run_image_selector(sys.argv[1])