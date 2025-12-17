import glob
import sys
import os
import re
import numpy as np
import tifffile as tiff
import tifffile
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
from collections import defaultdict
import pandas as pd
from skimage.io import imread

def plot_matched_images_from_paths(folder_paths):
    """
    Randomly selects one image from the first folder path, and finds the corresponding
    image (matching filename) in the other folders. Then plots all of them in a 1 x N row.

    Args:
        folder_paths (List[str]): List of folder paths (strings)
    """
    assert len(folder_paths) >= 2, "Need at least two folder paths."

    # Get all image files in the first folder
    image_files = [f for f in os.listdir(folder_paths[0])
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not image_files:
        raise ValueError("No images found in first folder.")

    # Randomly pick an image
    selected_name = random.choice(image_files)

    # Try to find matching images in all folders
    images = []
    for path in folder_paths:
        match = os.path.join(path, selected_name)
        if not os.path.exists(match):
            raise FileNotFoundError(f"{selected_name} not found in {path}")
        img = Image.open(match).convert('RGB')
        images.append(img)

    # Plot images side by side
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    for ax, img, path in zip(axes, images, folder_paths):
        ax.imshow(img)
        ax.set_title(os.path.basename(path), fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_patch_quadrants(p1, p2, p3, p4, titles=None, row_titles=None, cmap=None):
    """
    Plots patches from four lists in a 4-row grid.

    Args:
        p1, p2, p3, p4 (list of np.ndarray): Lists of patch images.
        titles (list of str, optional): Titles for each column (shared across rows).
        row_titles (list of str, optional): Titles for each row.
        cmap (str or None): Color map to use. If None, will default to 'gray' if image has 1 channel.
    """
    assert len(p1) == len(p2) == len(p3) == len(p4), "All patch lists must have the same length."
    assert row_titles is None or len(row_titles) == 4, "Row titles must be a list of length 4."
    
    n = len(p1)

    plt.figure(figsize=(3 * n, 12))  # Adjust height for 4 rows

    for i in range(n):
        for row, patches in enumerate([p1, p2, p3, p4], start=1):
            idx = (row - 1) * n + i + 1
            ax = plt.subplot(4, n, idx)

            patch = patches[i]
            # Determine if grayscale
            if cmap is not None:
                use_cmap = cmap
            elif patch.ndim == 2 or (patch.ndim == 3 and patch.shape[2] == 1):
                use_cmap = 'gray'
                patch = patch.squeeze()  # remove channel dim if present
            else:
                use_cmap = None

            ax.imshow(patch, cmap=use_cmap)
            ax.axis('off')
            if row == 1 and titles:
                ax.set_title(titles[i], fontsize=10)

            # Set row titles on the first column
            if i == 0 and row_titles is not None:
                ax_row_title = plt.subplot(4, n, (row - 1) * n + 1)
                ax_row_title.set_title(row_titles[row - 1], fontsize=12)
                ax_row_title.axis('off')  # Hide the axis for row title
    
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming p1, p2, p3, and p4 are defined and contain image data
# plot_patch_quadrants(p1, p2, p3, p4, titles=["Title1", "Title2", "Title3"], row_titles=["Row Title 1", "Row Title 2", "Row Title 3", "Row Title 4"], cmap='gray')

def create_files_dict(folder, pattern=None):
    """
    Creates a dictionary mapping:
        key: extracted full patch identifier (img=.._patch_..)
        value: file path
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Invalid directory: {folder}")

    patterns = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in patterns:
        files.extend(glob.glob(os.path.join(folder, ext)))

    files_dict = {}
    for f in files:
        try:
            key = extract_full_patch_tag(f, pattern=pattern)
            files_dict[key] = f
        except ValueError:
            # skip files that do not match expected pattern
            continue

    return files_dict

# def extract_full_patch_tag(filename):
#     base = os.path.splitext(os.path.basename(filename))[0]
#     m = re.search(r'img=\d+_patch_X=\d+_Y=\d+_Z=\d+_P=\d+', base)
#     if m:
#         return m.group(0)
#     raise ValueError(f"Could not find full patch tag in filename: {filename}")

import os
import re

def extract_full_patch_tag(filename, pattern=None):
    """
    Extract a patch-identifying substring from a filename.

    Parameters
    ----------
    filename : str
        Input filename or path.
    pattern : str or None
        Optional regex pattern to search for.
        Defaults to full patch tag:
        r'img=\\d+_patch_X=\\d+_Y=\\d+_Z=\\d+_P=\\d+'

    Returns
    -------
    str
        Matched patch tag.

    Raises
    ------
    ValueError
        If no match is found.
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    if pattern is None:
        pattern = r'img=\d+_patch_X=\d+_Y=\d+_Z=\d+_P=\d+'

    m = re.search(pattern, base)
    if m:
        return m.group(0)

    raise ValueError(
        f"Could not find patch tag using pattern '{pattern}' in filename: {filename}"
    )


# def get_n_random_patches(n, files_dict1, files_dict2, files_dict3):
#     random_keys = random.sample(list(files_dict1.keys()), n)

#     p1, p2, p3 = [], [], []
#     for key in random_keys:

#         print(files_dict1[key])
#         print(files_dict2[key])
#         print(files_dict3[key])
#         p1.append(imread(files_dict1[key]))
#         p2.append(imread(files_dict2[key]))
#         p3.append(imread(files_dict3[key]))

#     return p1, p2, p3

import random
from skimage.io import imread

def get_n_random_patches(n, *files_dicts):
    """
    Get `n` random patches from an arbitrary number of file dictionaries.

    Parameters:
        n (int): Number of random patches to extract.
        *files_dicts: Variable number of dictionaries mapping keys to image paths.

    Returns:
        List of lists: Each list contains `n` image patches from one file dictionary.
                       For example, if there are 3 dictionaries, returns [p1, p2, p3]
                       where p1[i], p2[i], and p3[i] correspond to the same key.
    """
    if not files_dicts:
        raise ValueError("At least one file dictionary must be provided.")

    # Ensure all dicts have the same keys
    shared_keys = set(files_dicts[0].keys())
    for fd in files_dicts[1:]:
        shared_keys &= set(fd.keys())
    shared_keys = list(shared_keys)

    if len(shared_keys) < n:
        raise ValueError(f"Only {len(shared_keys)} shared keys available, but {n} were requested.")

    random_keys = random.sample(shared_keys, n)

    all_patches = [[] for _ in files_dicts]  # Create empty list for each dictionary

    for key in random_keys:
        for i, fd in enumerate(files_dicts):
            print(fd[key])  # Optional: remove if too verbose
            all_patches[i].append(imread(fd[key]))

    return all_patches



def get_random_images_data(folder_path, n):
    """
    Get n random images from the specified folder and load their image data.
    
    Parameters:
    folder_path (str): The path to the folder containing images.
    n (int): The number of random images to retrieve.

    Returns:
    list: A list containing tuples of (image_path, image_data) for n random images.
    """
    # List to hold the paths and data of images
    image_data_list = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")
    
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter to only include image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff'))]

    # Check if there are enough images
    if n > len(image_files):
        raise ValueError(f"There are only {len(image_files)} images in '{folder_path}', but requested {n}.")

    # Select n random images
    selected_files = random.sample(image_files, n)
    
    # Load each selected image and add to the list
    for img_file in selected_files:
        img_path = os.path.join(folder_path, img_file)
        
        # Load the image data
        try:
            image_data_list.append(imread(img_path))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return image_data_list



import re

def extract_xyz_position(filename):
    """
    Extracts the position encoding in the form 'X=#_Y=#_Z=#' from a given filename.

    Args:
        filename (str): The filename to extract from.

    Returns:
        str or None: The extracted position string if found, else None.
    """
    match = re.search(r'X=\d+_Y=\d+_Z=\d+', filename)
    return match.group(0) if match else None

# Example usage:
# filename = "crop_BIT_kidney_tumor_2_X=2_Y=5_Z=4_expTime=2ms.tif_patches_stitch_metadata.txt"
# print(extract_xyz_position(filename))  # Output: X=2_Y=5_Z=4



def parse_position_from_filename(filename):
    """
    Extract X, Y, Z, P from a basename like 'MUSE_BIT_img=0_patch_X=0_Y=0_Z=1_P=3.tif'
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    position = {}
    for part in parts:
        if '=' in part:
            key, val = part.split('=')
            if key in {'X', 'Y', 'Z', 'P'}:
                position[key] = val.strip('.tif')
    return f"X={position['X']}_Y={position['Y']}_Z={position['Z']}_P={position['P']}"

def find_matching_file(folder_path, position_encoding):
    """
    Search for a file in folder_path that ends with the same position encoding
    """
    candidates = glob(os.path.join(folder_path, f"*{position_encoding}.tif"))
    return candidates[0] if candidates else None

def load_patch_images(folder1, folder2, folder3, n=10):
    """
    Randomly select n images from folder1 and load corresponding patches from all 3 folders
    Returns: p1, p2, p3 (lists of np.ndarray)
    """
    files1 = glob(os.path.join(folder1, "*.tif"))
    selected_files = random.sample(files1, n)

    p1, p2, p3 = [], [], []

    for f1 in selected_files:
        pos = parse_position_from_filename(f1)
        f2 = find_matching_file(folder2, pos)
        f3 = find_matching_file(folder3, pos)

        if f2 is None or f3 is None:
            print(f"Warning: Missing matching file for position {pos}")
            continue

        p1.append(imread(f1))
        p2.append(imread(f2))
        p3.append(imread(f3))

    return p1, p2, p3

def plot_image_channels_grid(folder_path, n=5, image_exts={'.png', '.jpg', '.jpeg', '.tif'}):
    """
    Load n images from a folder and plot:
    - Row 1: Original image
    - Row 2-4: Individual channels (0, 1, 2)
    Each image has its mean value shown as title.

    Parameters:
        folder_path (str): Path to folder with images
        n (int): Number of images to load
        image_exts (set): Allowed image extensions
    """
    # List image files
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_exts]
    image_files = sorted(image_files)[:n]

    if not image_files:
        raise ValueError("No valid image files found in folder.")

    fig, axes = plt.subplots(4, n, figsize=(3*n, 12))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)  # Ensure shape is (4, n)

    for i, fname in enumerate(image_files):
        img_path = os.path.join(folder_path, fname)
        img = imread(img_path)

        if img.ndim == 2:  # grayscale, expand to 3-channel
            img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] < 3:
            raise ValueError(f"Image {fname} has less than 3 channels.")

        # Clip to 3 channels if more
        img = img[:, :, :3]
        img = img.astype(np.float32)

        # Row 0: original image
        mean_val = img.mean()
        axes[0, i].imshow(img / img.max())
        axes[0, i].set_title(f"Mean: {mean_val:.4f}")
        axes[0, i].axis('off')

        # Rows 1-3: individual channels
        for ch in range(3):
            channel = img[:, :, ch]
            ch_mean = channel.mean()
            axes[ch+1, i].imshow(channel, cmap='gray')
            axes[ch+1, i].set_title(f"Ch{ch} Mean: {ch_mean:.4f}")
            axes[ch+1, i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_pretraining_losses_UVCGANv2(csv_file_path):
    """
    Reads a CSV file containing 'loss_a', 'loss_b', and 'epoch' columns,
    and plots the losses over epochs.

    Parameters:
        csv_file_path (str): Path to the history CSV file.
    """
    # Load CSV
    df = pd.read_csv(csv_file_path)

    # Check if required columns are present
    if not {'epoch', 'loss_a', 'loss_b'}.issubset(df.columns):
        raise ValueError("CSV must contain 'epoch', 'loss_a', and 'loss_b' columns")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss_a'], label='Loss A', marker='o')
    plt.plot(df['epoch'], df['loss_b'], label='Loss B', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pretraining Losses Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_losses_UVCGANv2(csv_path):
    """
    Reads a CSV file containing training losses and plots each loss component vs. epoch.

    Parameters:
    - csv_path (str): Path to the .csv file containing loss values and epochs
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract epoch and loss columns
    epochs = df['epoch']
    loss_keys = ['gen_ab', 'gen_ba', 'cycle_a', 'cycle_b',
                 'disc_a', 'disc_b', 'idt_a', 'idt_b', 'gp_a', 'gp_b']

    # Plot
    plt.figure(figsize=(12, 6))
    for key in loss_keys:
        plt.plot(epochs, df[key], label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Loss Components vs. Epoch")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def sample_matched_patches(folder_paths, num_samples, seed = 42, extensions=("tif", "tiff", "png", "jpg", "jpeg")):
    """
    Given N folder paths, sample I coordinate-matched images from each folder.

    Parameters:
        folder_paths (list of str): List of folder paths.
        num_samples (int): Number of matched coordinate keys to randomly sample.
        extensions (tuple): Valid image extensions.

    Returns:
        List of N lists, each containing I NumPy arrays of patches.
    """
    coord_pattern = re.compile(r"img=\d+_patch_X=\d+_Y=\d+_Z=\d+_P=\d+", re.IGNORECASE)

    # Build a dictionary of coord_key -> filename for each folder
    folder_coord_maps = []
    for folder in folder_paths:
        coord_to_file = {}
        for fname in os.listdir(folder):
            if any(fname.lower().endswith(ext) for ext in extensions):
                match = coord_pattern.search(fname)
                if match:
                    coord_key = match.group(0)
                    coord_to_file[coord_key] = fname
        folder_coord_maps.append(coord_to_file)

    # Find intersection of all coord keys across all folders
    common_coords = set(folder_coord_maps[0].keys())
    for coord_map in folder_coord_maps[1:]:
        common_coords &= set(coord_map.keys())

    if len(common_coords) < num_samples:
        raise ValueError(f"Only {len(common_coords)} common coordinate keys found, cannot sample {num_samples}")

    if seed is not None:
        random.seed(seed)
    sampled_coords = random.sample(list(common_coords), num_samples)

    # Load images for each folder and coord
    result = []
    for folder_idx, coord_map in enumerate(folder_coord_maps):
        folder_result = []
        folder = folder_paths[folder_idx]
        for coord_key in sampled_coords:
            img_path = os.path.join(folder, coord_map[coord_key])
            try:
                if img_path.lower().endswith((".tif", ".tiff")):
                    img = tifffile.imread(img_path)
                else:
                    img = np.array(Image.open(img_path))
            except Exception as e:
                raise RuntimeError(f"Failed to read {img_path}: {e}")
            folder_result.append(img)
        result.append(folder_result)

    return result  # shape: [n_folders][i_samples] of numpy arrays


import matplotlib.pyplot as plt

def plot_matched_patches_grid(patches, row_labels=None, figsize=(12, 8), cmap='gray'):
    n_folders = len(patches)
    i_samples = len(patches[0])

    fig, axes = plt.subplots(n_folders, i_samples, figsize=figsize)

    # Normalize axes shape
    if n_folders == 1:
        axes = [axes]
    if i_samples == 1:
        axes = [[ax] for ax in axes]

    for row in range(n_folders):
        for col in range(i_samples):
            ax = axes[row][col]
            img = patches[row][col]

            if img.ndim == 2:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

            ax.axis("off")

            # Add label only to the first column of each row
            if col == 0 and row_labels:
                ax.annotate(
                    row_labels[row],
                    xy=(0, 0.5),
                    xytext=(-fig.dpi, 0),
                    textcoords='offset points',
                    xycoords='axes fraction',
                    ha='right',
                    va='center',
                    fontsize=12,
                    rotation=0
                )

    plt.tight_layout()
    plt.show()



def sample_patches_from_images_within_roi(img1, img2, img3, roi_h, roi_w, patch_h, patch_w, n):
    """
    Randomly samples n patches from 3 images constrained within a predefined ROI.

    Args:
        img1, img2, img3: PIL Images or numpy arrays of the same shape.
        roi_h (int): ROI height (max Y-range).
        roi_w (int): ROI width (max X-range).
        patch_h (int): Height of each patch.
        patch_w (int): Width of each patch.
        n (int): Number of patches to sample.

    Returns:
        patches1, patches2, patches3: Lists of n patches (NumPy arrays) from each image.
    """
    # Convert to NumPy if PIL
    imgs = []
    for img in (img1, img2, img3):
        if isinstance(img, Image.Image):
            imgs.append(np.array(img))
        else:
            imgs.append(img)
    img1, img2, img3 = imgs

    # Sanity check
    H, W = img1.shape[:2]
    if roi_h > H or roi_w > W:
        raise ValueError("ROI size exceeds image dimensions.")

    # Define upper-left corner limits for patch sampling
    max_y = roi_h - patch_h
    max_x = roi_w - patch_w
    if max_y < 0 or max_x < 0:
        raise ValueError("Patch size is larger than ROI.")

    patches1, patches2, patches3 = [], [], []

    for _ in range(n):
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)

        patch1 = img1[top:top+patch_h, left:left+patch_w]
        patch2 = img2[top:top+patch_h, left:left+patch_w]
        patch3 = img3[top:top+patch_h, left:left+patch_w]

        patches1.append(patch1)
        patches2.append(patch2)
        patches3.append(patch3)

    return patches1, patches2, patches3


def extract_xyz(filename):
    """
    Extracts (X, Y, Z) tuple from a filename with format '...X=2_Y=4_Z=2...'
    """
    match = re.search(r'X=(\d+)_Y=(\d+)_Z=(\d+)', filename)
    if match:
        return tuple(map(int, match.groups()))
    else:
        raise ValueError(f"Filename does not contain X=Y=Z format: {filename}")

def group_txt_files_by_keyword(reconstruction_coordinates_path, keywords):
    """
    Groups .txt file paths by keyword and ensures all groups are ordered by X, Y, Z.

    Args:
        reconstruction_coordinates_path (str): Path to folder with .txt files.
        keywords (list of str): Keywords to group filenames by.

    Returns:
        dict: Mapping from keyword to list of sorted file paths.
    """
    file_list = glob.glob(os.path.join(reconstruction_coordinates_path, "*.txt"))
    grouped = defaultdict(list)

    # Group by keyword
    for file in file_list:
        filename = os.path.basename(file)
        for keyword in keywords:
            if keyword in filename:
                grouped[keyword].append(file)

    # Sort each group by (X, Y, Z) coordinates
    for keyword in grouped:
        grouped[keyword].sort(key=lambda f: extract_xyz(os.path.basename(f)))

    return grouped

def show_n_images(images, subtitles=None, main_title="Images", ncols=2, figsize_per_image=(5, 5)):
    """
    Display N images in a grid with subtitles and a main title.

    Parameters:
        images (list): List of PIL Images or numpy arrays.
        subtitles (list of str): Optional list of subtitles for each image.
        main_title (str): Title for the whole figure.
        ncols (int): Number of columns in the display grid.
        figsize_per_image (tuple): Width and height of each subplot in inches.
    """
    n = len(images)
    if subtitles is None:
        subtitles = [''] * n

    nrows = (n + ncols - 1) // ncols
    figsize = (figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)

    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(subtitles[i], fontsize=10)
        ax.axis('off')

    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


import os
import re
import numpy as np
import tifffile
from PIL import Image

def reconstruct_image_from_patches(metadata_path, patch_folder, return_as_array=False, force_dtype=None):
    """
    Reconstruct the original image (grayscale or RGB) from patches using metadata.
    Supports both .tif and .png patch files, as long as they match the coordinate pattern.

    Parameters:
        metadata_path (str): Path to the patch metadata file.
        patch_folder (str): Folder containing the patch images.
        return_as_array (bool): If True, return reconstructed image as NumPy array.
        force_dtype (np.dtype or str): Optional. Force output dtype (e.g., np.float32, np.uint8).

    Returns:
        np.ndarray or PIL.Image: Reconstructed image.
    """

    # --- Parse metadata ---
    with open(metadata_path, "r") as f:
        lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]

    with open(metadata_path, "r") as f:
        headers = f.readlines()
    w = int([line for line in headers if "width=" in line][0].split("width=")[1].split(",")[0])
    h = int([line for line in headers if "height=" in line][0].split("height=")[1])

    coord_pattern = re.compile(r"img=\d+_patch_X=\d+_Y=\d+_Z=\d+_P=\d+", re.IGNORECASE)

    # Build lookup: map coord pattern -> actual filename (.tif or .png)
    available_files = os.listdir(patch_folder)
    coord_to_file = {}
    for f in available_files:
        if f.lower().endswith((".tif", ".tiff", ".png")):
            match = coord_pattern.search(f)
            if match:
                coord_to_file[match.group(0)] = f

    # Determine RGB or grayscale from one patch
    first_patch_name = lines[0].split(",")[0]
    coord_match = coord_pattern.search(first_patch_name)
    if not coord_match:
        raise ValueError(f"Could not parse coordinates from {first_patch_name}")
    coord_key = coord_match.group(0)

    if coord_key not in coord_to_file:
        raise FileNotFoundError(f"No patch found matching coordinates {coord_key} in {patch_folder}")
    
    first_patch_path = os.path.join(patch_folder, coord_to_file[coord_key])
    first_patch = read_patch(first_patch_path)
    is_rgb = (first_patch.ndim == 3 and first_patch.shape[2] == 3)
    patch_dtype = first_patch.dtype
    num_channels = 3 if is_rgb else 1

    canvas = np.zeros((h, w, num_channels), dtype=np.float32)
    weight_map = np.zeros((h, w, num_channels), dtype=np.float32)

    # --- Accumulate patches ---
    for line in lines:
        parts = line.split(",")
        metadata_patch_name = parts[0]
        coord_match = coord_pattern.search(metadata_patch_name)
        if not coord_match:
            print(f"⚠️ Skipping malformed patch name: {metadata_patch_name}")
            continue

        coord_key = coord_match.group(0)
        if coord_key not in coord_to_file:
            print(f"⚠️ No file found for {coord_key}")
            continue

        patch_path = os.path.join(patch_folder, coord_to_file[coord_key])
        if not os.path.exists(patch_path):
            print(f"⚠️ Missing file: {patch_path}")
            continue

        patch = read_patch(patch_path).astype(np.float32)
        x, y, pw, ph = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

        if patch.ndim == 2:
            patch = patch[..., None]

        for c in range(num_channels):
            canvas[y:y+ph, x:x+pw, c] += patch[..., c]
            weight_map[y:y+ph, x:x+pw, c] += 1

    # Normalize and finalize
    weight_map[weight_map == 0] = 1
    reconstructed = canvas / weight_map

    if force_dtype is not None:
        dtype = np.dtype(force_dtype)
    elif patch_dtype == np.uint8:
        dtype = np.uint8
    else:
        dtype = np.float32

    if dtype == np.uint8:
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    else:
        reconstructed = reconstructed.astype(np.float32)

    if reconstructed.shape[2] == 1:
        reconstructed = reconstructed[..., 0]

    if return_as_array:
        return reconstructed
    else:
        return Image.fromarray(reconstructed.astype(np.uint8) if reconstructed.ndim == 3 else reconstructed)

def read_patch(path):
    """
    Load either a .tif or .png patch as numpy array
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        return tifffile.imread(path)
    elif ext == ".png":
        return np.array(Image.open(path))
    else:
        raise ValueError(f"Unsupported image format: {ext}")

        

import os
from PIL import Image

def keep_and_rename_fakeB_images_to_tif(folder, case_sensitive=True):
    """
    Only processes the folder if it contains images with '_fake_b' in the filename.
    Keeps only those images, renames them to remove '_fake_b', and saves as .tif.
    Deletes all other files.

    Args:
        folder (str): Path to the folder containing images.
        case_sensitive (bool): Whether to match 'fake_B' case-sensitively.
    """
    match_str = "_fake_B" if case_sensitive else "_fake_b"

    # First pass: check if any matching files exist
    has_fakeB = any(
        (match_str in (f if case_sensitive else f.lower()))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    )

    if not has_fakeB:
        print("❌ No '_fake_b' images found. Skipping folder.")
        return

    # Second pass: process the folder
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        check_name = filename if case_sensitive else filename.lower()

        if match_str in check_name:
            # Remove '_fake_B' or '_fake_b'
            if case_sensitive:
                base_name = filename.replace("_fake_B", "")
            else:
                base_name = filename.replace("_fake_B", "").replace("_fake_b", "")

            # Replace extension with .tif
            name_no_ext = os.path.splitext(base_name)[0]
            new_filename = name_no_ext + ".tif"
            new_path = os.path.join(folder, new_filename)

            try:
                img = Image.open(filepath)
                img.save(new_path, format='TIFF')
                os.remove(filepath)
                print(f"✅ Saved and renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
        else:
            os.remove(filepath)
            print(f"🗑️ Deleted: {filename}")

def get_random_image_paths(folder, n, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    """
    Selects n random image file paths from a folder.

    Args:
        folder (str): Path to the folder containing images.
        n (int): Number of random images to select.
        extensions (tuple): Valid image extensions to include.

    Returns:
        List[str]: List of n randomly selected image file paths.
    """
    all_images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]

    if n > len(all_images):
        raise ValueError(f"Requested {n} images, but only {len(all_images)} available.")

    return random.sample(all_images, n)

def parse_cyclegan_options(txt_path):
    """
    Parses a CycleGAN options .txt file and extracts lambda_cyc, lambda_idt, dataroot, trainA_handle, and trainB_handle.

    Args:
        txt_path (str): Path to the options .txt file.

    Returns:
        lambda_cyc (float): Sum of lambda_A and lambda_B.
        lambda_idt (float): Value of lambda_identity.
        dataroot (str): Path to the dataset.
        trainA_handle (str): Handle for domain A training data.
        trainB_handle (str): Handle for domain B training data.
    """
    lambda_A = None
    lambda_B = None
    lambda_identity = None
    dataroot = None
    trainA_handle = None
    trainB_handle = None

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("lambda_A:"):
                lambda_A = float(line.split(":")[1].strip())
            elif line.startswith("lambda_B:"):
                lambda_B = float(line.split(":")[1].strip())
            elif line.startswith("lambda_identity:"):
                lambda_identity = float(line.split(":")[1].strip())
            elif line.startswith("dataroot:"):
                dataroot = line.split(":")[1].split("\t")[0].strip()
            elif line.startswith("trainA_handle:"):
                trainA_handle = line.split(":")[1].split("\t")[0].strip()
            elif line.startswith("trainB_handle:"):
                trainB_handle = line.split(":")[1].split("\t")[0].strip()

    if None in (lambda_A, lambda_B, lambda_identity, dataroot, trainA_handle, trainB_handle):
        raise ValueError("One or more required values not found in the file.")

    lambda_cyc = lambda_A + lambda_B
    return lambda_cyc, lambda_identity, dataroot, trainA_handle, trainB_handle


def plot_cyclegan_average_epoch_losses(log_file_path, lambda_cyc=10.0, lambda_idt=5.0):
    """
    Parse a training log and plot:
    - Average individual losses per epoch (top figure)
    - Total CycleGAN loss per epoch (bottom figure)

    Args:
        log_file_path (str): Path to the log.txt file.
        lambda_cyc (float): Weight for cycle consistency loss.
        lambda_idt (float): Weight for identity loss.
    """
    # Dictionary to store lists of losses per epoch
    data = defaultdict(lambda: {
        'D_A': [], 'G_A': [], 'cycle_A': [], 'idt_A': [],
        'D_B': [], 'G_B': [], 'cycle_B': [], 'idt_B': []
    })

    # Regex pattern to extract epoch and losses
    pattern = re.compile(
        r"epoch: (\d+), iters: \d+.*D_A: ([\d\.]+), G_A: ([\d\.]+), "
        r"cycle_A: ([\d\.]+), idt_A: ([\d\.]+), D_B: ([\d\.]+), G_B: ([\d\.]+), "
        r"cycle_B: ([\d\.]+), idt_B: ([\d\.]+)"
    )

    # Read and parse the log file
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                data[epoch]['D_A'].append(float(match.group(2)))
                data[epoch]['G_A'].append(float(match.group(3)))
                data[epoch]['cycle_A'].append(float(match.group(4)))
                data[epoch]['idt_A'].append(float(match.group(5)))
                data[epoch]['D_B'].append(float(match.group(6)))
                data[epoch]['G_B'].append(float(match.group(7)))
                data[epoch]['cycle_B'].append(float(match.group(8)))
                data[epoch]['idt_B'].append(float(match.group(9)))

    # Prepare averaged data per epoch
    epochs = sorted(data.keys())
    avg_losses = {key: [] for key in data[epochs[0]]}  # keys: D_A, G_A, etc.
    total_losses = []

    for epoch in epochs:
        for key in avg_losses:
            avg_losses[key].append(np.mean(data[epoch][key]))

        # Compute total CycleGAN loss (exclude D_A and D_B if desired)
        G_A = np.mean(data[epoch]['G_A'])
        G_B = np.mean(data[epoch]['G_B'])
        cycle_A = np.mean(data[epoch]['cycle_A'])
        cycle_B = np.mean(data[epoch]['cycle_B'])
        idt_A = np.mean(data[epoch]['idt_A'])
        idt_B = np.mean(data[epoch]['idt_B'])

        total_loss = G_A + G_B + lambda_cyc * (cycle_A + cycle_B) + lambda_idt * (idt_A + idt_B)
        total_losses.append(total_loss)

    # Plot individual losses
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    for key, values in avg_losses.items():
        plt.plot(epochs, values, marker='o', label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Components per Epoch')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot total loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, total_losses, marker='o', color='black', label='Total CycleGAN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total CycleGAN Loss per Epoch')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def get_images_with_string_match(folder_path, string_match, extensions=None):
    """
    Get all images in a folder that contain 'fake_B' in their filename.

    Parameters:
        folder_path (str): Path to the folder containing images.
        extensions (list of str, optional): List of allowed file extensions. Defaults to common image types.

    Returns:
        list: List of full file paths matching the criteria.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if string_match in f and os.path.splitext(f)[1].lower() in extensions
    ]
    return images



def sort_images_by_epoch(image_paths, keyword="epoch"):
    """
    Sort a list of image paths by the numeric value following `keyword`.
    Example filenames: img_epoch001.png, img_epoch002.png
    """
    def extract_epoch(path):
        basename = os.path.basename(path)
        match = re.search(rf'{keyword}(\d+)', basename)
        return int(match.group(1)) if match else -1
    
    return sorted(image_paths, key=extract_epoch)

def plot_patch_triplets(p1, p2, p3, titles=None, cmap=None):
    """
    Plots patches from three lists in a 3-row grid.

    Args:
        p1, p2, p3 (list of np.ndarray): Lists of patch images.
        titles (list of str, optional): Titles for each column (shared across rows).
        cmap (str or None): Color map to use. If None, will default to 'gray' if image has 1 channel.
    """
    assert len(p1) == len(p2) == len(p3), "All patch lists must have the same length."
    n = len(p1)

    plt.figure(figsize=(3 * n, 9))  # Adjust width for number of patches

    for i in range(n):
        for row, patches in enumerate([p1, p2, p3], start=1):
            idx = (row - 1) * n + i + 1
            ax = plt.subplot(3, n, idx)

            patch = patches[i]
            # Determine if grayscale
            if cmap is not None:
                use_cmap = cmap
            elif patch.ndim == 2 or (patch.ndim == 3 and patch.shape[2] == 1):
                use_cmap = 'gray'
                patch = patch.squeeze()  # remove channel dim if present
            else:
                use_cmap = None

            ax.imshow(patch, cmap=use_cmap)
            ax.axis('off')
            if row == 1 and titles:
                ax.set_title(titles[i], fontsize=10)

    plt.tight_layout()
    plt.show()




def plot_patch_quadrants(p1, p2, p3, p4, titles=None, row_titles=None, cmap=None):
    """
    Plots patches from four lists in a 4-row grid.

    Args:
        p1, p2, p3, p4 (list of np.ndarray): Lists of patch images.
        titles (list of str, optional): Titles for each column (shared across rows).
        row_titles (list of str, optional): Titles for each row.
        cmap (str or None): Color map to use. If None, will default to 'gray' if image has 1 channel.
    """
    assert len(p1) == len(p2) == len(p3) == len(p4), "All patch lists must have the same length."
    assert row_titles is None or len(row_titles) == 4, "Row titles must be a list of length 4."
    
    n = len(p1)

    plt.figure(figsize=(3 * n, 12))  # Adjust height for 4 rows

    for i in range(n):
        for row, patches in enumerate([p1, p2, p3, p4], start=1):
            idx = (row - 1) * n + i + 1
            ax = plt.subplot(4, n, idx)

            patch = patches[i]
            # Determine if grayscale
            if cmap is not None:
                use_cmap = cmap
            elif patch.ndim == 2 or (patch.ndim == 3 and patch.shape[2] == 1):
                use_cmap = 'gray'
                patch = patch.squeeze()  # remove channel dim if present
            else:
                use_cmap = None

            ax.imshow(patch, cmap=use_cmap)
            ax.axis('off')
            if row == 1 and titles:
                ax.set_title(titles[i], fontsize=10)

            # Set row titles on the first column
            if i == 0 and row_titles is not None:
                ax_row_title = plt.subplot(4, n, (row - 1) * n + 1)
                ax_row_title.set_title(row_titles[row - 1], fontsize=12)
                ax_row_title.axis('off')  # Hide the axis for row title
    
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming p1, p2, p3, and p4 are defined and contain image data
# plot_patch_quadrants(p1, p2, p3, p4, titles=["Title1", "Title2", "Title3"], row_titles=["Row Title 1", "Row Title 2", "Row Title 3", "Row Title 4"], cmap='gray')

import numpy as np
import matplotlib.pyplot as plt

def plot_patch_rows(*rows, titles=None, row_titles=None, cmap=None):
    """
    Plot patches from p lists in a p-row by n-column grid.

    Args:
        *rows: Any number (p) of lists of np.ndarray. All lists must have same length (n).
        titles (list[str] or None): Column titles (length n), shown on the first row.
        row_titles (list[str] or None): Row titles (length p), shown on the first column of each row.
        cmap (str or None): Colormap. If None, defaults to 'gray' for 2D or 1-channel images.
    """
    p = len(rows)
    if p == 0:
        raise ValueError("Provide at least one list of patches.")
    lengths = [len(r) for r in rows]
    if len(set(lengths)) != 1:
        raise AssertionError("All patch lists must have the same length.")
    n = lengths[0]
    if titles is not None and len(titles) != n:
        raise AssertionError("titles must have length equal to the number of columns.")
    if row_titles is not None and len(row_titles) != p:
        raise AssertionError("row_titles must have length equal to the number of rows.")

    fig, axes = plt.subplots(p, n, figsize=(3 * n, 3 * p))
    axes = np.array(axes)
    if p == 1:
        axes = axes.reshape(1, n)
    elif n == 1:
        axes = axes.reshape(p, 1)

    for r, patches in enumerate(rows):
        for i in range(n):
            ax = axes[r, i]
            patch = patches[i]

            # Determine colormap and handle grayscale
            if cmap is not None:
                use_cmap = cmap
            elif patch.ndim == 2:
                use_cmap = 'gray'
            elif patch.ndim == 3 and patch.shape[-1] == 1:
                patch = patch.squeeze(-1)
                use_cmap = 'gray'
            else:
                use_cmap = None

            ax.imshow(patch, cmap=use_cmap)
            ax.axis('off')

            # Column titles on first row
            if r == 0 and titles is not None:
                ax.set_title(titles[i], fontsize=10)

        # Row titles on first column of each row
        if row_titles is not None:
            axes[r, 0].set_title(row_titles[r], fontsize=12, pad=10)

    plt.tight_layout()
    plt.show()
