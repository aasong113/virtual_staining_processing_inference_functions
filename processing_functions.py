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
from skimage.transform import resize


########### Data processing


def compute_optimal_stride(img_shape, patch_size, overlap_percent=0.10):
    """
    Computes the optimal stride (x, y) for given image shape and patch size
    to achieve the desired overlap and ensure full coverage.

    Args:
        img_shape (tuple): (height, width) of the image
        patch_size (int): size of the square patch
        overlap_percent (float): desired fractional overlap (e.g., 0.1 for 10%)

    Returns:
        tuple: (x_stride, y_stride)
    """
    import math

    h, w = img_shape

    # Initial stride from overlap percentage
    stride = int(patch_size * (1 - overlap_percent))
    stride = max(1, stride)

    # Compute number of patches along height and width (round up to ensure coverage)
    n_patches_y = math.ceil((h - patch_size) / stride) + 1
    n_patches_x = math.ceil((w - patch_size) / stride) + 1

    # Recompute stride so that last patch ends exactly at (h - patch_size) and (w - patch_size)
    if n_patches_y > 1:
        y_stride = (h - patch_size) // (n_patches_y - 1)
    else:
        y_stride = 0

    if n_patches_x > 1:
        x_stride = (w - patch_size) // (n_patches_x - 1)
    else:
        x_stride = 0

    return y_stride, x_stride

def downsample_images_any_dtype(input_dir, output_dir, factor=4):
    """
    Downsamples all .tif images in a directory by a given factor and saves
    them with the same name to the output directory, preserving dtype and shape.

    Args:
        input_dir (str): Folder with input images.
        output_dir (str): Folder to save downsampled images.
        factor (int): Downsampling factor (e.g., 2 halves width and height).
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.tif', '.tiff')  # You can expand this to include more formats if needed

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(valid_exts):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        img = tifffile.imread(in_path)
        dtype = img.dtype

        # Handle grayscale (H, W) or RGB (H, W, C)
        if img.ndim == 2:
            h, w = img.shape
            new_shape = (h // factor, w // factor)
        elif img.ndim == 3:
            h, w, c = img.shape
            new_shape = (h // factor, w // factor, c)
        else:
            print(f"Skipping unsupported image shape {img.shape} in {fname}")
            continue

        # Resize using skimage, preserve_range to keep values in original range
        img_down = resize(img, new_shape, preserve_range=True, anti_aliasing=True).astype(dtype)

        # Save as original dtype
        tifffile.imwrite(out_path, img_down)

    print(f"Saved downsampled images to: {output_dir}")

def invert_images_to_new_folder(input_folder, output_folder, image_extensions=['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
    """
    Invert all images in the input folder and save them to the output folder using the same filenames.

    Parameters:
        input_folder (str): Path to the input image folder.
        output_folder (str): Path to the output folder to save inverted images.
        image_extensions (list): List of allowed image extensions.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Gather all image paths
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))

    if not image_paths:
        print("❌ No images found in the input folder.")
        return

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Could not read image: {path}")
            continue

        inverted = cv2.bitwise_not(img)

        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, inverted)
        print(f"✅ Saved inverted image: {output_path}")

def extract_patch_key(filename):
    """
    Extracts a unique identifier from filename: img=..._patch_X=..._Y=..._Z=..._P=...
    Returns the key string if matched, else None.
    """
    match = re.search(r'img=\d+_patch_X=\d+_Y=\d+_Z=\d+_P=\d+', filename)
    return match.group(0) if match else None

def combine_trios_to_rgb(bit_folder, blue_folder, green_folder, output_folder):
    """
    Combines matching BIT, MUSE_Blue, and MUSE_Green images into RGB and saves them.

    BIT → R, MUSE_Green → G, MUSE_Blue → B
    Assumes all images are already uint8 and normalized to [0, 255].
    """

    os.makedirs(output_folder, exist_ok=True)

    # Index filenames by shared key
    def index_folder(folder):
        file_dict = {}
        for fname in os.listdir(folder):
            key = extract_patch_key(fname)
            if key:
                file_dict[key] = os.path.join(folder, fname)
        return file_dict

    bit_dict = index_folder(bit_folder)
    blue_dict = index_folder(blue_folder)
    green_dict = index_folder(green_folder)

    # Find shared keys
    common_keys = set(bit_dict) & set(blue_dict) & set(green_dict)
    if not common_keys:
        print("❌ No matching patch keys found.")
        return

    print(f"✅ Found {len(common_keys)} matched image trios.")

    for key in sorted(common_keys):
        bit_img = tifffile.imread(bit_dict[key]).astype(np.uint8)
        green_img = tifffile.imread(green_dict[key]).astype(np.uint8)
        blue_img = tifffile.imread(blue_dict[key]).astype(np.uint8)

        # Stack into RGB: (H, W, 3)
        rgb_img = np.stack([bit_img, blue_img, green_img], axis=-1)

        # Save output
        out_name = f"MUSE_BIT_{key}.tif"
        out_path = os.path.join(output_folder, out_name)
        tifffile.imwrite(out_path, rgb_img)

    print(f"✅ Saved {len(common_keys)} combined RGB images to: {output_folder}")

### Functions for Processing NDPI WSI Data ###

def compute_rgb_mean_std(folder_path, output_file="image_stats_rgb.txt"):
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]

    channel_sum = np.zeros(3, dtype=np.float32)
    channel_squared_sum = np.zeros(3, dtype=np.float32)
    pixel_count = 0

    for fname in tif_files:
        path = os.path.join(folder_path, fname)
        img = tiff.imread(path)

        # Only process RGB images
        if img.ndim != 3 or img.shape[2] != 3:
            continue

        img = img.astype(np.float32)
        h, w, _ = img.shape
        pixel_count += h * w

        for i in range(3):  # R, G, B
            channel = img[:, :, i]
            channel_sum[i] += np.sum(channel)
            channel_squared_sum[i] += np.sum(channel ** 2)

    if pixel_count == 0:
        raise ValueError("No RGB images found.")

    means = channel_sum / pixel_count
    stds = np.sqrt(channel_squared_sum / pixel_count - means ** 2)

    # Print to terminal
    print("\n📊 RGB Channel Statistics (np.float32, no rescaling):")
    print(f"R mean: {means[0]:.2f}, std: {stds[0]:.2f}")
    print(f"G mean: {means[1]:.2f}, std: {stds[1]:.2f}")
    print(f"B mean: {means[2]:.2f}, std: {stds[2]:.2f}")

    # Save to .txt file
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, "w") as f:
        f.write("RGB Channel Statistics (np.float32, no rescaling)\n")
        f.write(f"R mean: {means[0]:.6f}, std: {stds[0]:.6f}\n")
        f.write(f"G mean: {means[1]:.6f}, std: {stds[1]:.6f}\n")
        f.write(f"B mean: {means[2]:.6f}, std: {stds[2]:.6f}\n")

    print(f"\n📁 Results saved to: {output_path}")
    return means, stds

def save_patch_metadata_txt(ndpi_filename, roi_name, patch_size, overlap, img_ext,
                            roi_x1, roi_y1, roi_x2, roi_y2, width, height,
                            output_path="patch_metadata.txt"):
    """
    Save metadata for a patch extraction task into a text file.
    Each parameter is written on a new line.
    """
    with open(output_path, 'w') as f:
        f.write(f"NDPI Filename: {ndpi_filename}\n")
        f.write(f"ROI name: {roi_name}\n")
        f.write(f"Patch size: {patch_size[0]},{patch_size[1]}\n")
        f.write(f"# Pixel overlap: {overlap}\n")
        f.write(f"IMG EXT: {img_ext}\n")
        f.write(f"roi x1: {roi_x1}\n")
        f.write(f"roi_y1: {roi_y1}\n")
        f.write(f"roi_x2: {roi_x2}\n")
        f.write(f"roi_y2: {roi_y2}\n")
        f.write(f"width: {width}\n")
        f.write(f"height: {height}\n")

def split_image_into_patches(image, patch_size, overlap, output_folder, handle="patch", extension="tif"):
    """
    Splits an image into n x n patches with overlap, and saves them to a folder as .tif files.
    Only saves complete patches of the specified size.

    Args:
    - image: PIL image to split
    - patch_size: Size of the patches (e.g., (256, 256))
    - overlap: Overlap between patches (e.g., 50 for 50-pixel overlap)
    - output_folder: Folder to save the patches
    - handle: Prefix for the patch file name
    - extension: File extension to save (default: "tif")

    Returns:
    - None
    """
    os.makedirs(output_folder, exist_ok=True)
    np_image = np.array(image)
    height, width = np_image.shape[0], np_image.shape[1]
    step = patch_size[0] - overlap
    patch_number = 1

    for y in range(0, height - patch_size[0] + 1, step):
        for x in range(0, width - patch_size[1] + 1, step):
            patch = np_image[y:y + patch_size[0], x:x + patch_size[1]]
            patch_pil = Image.fromarray(patch)
            patch_filename = f"{handle}_{patch_number}.{extension}"
            patch_path = os.path.join(output_folder, patch_filename)
            patch_pil.save(patch_path, format='TIFF')  # Explicit TIFF format
            patch_number += 1

def load_roi(file_path):
    # Initialize variables to hold the extracted values
    roi_x1, roi_y1, roi_x2, roi_y2, width, height = None, None, None, None, None, None
    # Open the text file
    with open(file_path, "r") as file:
        # Read each line of the file
        lines = file.readlines()
        # Loop through lines to find and extract the relevant data
        for line in lines:
            if "Top-left" in line:
                # Extract the top-left coordinates
                roi_x1, roi_y1 = map(int, line.split(":")[1].strip()[1:-1].split(","))
            elif "Bottom-right" in line:
                # Extract the bottom-right coordinates
                roi_x2, roi_y2 = map(int, line.split(":")[1].strip()[1:-1].split(","))
            elif "Width" in line:
                # Extract the width
                width = int(line.split(":")[1].strip())
            elif "Height" in line:
                # Extract the height
                height = int(line.split(":")[1].strip())
    # Return the extracted values
    return roi_x1, roi_y1, roi_x2, roi_y2, width, height

### Functions for processing MUSE-BIT Training Data ###

def normalize_and_save_float32_images(image_folder, stats_txt_path, output_folder, pattern='*.tif'):
    """
    Normalize each image using mean/std from a text file and save as float32 TIFFs.

    Parameters:
        image_folder (str): Path to input images.
        stats_txt_path (str): Path to 'image_stats.txt' file with mean/std.
        output_folder (str): Folder to save normalized float32 images.
        pattern (str): File pattern (e.g., '*.tif').
    """
    # Step 1: Load mean and std from text
    with open(stats_txt_path, 'r') as f:
        lines = f.readlines()
    mean = float(lines[0].split(':')[1].strip())
    std = float(lines[1].split(':')[1].strip())

    # Step 2: Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Step 3: Load, normalize, and save each image
    image_paths = glob.glob(os.path.join(image_folder, pattern))
    if not image_paths:
        raise ValueError(f"No files found in {image_folder} matching {pattern}")

    for path in image_paths:
        # Load image as float32
        img_np = tifffile.imread(path).astype(np.float32)

        # Normalize
        norm_img = (img_np - mean) / std

        # Save normalized image as float32 TIFF
        save_path = os.path.join(output_folder, os.path.basename(path))
        tifffile.imwrite(save_path, norm_img.astype(np.float32))

    print(f"✅ Saved {len(image_paths)} normalized float32 images to: {output_folder}")

def compute_mean_std_and_save(image_folder, pattern='*.tif'):
    """
    Computes the mean and std of all images in a folder and saves to image_stats.txt.

    Parameters:
        image_folder (str): Path to the folder containing image patches.
        pattern (str): Glob pattern to match image files (default '*.tif').
    """
    image_paths = glob.glob(os.path.join(image_folder, pattern))
    
    if not image_paths:
        raise ValueError(f"No images found in {image_folder} matching {pattern}")

    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    total_pixels = 0

    for path in image_paths:
        img = Image.open(path)
        img_np = np.asarray(img).astype(np.float32)

        pixel_sum += img_np.sum()
        pixel_squared_sum += (img_np ** 2).sum()
        total_pixels += img_np.size

    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_squared_sum / total_pixels - mean ** 2)

    # Save to image_stats.txt in the same folder
    output_txt_path = os.path.join(image_folder, "image_stats.txt")
    with open(output_txt_path, 'w') as f:
        f.write(f"Mean: {mean:.6f}\n")
        f.write(f"Std: {std:.6f}\n")

    print(f"Saved mean and std to: {output_txt_path}")

def extract_xyz(filename):
    """Extract (X, Y, Z) tuple from filename using regex"""
    match = re.search(r'_X=(\d+)_Y=(\d+)_Z=(\d+)', filename)
    if match:
        return tuple(map(int, match.groups()))
    return None

def index_files_by_xyz(file_list):
    """Return a dict mapping (X, Y, Z) -> filename"""
    return {extract_xyz(f): f for f in file_list if extract_xyz(f) is not None}

def flatfield_correct(image, sigma=50):
    """
    Perform flat-field correction on an image using Gaussian smoothing.
    
    Parameters:
        image: np.ndarray
            Input image (grayscale or multi-channel).
        sigma: float
            Gaussian blur sigma for estimating the background illumination.
    
    Returns:
        corrected: np.ndarray
            Flat-field corrected image as float32.
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    #image = np.asarray(image).astype(np.float32) # ensure image is a numpy array. 
    blurred = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = image / (blurred + 1e-8)  # avoid division by zero
    return corrected

def break_image_into_training_patches(img_type, img_num, img_path, img, output_folder, patch_size, step_size, n_random=0, seed=42):
    """
    Break an image into patches and save metadata for later stitching. Also samples n_random additional full patches.

    Parameters:
        img_path (str): Path to the input image
        img (PIL.Image or np.ndarray): Loaded image
        output_folder (str): Folder to save patches and metadata
        patch_size (int): Size of each square patch
        step_size (int): Stride between patches
        n_random (int): Number of additional random patches to extract
        seed (int): Random seed for reproducibility
    """
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.basename(img_path)
    
    # Parse X, Y, Z coordinates from filename if available
    coords = {}
    for part in base_name.split('_'):
        if '=' in part:
            k, v = part.split('=')
            coords[k] = v.split('.')[0]  # remove extension
    
    img_np = np.array(img)
    n_channels = 1 if img_np.ndim == 2 else img_np.shape[2]
    h, w = img_np.shape[:2]

    patch_counter = 1
    patch_info_list = []

    # Step 1: Grid-based patches
    taken_coords = set()
    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            patch = img_np[y:y+patch_size, x:x+patch_size]
            patch_name = f"{img_type}_img={img_num}_patch_X={coords.get('X','0')}_Y={coords.get('Y','0')}_Z={coords.get('Z','0')}_P={patch_counter}.tif"
            patch_path = os.path.join(output_folder, patch_name)

            Image.fromarray(patch).save(patch_path)
            patch_info_list.append(f"{patch_name},{x},{y},{patch_size},{patch_size}")
            taken_coords.add((x, y))
            patch_counter += 1

    # Step 2: Random patches
    random.seed(seed)
    max_x = w - patch_size
    max_y = h - patch_size
    attempts = 0
    while len(patch_info_list) < patch_counter - 1 + n_random and attempts < n_random * 10:
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Only include if not already in grid patches
        grid_x = (x // step_size) * step_size
        grid_y = (y // step_size) * step_size
        if (grid_x, grid_y) not in taken_coords:
            patch = img_np[y:y+patch_size, x:x+patch_size]
            patch_name = f"{img_type}_img={img_num}_patch_X={coords.get('X','0')}_Y={coords.get('Y','0')}_Z={coords.get('Z','0')}_P={patch_counter}.tif"
            patch_path = os.path.join(output_folder, patch_name)

            Image.fromarray(patch).save(patch_path)
            patch_info_list.append(f"{patch_name},{x},{y},{patch_size},{patch_size}")
            patch_counter += 1
        attempts += 1

    # Save metadata
    metadata_file = os.path.join(output_folder, f"{base_name}_patches_stitch_metadata.txt")
    with open(metadata_file, "w") as f:
        f.write(f"# Original image: {base_name}\n")
        f.write(f"# Original size: width={w}, height={h}\n")
        f.write(f"# Patch size: {patch_size}\n")
        f.write(f"# Step size: {step_size}\n")
        f.write(f"# Channels: {n_channels}\n")
        f.write("# Format: patch_filename,x,y,width,height\n")
        for info in patch_info_list:
            f.write(info + "\n")

    print(f"Saved {patch_counter-1} patches and metadata to {metadata_file}")


##### Reconstruct Images from Patches

def reconstruct_image_from_patches_preprocess(metadata_path, patch_folder, return_as_array=False):
    """
    Reconstruct the original image from patches using metadata.
    
    Parameters:
        metadata_path (str): Path to the patch metadata file.
        patch_folder (str): Folder containing the patch images.
        return_as_array (bool): If True, return reconstructed image as NumPy array.

    Returns:
        Reconstructed image (as NumPy array if return_as_array=True)
    """
    with open(metadata_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
    
    # Parse image and patch metadata from headers
    with open(metadata_path, 'r') as f:
        headers = f.readlines()
    
    w = int([line for line in headers if "width=" in line][0].split("width=")[1].split(",")[0])
    h = int([line for line in headers if "height=" in line][0].split("height=")[1])

    # Assume grayscale for now, change to 3 for RGB if needed
    canvas = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for line in lines:
        parts = line.split(',')
        patch_name, x, y, pw, ph = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        patch_path = os.path.join(patch_folder, patch_name)
        
        patch = tifffile.imread(patch_path).astype(np.float32)
        if patch.ndim == 3 and patch.shape[2] == 3:
            raise ValueError("This code currently supports grayscale patches only.")
        
        canvas[y:y+ph, x:x+pw] += patch
        weight_map[y:y+ph, x:x+pw] += 1

    # Avoid divide-by-zero
    weight_map[weight_map == 0] = 1
    reconstructed = canvas / weight_map
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    if return_as_array:
        return reconstructed
    else:
        return Image.fromarray(reconstructed)
    
def show_three_images(img1, img2, img3, subtitles, main_title="Images"):
    """
    Display 3 images in a row with individual subtitles and a main title.

    Parameters:
        img1, img2, img3: PIL Images or numpy arrays
        subtitles (list of str): List of 3 subtitles
        main_title (str): Title for the whole figure
    """
    images = [img1, img2, img3]

    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        # Convert PIL Image to numpy array if necessary
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(subtitles[i])
        ax.axis('off')

    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def get_txt_files_sorted_by_xyz(folder):
    """
    Get all .txt files in `folder` and sort them by X, Y, Z coordinates in the filename.
    Example filename: crop_darksect_reg_MUSE_blue_kidney_normal_1_X=2_Y=4_Z=2_expTime=2000ms.tif_patches_stitch_metadata.txt
    """
    # Get all .txt files
    txt_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]

    def extract_xyz(path):
        """Extracts (X, Y, Z) as integers from the filename."""
        basename = os.path.basename(path)
        match = re.search(r'X=(\d+)_Y=(\d+)_Z=(\d+)', basename)
        if match:
            return tuple(int(v) for v in match.groups())
        else:
            return (-1, -1, -1)  # If no match, put at beginning

    # Sort by X, then Y, then Z
    txt_files_sorted = sorted(txt_files, key=extract_xyz)
    return txt_files_sorted

def get_matching_image_paths(folders, pattern="img=(\d+)_patch_X=(\d+)_Y=(\d+)_Z=(\d+)_P=(\d+)"):
    """
    folders: list of folder paths [folder1, folder2, folder3]
    pattern: regex to extract img, X, Y, Z, P from filename
    returns: list of tuples [(path1, path2, path3), ...] where each tuple is matched
    """
    # Build dicts mapping the key (img,X,Y,Z,P) -> file path
    folder_dicts = []
    regex = re.compile(pattern)
    
    for folder in folders:
        mapping = {}
        for f in os.listdir(folder):
            m = regex.search(f)
            if m:
                key = tuple(map(int, m.groups()))
                mapping[key] = os.path.join(folder, f)
        folder_dicts.append(mapping)
    
    # Find common keys across all folders
    common_keys = set(folder_dicts[0].keys())
    for d in folder_dicts[1:]:
        common_keys = common_keys.intersection(d.keys())
    
    # Build list of matched paths
    matched_paths = [tuple(d[key] for d in folder_dicts) for key in sorted(common_keys)]
    
    return matched_paths

def compute_gradient_image(img, show=False):
    """
    Compute the gradient magnitude image using Sobel operator.

    Args:
        image_path (str): Path to the input grayscale image.
        show (bool): Whether to display the gradient image. Default is False.

    Returns:
        grad_mag_norm (np.ndarray): Normalized gradient magnitude image (uint8).
    """
    # Load grayscale image
    
    # Compute gradients along x and y
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255 for display
    grad_mag_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Display if requested
    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(grad_mag_norm, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.axis('off')
        plt.show()
    
    return grad_mag_norm

def extract_key(filename):
    """
    Extracts the matching key from filename like:
    img=37_patch_X=2_Y=4_Z=2_P=1.tif -> img=37_X=2_Y=4_Z=2_P=1
    """
    match = re.search(r'(img=\d+)_patch_(X=\d+_Y=\d+_Z=\d+_P=\d+)', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

def get_matching_pairs(folder1, folder2):
    # Build dict of {key: path} for folder1
    dict1 = {extract_key(f): os.path.join(folder1, f) 
             for f in os.listdir(folder1) if extract_key(f)}
    # Build dict of {key: path} for folder2
    dict2 = {extract_key(f): os.path.join(folder2, f) 
             for f in os.listdir(folder2) if extract_key(f)}
    # Find keys present in both
    common_keys = list(set(dict1.keys()) & set(dict2.keys()))
    # Return list of tuples (file1, file2)
    return [(dict1[k], dict2[k]) for k in common_keys]

def show_random_pairs(pairs, n=2):
    sample_pairs = random.sample(pairs, min(n, len(pairs)))
    for idx, (img1_path, img2_path) in enumerate(sample_pairs):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        # Convert BGR to RGB for plt
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img1)
        plt.title("Folder 1")
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(img2)
        plt.title("Folder 2")
        plt.axis('off')

        plt.show()


def invert_uint8_images(input_folder, output_folder):
    """
    Invert all uint8 images in `input_folder` and save to `output_folder` with the same filenames.
    """
    os.makedirs(output_folder, exist_ok=True)

    # List all image files
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        # Read image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue  # skip non-images

        # Only process uint8 images
        if img.dtype == np.uint8:
            inverted_img = 255 - img
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, inverted_img)


import os
import re
import glob
import numpy as np
import tifffile
from PIL import Image

def reconstruct_image_from_patches_JUST_BIT(metadata_path, patch_folder, img_num, return_as_array=False):
    """
    Reconstruct the original image using metadata coordinates (x, y, width, height),
    loading patches from patch_folder by basename pattern:
      *img=<img_num>_P=<P>.(png|tif|tiff)  or  *img_<img_num>_P=<P>.(png|tif|tiff)

    The metadata filenames are ignored except for extracting P numbers and coordinates.

    Parameters:
        metadata_path (str): Path to metadata file.
        patch_folder (str): Folder containing patch images.
        img_num (int): Image number to reconstruct (e.g., 23 -> use img=23_P=... or img_23_P=...).
        return_as_array (bool): If True, return NumPy array; else return PIL.Image.

    Returns:
        np.ndarray or PIL.Image.Image
    """
    # Read metadata file
    with open(metadata_path, 'r') as f:
        lines = f.read().splitlines()

    # Parse original size (width, height) from header
    w = h = None
    for line in lines:
        if "Original size:" in line:
            mw = re.search(r'width=(\d+)', line)
            mh = re.search(r'height=(\d+)', line)
            if mw and mh:
                w = int(mw.group(1)); h = int(mh.group(1))
            break
    if w is None or h is None:
        raise ValueError("Failed to parse Original size (width/height) from metadata.")

    # Collect entries: (P, x, y, pw, ph) from non-comment lines
    entries = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(',')
        if len(parts) < 5:
            continue
        fname, x, y, pw, ph = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        mP = re.search(r'P=(\d+)', fname)
        if not mP:
            continue
        P = int(mP.group(1))
        entries.append((P, x, y, pw, ph))
    if not entries:
        raise ValueError("No patch entries found in metadata (no P=... lines).")

    # Helper: find actual file for given img_num and P
    def find_patch_file(folder, img_num, P):
        patterns = [
            f'*img={img_num}_P={P}',
            f'*img_{img_num}_P={P}',
        ]
        for pat in patterns:
            for ext in ('png', 'tif', 'tiff'):
                matches = glob.glob(os.path.join(folder, pat + f'.{ext}'))
                if matches:
                    return matches[0]
        # Fallback: any extension if present
        for pat in patterns:
            matches = glob.glob(os.path.join(folder, pat + '.*'))
            if matches:
                return matches[0]
        return None

    # Helper: load raw patch as numpy array
    def load_raw(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.tif', '.tiff'):
            return tifffile.imread(path)
        elif ext == '.png':
            with Image.open(path) as im:
                return np.array(im)
        else:
            raise ValueError(f"Unsupported patch extension: {ext}")

    # Determine mode (color vs gray) from the first available patch
    first_path = None
    for P, _, _, _, _ in sorted(entries):
        pth = find_patch_file(patch_folder, img_num, P)
        if pth:
            first_path = pth
            break
    if not first_path:
        raise FileNotFoundError(f"No patch files found for img={img_num} in {patch_folder}.")
    first_arr = load_raw(first_path)
    mode = 'color' if (first_arr.ndim == 3 and first_arr.shape[-1] >= 3) else 'gray'

    # Prepare canvas and weight map
    canvas = np.zeros((h, w, 3), dtype=np.float32) if mode == 'color' else np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    def prepare_patch(raw):
        # Convert raw array to chosen mode
        if raw.ndim == 2:
            gray = raw.astype(np.float32)
            return np.stack([gray, gray, gray], axis=-1) if mode == 'color' else gray
        if raw.ndim == 3:
            c = raw.shape[-1]
            if c == 1:
                gray = raw[..., 0].astype(np.float32)
                return np.stack([gray, gray, gray], axis=-1) if mode == 'color' else gray
            if c >= 3:
                rgb = raw[..., :3].astype(np.float32)
                if mode == 'gray':
                    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
                return rgb
        squeezed = np.squeeze(raw)
        if squeezed.ndim not in (2, 3):
            raise ValueError(f"Unsupported patch shape {raw.shape}")
        return prepare_patch(squeezed)

    used = 0
    missing = []
    for P, x, y, pw, ph in entries:
        patch_path = find_patch_file(patch_folder, img_num, P)
        if patch_path is None:
            missing.append(P)
            continue
        raw = load_raw(patch_path)
        patch = prepare_patch(raw)

        H, Wp = (patch.shape[:2] if mode == 'color' else patch.shape)
        if H < ph or Wp < pw:
            raise ValueError(f"Patch size {patch.shape} smaller than metadata ({ph}, {pw}) for {patch_path}")
        if H != ph or Wp != pw:
            patch = (patch[:ph, :pw, :] if mode == 'color' else patch[:ph, :pw])

        if mode == 'color':
            canvas[y:y+ph, x:x+pw, :] += patch
        else:
            canvas[y:y+ph, x:x+pw] += patch
        weight_map[y:y+ph, x:x+pw] += 1
        used += 1

    if missing:
        raise FileNotFoundError(f"Missing patches for img={img_num}, P values: {missing}")
    if used == 0:
        raise ValueError(f"No patches were used for reconstruction for img={img_num}.")

    # Normalize and finalize
    if mode == 'color':
        reconstructed = canvas / weight_map[..., None]
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return reconstructed if return_as_array else Image.fromarray(reconstructed, mode='RGB')
    else:
        reconstructed = canvas / weight_map
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return reconstructed if return_as_array else Image.fromarray(reconstructed, mode='L')
    

import os
import re

def count_unique_img_indices(folder):
    """
    Scan a folder and return:
      - set of unique image indices (from 'img=NN' or 'img_NN')
      - count of unique indices
      - dict mapping each index -> set of unique base names up to img=NN/img_NN
      - list of unique base names (deduped), e.g., '..._img=4'

    Example filenames:
      crypts_..._img=0_P=1.png
      crypts_..._img=0_P=2.tif
      crypts_..._img=4_P=6.png

    Unique base names:
      crypts_..._img=0
      crypts_..._img=4
    """
    # Match img index (img=NN or img_NN)
    idx_pattern = re.compile(r'img[=_](\d+)')
    # Capture base up to img=NN/img_NN
    base_pattern = re.compile(r'^(.*?img[=_]\d+)')

    img_indices = set()
    names_by_index = {}  # int -> set[str]

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue

        m_idx = idx_pattern.search(fname)
        if not m_idx:
            continue

        idx = int(m_idx.group(1))
        img_indices.add(idx)

        m_base = base_pattern.match(fname)
        if m_base:
            base_name = m_base.group(1)  # e.g., 'crypts_..._img=4'
            names_by_index.setdefault(idx, set()).add(base_name)

    # Flat list of unique base names (sorted for reproducibility)
    unique_base_names = sorted({b for bases in names_by_index.values() for b in bases})
    return img_indices, len(img_indices), names_by_index, unique_base_names

# Example usage:
# indices, count, names_map, unique_names = count_unique_img_indices("/path/to/folder")
# print(indices, count)
# print(unique_names)            # ['..._img=0', '..._img=4']
# print(names_map.get(0, set())) # {'..._img=0'}