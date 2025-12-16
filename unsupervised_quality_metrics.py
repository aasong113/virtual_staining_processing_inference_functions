import os
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm
import lpips
import torch
from cleanfid import fid
from datetime import datetime
from PIL import Image
import tempfile
import shutil
from torch_fidelity import calculate_metrics
import random
import matplotlib.pyplot as plt


import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def plot_random_images(folder1, folder2, n, label1, label2):
    # Ensure the folders exist
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        raise ValueError("One of the specified folders does not exist.")

    # Get lists of image filenames in both folders
    images_folder1 = [f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif'))]
    images_folder2 = [f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif'))]
    
    # Check if we have enough images
    if len(images_folder1) < n or len(images_folder2) < n:
        raise ValueError("Not enough images in one or both folders.")

    # Randomly select n images from each folder
    selected_images_folder1 = random.sample(images_folder1, n)
    selected_images_folder2 = random.sample(images_folder2, n)

    # Create a figure with 2 rows and n columns
    fig, axes = plt.subplots(2, n, figsize=(15, 6))

    # Plot images from folder1
    for i, img_file in enumerate(selected_images_folder1):
        img_path = os.path.join(folder1, img_file)
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')  # Hide axes
        axes[0, i].set_title(label1, fontsize=12, pad=10)  # Set title for the first row

    # Plot images from folder2
    for i, img_file in enumerate(selected_images_folder2):
        img_path = os.path.join(folder2, img_file)
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')  # Hide axes
        axes[1, i].set_title(label2, fontsize=12, pad=10)  # Set title for the second row

    plt.tight_layout()
    plt.show()

# Example usage
# plot_random_images('path/to/folder1', 'path/to/folder2', n=5, label1='Folder 1 Images', label2='Folder 2 Images')
# Example usage
# plot_random_images('path/to/folder1', 'path/to/folder2', n=5)

def convert_tif_folder_to_png_temp(original_dir):
    temp_dir = tempfile.mkdtemp()
    for fname in os.listdir(original_dir):
        if fname.lower().endswith('.tif'):
            img = Image.open(os.path.join(original_dir, fname)).convert('RGB')
            png_name = os.path.splitext(fname)[0] + '.png'
            img.save(os.path.join(temp_dir, png_name))
        elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(os.path.join(original_dir, fname), os.path.join(temp_dir, fname))
    return temp_dir

def compute_virtual_staining_metrics(fake_dir, real_dir, output_dir=None, experiment_name="virtual_staining"):
    temp_fake_dir = convert_tif_folder_to_png_temp(fake_dir)
    temp_real_dir = convert_tif_folder_to_png_temp(real_dir)

    fake_images = sorted([f for f in os.listdir(temp_fake_dir) if f.lower().endswith(('png', 'jpg'))])
    real_images = sorted([f for f in os.listdir(temp_real_dir) if f.lower().endswith(('png', 'jpg'))])

    min_len = min(len(fake_images), len(real_images))
    fake_images = fake_images[:min_len]
    real_images = real_images[:min_len]

    assert len(fake_images) == len(real_images), "Mismatch in number of images"

    psnr_list, ssim_list, lpips_list = [], [], []
    lpips_model = lpips.LPIPS(net='alex')

    for f_img, r_img in tqdm(zip(fake_images, real_images), total=len(fake_images), desc="Evaluating image pairs"):
        img_fake = imread(os.path.join(temp_fake_dir, f_img)) / 255.0
        img_real = imread(os.path.join(temp_real_dir, r_img)) / 255.0

        if img_fake.ndim == 2:
            img_fake = np.stack([img_fake] * 3, axis=-1)
            img_real = np.stack([img_real] * 3, axis=-1)

        psnr_list.append(psnr(img_real, img_fake, data_range=1.0))
        ssim_list.append(ssim(img_real, img_fake, channel_axis=-1, data_range=1.0))

        img_fake_t = torch.tensor(img_fake).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        img_real_t = torch.tensor(img_real).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            d = lpips_model(img_fake_t, img_real_t).item()
        lpips_list.append(d)

    # FID with cleanfid
    fid_score = fid.compute_fid(temp_fake_dir, temp_real_dir)

    # Torch-fidelity metrics (KID & IS)
    kid_subset_size = min(len(fake_images), len(real_images))
    fidelity_metrics = calculate_metrics(
        input1=temp_real_dir,
        input2=temp_fake_dir,
        cuda=torch.cuda.is_available(),
        isc=True,
        fid=False,
        kid=True,
        verbose=False,
        kid_subset_size=kid_subset_size
    )
    kid_score = fidelity_metrics['kernel_inception_distance_mean']
    inception_score = fidelity_metrics['inception_score_mean']

    if output_dir is None:
        output_dir = temp_fake_dir
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "fake_dir": fake_dir,
        "real_dir": real_dir,
        "num_images": len(fake_images),
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "LPIPS": np.mean(lpips_list),
        "FID": fid_score,
        "KID": kid_score,
        "IS": inception_score
    }

    output_txt = os.path.join(output_dir, experiment_name + "_vHE_metrics.txt")
    with open(output_txt, "w") as f:
        f.write(f"# Virtual Staining Metrics ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Fake folder: {fake_dir}\n")
        f.write(f"Real folder: {real_dir}\n")
        f.write(f"Number of paired images: {results['num_images']}\n\n")
        f.write(f"PSNR:  {results['PSNR']:.4f}\n")
        f.write(f"SSIM:  {results['SSIM']:.4f}\n")
        f.write(f"LPIPS: {results['LPIPS']:.4f}\n")
        f.write(f"FID:   {results['FID']:.4f}\n")
        f.write(f"KID:   {results['KID']:.4f}\n")
        f.write(f"IS:    {results['IS']:.4f}\n")

    print(f"\n✔️ Metrics saved to: {output_txt}")

    shutil.rmtree(temp_fake_dir)
    shutil.rmtree(temp_real_dir)

    return results
