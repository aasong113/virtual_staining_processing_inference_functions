import numpy as np
import os
import tifffile as tiff


def flatfield_correct_OGdtype(image, sigma=50):
    """
    Perform flat-field correction on an image using Gaussian smoothing.
    
    Parameters:
        image: np.ndarray
            Input image (grayscale or multi-channel).
        sigma: float
            Gaussian blur sigma for estimating the background illumination.
    
    Returns:
        corrected: np.ndarray
            Flat-field corrected image with same dtype as input.
    """
    original_dtype = image.dtype
    image_float = np.ascontiguousarray(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(image_float, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = image_float / (blurred + 1e-8)  # avoid division by zero

    # Rescale to original intensity range and dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        corrected = corrected * image_float.max()  # rescale to original max
        corrected = np.clip(corrected, info.min, info.max).astype(original_dtype)
    else:
        corrected = np.clip(corrected, 0.0, 1.0).astype(original_dtype)

    return corrected

def guoy_phase_shift_tomography(output_path, BIT_stack_path, z_spacing=1):
    """
    Create a tomographic reconstruction by subtracting each image slice with a slice z_spacing apart.

    Args:
        output_path (str): Directory to save the resulting .tif stack.
        BIT_stack_path (str): Path to input 3D image stack (.tif).
        z_spacing (int): Number of slices between subtraction.
    """
    # Load input stack
    img_stack = tiff.imread(BIT_stack_path)
    img_stack = np.array(img_stack, dtype=img_stack.dtype)
    num_slices = img_stack.shape[0]

    # Create subtraction stack
    sub_stack = []
    for i in range(num_slices - z_spacing):

        img_z = img_stack[i].astype(np.int32)
        img_zp = img_stack[i + z_spacing].astype(np.int32)

        #img_z = flatfield_correct_OGdtype(img_z)
        #img_zp = flatfield_correct_OGdtype(img_zp)

        sub = img_z - img_zp
        # Clip or rescale if necessary, then cast back to original dtype
        sub = np.clip(sub + np.iinfo(img_stack.dtype).max // 2, 0, np.iinfo(img_stack.dtype).max)
        sub_stack.append(sub.astype(img_stack.dtype))

    sub_stack = np.stack(sub_stack)

    # Create output filename
    base_name = os.path.splitext(os.path.basename(BIT_stack_path))[0]
    out_name = f"flatfield_tomography_z={z_spacing}slice_{base_name}.tif"
    out_path = os.path.join(output_path, out_name)

    # Save stack
    tiff.imwrite(out_path, sub_stack)
    print(f"Saved tomography stack to: {out_path}")

import cv2
import numpy as np


def optical_flow(output_path, BIT_stack_path, z_spacing=1, flow_method="farneback"):
    """
    Create a tomographic reconstruction by subtracting each image slice with a slice z_spacing apart.

    Args:
        output_path (str): Directory to save the resulting .tif stack.
        BIT_stack_path (str): Path to input 3D image stack (.tif).
        z_spacing (int): Number of slices between subtraction.
    """
    # Load input stack
    img_stack = tiff.imread(BIT_stack_path)
    img_stack = np.array(img_stack, dtype=img_stack.dtype)
    num_slices = img_stack.shape[0]

    # Create subtraction stack
    sub_stack = []
    sub_stack_dx = []
    sub_stack_dy = []
    for i in range(num_slices - z_spacing):

        img_z = img_stack[i].astype(np.int32)
        img_zp = img_stack[i + z_spacing].astype(np.int32)

        #img_z = flatfield_correct_OGdtype(img_z)
        #img_zp = flatfield_correct_OGdtype(img_zp)

        
        # Compute optical flow
        flow = compute_optical_flow(img_z, img_zp, flow_method=flow_method)
        dx = flow[..., 0]  # horizontal flow
        dy = flow[..., 1]  # vertical flow
        magnitude = np.sqrt(dx**2 + dy**2) # magnitude of flow vector

        if i % 10 == 0:
            plt.figure(figsize=(18, 6))

            grad_mag_z, grad_dir_z = compute_gradient_image(img_z)
            grad_mag_zp, grad_dir_zp = compute_gradient_image(img_zp)


            plt.subplot(2, 3, 1)
            plt.title("img_z (t)")
            plt.imshow(img_z, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.title("img_zp (t+1)")
            plt.imshow(img_zp, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 3)


            flow = compute_optical_flow(grad_dir_z, grad_dir_zp, flow_method=flow_method)
            dx = flow[..., 0]  # horizontal flow
            dy = flow[..., 1]  # vertical flow
            magnitude = np.sqrt(dx**2 + dy**2) # magnitude of flow vector

            dx = flow[..., 0]  # horizontal flow
            dy = flow[..., 1]  # vertical flow
            h, w = flow.shape[:2]
            magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
                    
            # Normalize
            mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            angle_norm = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)
            dx_norm = cv2.normalize(dx, None, 0, 255, cv2.NORM_MINMAX)
            dy_norm = cv2.normalize(dy, None, 0, 255, cv2.NORM_MINMAX)

            # HSV encoding: Hue=angle, Saturation=255, Value=magnitude
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV uses [0,179] for Hue
            hsv[..., 1] = 255
            hsv[..., 2] = mag_norm.astype(np.uint8)

            # Convert HSV to RGB
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            plt.imshow(mag_norm, cmap='magma')
            plt.title("Optical Flow Magnitude")
            plt.colorbar()
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.title("img_z (t) gradient magnitude")
            plt.imshow(grad_mag_zp, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.title("img_z (t) Phase direction")
            plt.imshow(grad_dir_zp, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.title("Angle")
            plt.imshow(angle_norm, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        #break

        
        sub_stack.append(mag_norm.astype(img_stack.dtype))
        sub_stack_dx.append(dx_norm.astype(img_stack.dtype))
        sub_stack_dy.append(dy_norm.astype(img_stack.dtype))

    sub_stack = np.stack(sub_stack)
    sub_stack_dx = np.stack(sub_stack_dx)
    sub_stack_dy = np.stack(sub_stack_dy)

    # Create output filename
    base_name = os.path.splitext(os.path.basename(BIT_stack_path))[0]
    out_name = f"opticalFlow={flow_method}_zSpacing={z_spacing}slice_{base_name}.tif"
    out_name_dx = f"opticalFlowDX={flow_method}_zSpacing={z_spacing}slice_{base_name}.tif"
    out_name_dy = f"opticalFlowDY={flow_method}_zSpacing={z_spacing}slice_{base_name}.tif"
    out_path = os.path.join(output_path, out_name)
    out_path_dx = os.path.join(output_path, out_name_dx)
    out_path_dy = os.path.join(output_path, out_name_dy)

    # Save stack
    tiff.imwrite(out_path, sub_stack)
    tiff.imwrite(out_path_dx, sub_stack_dx)
    tiff.imwrite(out_path_dy, sub_stack_dy)
    print(f"Saved tomography stack to: {out_path}")



def compute_optical_flow(img1, img2, flow_method="farneback", **kwargs):
    """
    Compute optical flow between two images using various OpenCV methods.

    Parameters
    ----------
    img1 : np.ndarray
        Image at time t
    img2 : np.ndarray
        Image at time t+1
    flow_method : str
        One of:
        ['farneback', 'tvl1', 'dis', 'deepflow', 'pcaflow', 'simpleflow', 'lk']
    **kwargs : dict
        Optional method-specific parameters

    Returns
    -------
    flow : np.ndarray
        Dense flow (H, W, 2) for dense methods
        OR
        (points_prev, points_next, status) for LK
    """

    # Convert to grayscale if needed
    if img1.ndim == 3:
        img1 = img1.astype(np.float32)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = img2.astype(np.float32)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    #img1 = img1.astype(np.uint8)
    #img2 = img2.astype(np.uint8)

    # =========================
    # Dense optical flow
    # =========================
    if flow_method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            pyr_scale=kwargs.get("pyr_scale", 0.5),
            levels=kwargs.get("levels", 3),
            winsize=kwargs.get("winsize", 15),
            iterations=kwargs.get("iterations", 3),
            poly_n=kwargs.get("poly_n", 5),
            poly_sigma=kwargs.get("poly_sigma", 1.2),
            flags=kwargs.get("flags", 0),
        )

    elif flow_method == "tvl1":
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()
        flow = tvl1.calc(img1, img2, None)

    elif flow_method == "dis":
        dis = cv2.DISOpticalFlow_create(
            kwargs.get("preset", cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        )
        flow = dis.calc(img1, img2, None)

    elif flow_method == "deepflow":
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        flow = deepflow.calc(img1, img2, None)

    elif flow_method == "pcaflow":
        pcaflow = cv2.optflow.createOptFlow_PCAFlow()
        flow = pcaflow.calc(img1, img2, None)

    else:
        raise ValueError(
            f"Unknown flow_method '{flow_method}'. "
            "Choose from ['farneback','tvl1','dis','deepflow','pcaflow','simpleflow']"
        )

    return flow


def compute_gradient_image(image):
    """
    Compute gradient magnitude and direction using Sobel operator.

    Parameters:
        image (np.ndarray): Input 2D image (grayscale).

    Returns:
        grad_mag (np.ndarray): Gradient magnitude.
        grad_dir (np.ndarray): Gradient direction (in degrees).
    """
    # Ensure float32 for precision
    img = image.astype(np.float32)

    # Compute gradients in X and Y direction
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Compute magnitude and direction
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_dir = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    return grad_mag, grad_dir

# flow shape: [H, W, 2]
# flow[...,0] = dx, flow[...,1] = dy
import os
import re
import numpy as np
from skimage.io import imread

def load_image_stack_by_img_index(folder_path, ext='.tif'):
    """
    Load grayscale or RGB images from a folder into a stack, ordered by the index after 'img=' in the filename.

    Parameters:
        folder_path (str): Path to the folder containing images.
        ext (str): Extension of image files to load (e.g., '.tif').

    Returns:
        np.ndarray: Image stack of shape (N, H, W) for grayscale or (N, H, W, C) for RGB.
    """
    def extract_img_index(filename):
        match = re.search(r'img=(\d+)', filename)
        return int(match.group(1)) if match else -1

    # List and sort image files by the extracted img= index
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(ext)],
        key=extract_img_index
    )

    # Load images into stack
    image_list = []
    for f in files:
        img = imread(os.path.join(folder_path, f))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # Convert grayscale to shape (H, W, 1)
        image_list.append(img)

    # Stack into (N, H, W, C)
    image_stack = np.stack(image_list, axis=0)

    # If all images are grayscale (C=1), optionally squeeze the channel dim
    if image_stack.shape[-1] == 1:
        image_stack = np.squeeze(image_stack, axis=-1)  # (N, H, W)

    return image_stack

# Example usage:
# stack = load_image_stack_by_img_index('/path/to/rgb_or_gray_tifs')


def optical_flow_stack(output_path, base_name, img_stack, z_spacing=1, flow_method="farneback"):
    """
    Create a tomographic reconstruction by subtracting each image slice with a slice z_spacing apart.

    Args:
        output_path (str): Directory to save the resulting .tif stack.
        BIT_stack_path (str): Path to input 3D image stack (.tif).
        z_spacing (int): Number of slices between subtraction.
    """
    # Load input stack
    num_slices = img_stack.shape[0]

    # Create subtraction stack
    sub_stack = []
    sub_stack_dx = []
    sub_stack_dy = []
    for i in range(num_slices - z_spacing):

        img_z = img_stack[i].astype(np.int32)
        img_zp = img_stack[i + z_spacing].astype(np.int32)

        #img_z = flatfield_correct_OGdtype(img_z)
        #img_zp = flatfield_correct_OGdtype(img_zp)

        
        # Compute optical flow
        flow = compute_optical_flow(img_z, img_zp, flow_method=flow_method)
        dx = flow[..., 0]  # horizontal flow
        dy = flow[..., 1]  # vertical flow
        magnitude = np.sqrt(dx**2 + dy**2) # magnitude of flow vector

        if i % 10 == 0:
            plt.figure(figsize=(18, 6))

            grad_mag_z, grad_dir_z = compute_gradient_image(img_z)
            grad_mag_zp, grad_dir_zp = compute_gradient_image(img_zp)


            plt.subplot(2, 3, 1)
            plt.title("img_z (t)")
            plt.imshow(img_z, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.title("img_zp (t+1)")
            plt.imshow(img_zp, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 3)


            flow = compute_optical_flow(grad_dir_z, grad_dir_zp, flow_method=flow_method)
            dx = flow[..., 0]  # horizontal flow
            dy = flow[..., 1]  # vertical flow
            magnitude = np.sqrt(dx**2 + dy**2) # magnitude of flow vector

            dx = flow[..., 0]  # horizontal flow
            dy = flow[..., 1]  # vertical flow
            h, w = flow.shape[:2]
            magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
                    
            # Normalize
            mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            angle_norm = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)
            dx_norm = cv2.normalize(dx, None, 0, 255, cv2.NORM_MINMAX)
            dy_norm = cv2.normalize(dy, None, 0, 255, cv2.NORM_MINMAX)

            # HSV encoding: Hue=angle, Saturation=255, Value=magnitude
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV uses [0,179] for Hue
            hsv[..., 1] = 255
            hsv[..., 2] = mag_norm.astype(np.uint8)

            # Convert HSV to RGB
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            plt.imshow(mag_norm, cmap='magma')
            plt.title("Optical Flow Magnitude")
            plt.colorbar()
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.title("img_z (t) gradient magnitude")
            plt.imshow(grad_mag_zp, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.title("img_z (t) Phase direction")
            plt.imshow(grad_dir_zp, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.title("Angle")
            plt.imshow(angle_norm, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        #break

        
        sub_stack.append(mag_norm.astype(img_stack.dtype))
        sub_stack_dx.append(dx_norm.astype(img_stack.dtype))
        sub_stack_dy.append(dy_norm.astype(img_stack.dtype))

    sub_stack = np.stack(sub_stack)
    sub_stack_dx = np.stack(sub_stack_dx)
    sub_stack_dy = np.stack(sub_stack_dy)

    # Create output filename
    out_name = f"opticalFlow={flow_method}_zSpacing={z_spacing}slice_{base_name}.tif"
    out_name_dx = f"opticalFlowDX={flow_method}_zSpacing={z_spacing}slice_{base_name}.tif"
    out_name_dy = f"opticalFlowDY={flow_method}_zSpacing={z_spacing}slice_{base_name}.tif"
    out_path = os.path.join(output_path, out_name)
    out_path_dx = os.path.join(output_path, out_name_dx)
    out_path_dy = os.path.join(output_path, out_name_dy)

    # Save stack
    tiff.imwrite(out_path, sub_stack)
    tiff.imwrite(out_path_dx, sub_stack_dx)
    tiff.imwrite(out_path_dy, sub_stack_dy)
    print(f"Saved tomography stack to: {out_path}")
