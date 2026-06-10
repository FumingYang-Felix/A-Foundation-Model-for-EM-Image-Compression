"""
Data loading and preprocessing utilities for high-resolution images.
Handles large images by tiling into fixed-size patches.
"""

import os
import re
import glob
import zipfile
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image


def natural_sort_key(s):
    """Natural sorting key for filenames (handles numeric parts correctly)."""
    return [int(t) if t.isdigit() else t.lower() 
            for t in re.split(r'(\d+)', os.path.basename(s))]


def extract_dataset_zip(zip_path, extract_dir):
    """
    Extract a dataset zip file to a local directory.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to
        
    Returns:
        Path to extraction directory
    """
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print(f"Extraction complete.")
    return extract_dir


def collect_image_files(root_dir, extensions=None):
    """
    Recursively collect all image files from a directory.
    
    Args:
        root_dir: Root directory to search
        extensions: Tuple of valid extensions (default: common image formats)
        
    Returns:
        List of image file paths, naturally sorted
    """
    if extensions is None:
        extensions = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG", 
                     ".tif", ".tiff", ".TIF", ".TIFF")
    
    all_files = [
        p for p in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
        if os.path.splitext(p)[1] in extensions
    ]
    all_files = sorted(all_files, key=natural_sort_key)
    return all_files


def load_and_preprocess_image(path, target_channels=1):
    """
    Load and preprocess an image for VQ-VAE training.
    
    Handles PNG, JPG, and TIFF formats. Converts to grayscale if needed,
    normalizes to [-0.5, 0.5] range.
    
    Args:
        path: Path to image file (can be tf.Tensor or string)
        target_channels: Number of channels (1 for grayscale, 3 for RGB)
        
    Returns:
        Preprocessed image tensor [H, W, C] in range [-0.5, 0.5]
    """
    path = tf.convert_to_tensor(path)
    ext = tf.strings.lower(tf.strings.regex_replace(path, r'^.*\.', '.'))
    img_bin = tf.io.read_file(path)

    def decode_png_jpg():
        img = tf.io.decode_image(img_bin, channels=0, expand_animations=False)
        return img

    def decode_tiff_py(p):
        """Fallback for TIFF using PIL."""
        arr = np.array(Image.open(p.decode('utf-8')))
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr

    # Check if TIFF
    is_tiff = tf.reduce_any([tf.equal(ext, s) for s in [".tif", ".tiff"]])
    img = tf.cond(
        is_tiff,
        lambda: tf.numpy_function(decode_tiff_py, [path], Tout=tf.uint8),
        lambda: decode_png_jpg()
    )
    img.set_shape([None, None, None])  # H W C

    # Convert to target channels
    c = tf.shape(img)[-1]
    if target_channels == 1:
        img = tf.cond(
            tf.equal(c, 1),
            lambda: img,
            lambda: tf.image.rgb_to_grayscale(img)
        )
    
    # Normalize to [-0.5, 0.5]
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]
    img = img - 0.5  # [-0.5, 0.5]
    return img


def pad_to_multiple(img, multiple=1024):
    """
    Pad image to be a multiple of given size using symmetric padding.
    
    Args:
        img: Image tensor [H, W, C]
        multiple: Size multiple (e.g., 1024 for tiling)
        
    Returns:
        Padded image tensor
    """
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    img = tf.pad(img, [[0, pad_h], [0, pad_w], [0, 0]], mode='SYMMETRIC')
    return img


def extract_tiles(img, tile_size=1024):
    """
    Extract non-overlapping tiles from an image.
    
    Args:
        img: Image tensor [H, W, C] (should be padded to multiple of tile_size)
        tile_size: Size of each square tile
        
    Returns:
        Tensor of tiles [N, tile_size, tile_size, C]
    """
    ks = [1, tile_size, tile_size, 1]
    st = [1, tile_size, tile_size, 1]
    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=ks,
        strides=st,
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    n_h = tf.shape(patches)[1]
    n_w = tf.shape(patches)[2]
    patches = tf.reshape(patches, [n_h * n_w, tile_size, tile_size, -1])
    return patches


def create_tiled_dataset(file_paths, tile_size=1024, batch_size=2, 
                         shuffle=False, shuffle_buffer=2048):
    """
    Create a tf.data.Dataset that loads images and tiles them.
    
    Args:
        file_paths: List of image file paths
        tile_size: Size of each tile (must be power of 2)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        shuffle_buffer: Buffer size for shuffling
        
    Returns:
        tf.data.Dataset yielding (tile, tile) pairs for autoencoding
    """
    AUTO = tf.data.AUTOTUNE
    
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)

    def map_decode_and_tile(path):
        img = load_and_preprocess_image(path)
        img = pad_to_multiple(img, tile_size)
        tiles = extract_tiles(img, tile_size)
        tile_ds = tf.data.Dataset.from_tensor_slices(tiles)
        return tile_ds

    # Interleave: load multiple images in parallel and flatten their tiles
    ds = ds.interleave(
        map_decode_and_tile,
        cycle_length=8,
        num_parallel_calls=AUTO,
        deterministic=False
    )
    
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    
    # Autoencoder: input = target
    ds = ds.map(lambda x: (x, x), num_parallel_calls=AUTO)
    ds = ds.batch(batch_size).prefetch(AUTO)
    return ds


def reconstruct_full_image(model, image_path, tile_size=1024, batch_size=4):
    """
    Reconstruct a full-resolution image using a trained model by tiling.
    
    Args:
        model: Trained VQ-VAE model
        image_path: Path to input image
        tile_size: Size of tiles to process
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (original_image, reconstructed_image) as numpy arrays [H, W] in [0, 1]
    """
    # Load and preprocess
    img_t = load_and_preprocess_image(tf.constant(image_path))
    img = img_t.numpy()
    h0, w0 = img.shape[:2]

    # Pad to multiple of tile_size
    pad_h = (tile_size - (h0 % tile_size)) % tile_size
    pad_w = (tile_size - (w0 % tile_size)) % tile_size
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='symmetric')
    h, w = img_pad.shape[:2]

    n_h = h // tile_size
    n_w = w // tile_size

    # Process tiles in batches
    recons_tiles = []
    batch_buf = []
    
    for ih in range(n_h):
        for iw in range(n_w):
            y = ih * tile_size
            x = iw * tile_size
            patch = img_pad[y:y+tile_size, x:x+tile_size, :]
            batch_buf.append(patch)
            
            if len(batch_buf) == batch_size:
                xb = tf.convert_to_tensor(np.stack(batch_buf, axis=0), dtype=tf.float32)
                yb = model(xb, training=False).numpy()
                recons_tiles.append(yb)
                batch_buf = []
    
    # Process remaining tiles
    if len(batch_buf) > 0:
        xb = tf.convert_to_tensor(np.stack(batch_buf, axis=0), dtype=tf.float32)
        yb = model(xb, training=False).numpy()
        recons_tiles.append(yb)

    # Stitch tiles back together
    recons = np.concatenate(recons_tiles, axis=0) if len(recons_tiles) else \
             np.zeros((0, tile_size, tile_size, 1), np.float32)
    
    canvas = np.zeros((h, w, 1), dtype=np.float32)
    k = 0
    for ih in range(n_h):
        for iw in range(n_w):
            y = ih * tile_size
            x = iw * tile_size
            canvas[y:y+tile_size, x:x+tile_size, :] = recons[k]
            k += 1

    # Crop to original size
    canvas = canvas[:h0, :w0, :]
    orig = img[:h0, :w0, :]
    
    # Convert to [0, 1] range
    orig_01 = np.clip(orig + 0.5, 0.0, 1.0)
    recon_01 = np.clip(canvas + 0.5, 0.0, 1.0)
    
    return orig_01[..., 0], recon_01[..., 0]

