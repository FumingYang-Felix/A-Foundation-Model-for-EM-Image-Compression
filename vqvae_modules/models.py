"""
VQ-VAE model architectures for high-resolution image compression.
"""

import math
import tensorflow as tf
from .layers import residual_stack, VectorQuantizerEMA


def build_encoder_to_top(img_size, top_grid, num_hiddens):
    """
    Build encoder that downsamples from img_size to top_grid.
    
    Args:
        img_size: Input image size (e.g., 1024)
        top_grid: Output latent grid size (e.g., 32)
        num_hiddens: Number of hidden channels
        
    Returns:
        Keras Model for encoding
    """
    assert (img_size % top_grid) == 0, f"img_size {img_size} must be divisible by top_grid {top_grid}"
    n_downs = int(math.log2(img_size // top_grid))  # e.g., 1024->32 = 5 downsamples
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))
    x = inputs
    
    # Progressive downsampling with stride-2 convolutions
    chans = [64, 128, 128, 128, num_hiddens][:n_downs]
    for c in chans:
        x = tf.keras.layers.Conv2D(c, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)
    
    # Residual refinement at bottleneck
    x = residual_stack(x, num_hiddens, num_blocks=2)
    
    return tf.keras.Model(inputs, x, name=f'encoder_to_top_{top_grid}')


def build_decoder_top_to_image(img_size, top_grid, in_ch):
    """
    Build decoder that upsamples from top_grid to img_size.
    
    Args:
        img_size: Output image size (e.g., 1024)
        top_grid: Input latent grid size (e.g., 32)
        in_ch: Number of input channels
        
    Returns:
        Keras Model for decoding
    """
    assert (img_size % top_grid) == 0, f"img_size {img_size} must be divisible by top_grid {top_grid}"
    n_ups = int(math.log2(img_size // top_grid))  # e.g., 32->1024 = 5 upsamples
    
    inputs = tf.keras.Input(shape=(top_grid, top_grid, in_ch))
    x = inputs
    
    # Progressive upsampling with transposed convolutions
    ups_ch = [256, 128, 128, 64, 32][:n_ups]
    for c in ups_ch:
        x = tf.keras.layers.Conv2DTranspose(c, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)
    
    # Final output layer
    out = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
    
    return tf.keras.Model(inputs, out, name=f'decoder_top_{top_grid}_to_{img_size}')


def build_encoder_to_bottom(img_size, bottom_grid, num_hiddens):
    """
    Build encoder that downsamples from img_size to bottom_grid (for two-level VQ-VAE).
    
    Args:
        img_size: Input image size (e.g., 1024)
        bottom_grid: Output latent grid size (e.g., 64)
        num_hiddens: Number of hidden channels
        
    Returns:
        Keras Model for encoding to bottom level
    """
    assert (img_size % bottom_grid) == 0
    n_downs = int(math.log2(img_size // bottom_grid))  # e.g., 1024->64 = 4
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))
    x = inputs
    
    chans = [64, 128, 128, num_hiddens][:n_downs]
    for c in chans:
        x = tf.keras.layers.Conv2D(c, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)
    
    x = residual_stack(x, num_hiddens, num_blocks=2)
    return tf.keras.Model(inputs, x, name=f'encoder_to_bottom_{bottom_grid}')


def build_decoder_bottom_to_image(img_size, bottom_grid, in_ch):
    """
    Build decoder that upsamples from bottom_grid to img_size.
    
    Args:
        img_size: Output image size (e.g., 1024)
        bottom_grid: Input latent grid size (e.g., 64)
        in_ch: Number of input channels
        
    Returns:
        Keras Model for decoding from bottom level
    """
    assert (img_size % bottom_grid) == 0
    n_ups = int(math.log2(img_size // bottom_grid))
    
    inputs = tf.keras.Input(shape=(bottom_grid, bottom_grid, in_ch))
    x = inputs
    
    ups_ch = [256, 128, 64, 32][:n_ups]
    for c in ups_ch:
        x = tf.keras.layers.Conv2DTranspose(c, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)
    
    out = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
    return tf.keras.Model(inputs, out, name=f'decoder_bottom_{bottom_grid}_to_{img_size}')


class VQVAETopOnly(tf.keras.Model):
    """
    Top-only VQ-VAE model for extreme compression (~1024x).
    Uses only the top-level quantizer with large downsampling.
    """
    
    def __init__(self, image_size=1024, top_grid=32,
                 num_hiddens_top=128, embedding_dim_top=96, num_embeddings_top=256,
                 commitment_cost_top=1.0):
        super().__init__()
        
        self.image_size = image_size
        self.top_grid = top_grid
        
        # Build encoder-decoder
        self.enc_top = build_encoder_to_top(image_size, top_grid, num_hiddens_top)
        self.pre_vq_top = tf.keras.layers.Conv2D(embedding_dim_top, 1, name='pre_vq_top')
        self.vq_top = VectorQuantizerEMA(
            num_embeddings_top, embedding_dim_top,
            commitment_cost=commitment_cost_top, decay=0.99, epsilon=1e-5, name='vq_top'
        )
        self.post_vq_top = tf.keras.layers.Conv2D(num_hiddens_top, 1, name='post_vq_top')
        self.dec_top = build_decoder_top_to_image(image_size, top_grid, in_ch=num_hiddens_top)
        
        # Regularization
        self.norm_t = tf.keras.layers.LayerNormalization(axis=-1)
        self.drop_t = tf.keras.layers.SpatialDropout2D(0.1)

    def call(self, x, training=None):
        # Encode
        ht = self.enc_top(x)  # [B, 32, 32, num_hiddens_top]
        zt = self.pre_vq_top(ht)  # [B, 32, 32, embedding_dim_top]
        
        # Quantize
        ztq = self.vq_top(zt, training=training)
        
        # Decode
        ztq = self.post_vq_top(ztq)  # [B, 32, 32, num_hiddens_top]
        ztq = self.norm_t(ztq)
        ztq = self.drop_t(ztq, training=training)
        y = self.dec_top(ztq)  # [B, 1024, 1024, 1]
        
        return y


class VQVAETwoLevel(tf.keras.Model):
    """
    Two-level VQ-VAE model for moderate compression (~204x).
    Uses both top (coarse) and bottom (fine) quantizers.
    """
    
    def __init__(self, image_size=1024, top_grid=32, bottom_grid=64,
                 num_hiddens_top=128, embedding_dim_top=96, num_embeddings_top=256,
                 num_hiddens_bottom=128, embedding_dim_bottom=64, num_embeddings_bottom=256,
                 commitment_cost_top=0.25, commitment_cost_bottom=0.15):
        super().__init__()
        
        self.image_size = image_size
        self.top_grid = top_grid
        self.bottom_grid = bottom_grid
        
        # Bottom encoder-decoder
        self.enc_bottom = build_encoder_to_bottom(image_size, bottom_grid, num_hiddens_bottom)
        self.pre_vq_bottom = tf.keras.layers.Conv2D(embedding_dim_bottom, 1, name='pre_vq_bottom')
        self.vq_bottom = VectorQuantizerEMA(
            num_embeddings_bottom, embedding_dim_bottom,
            commitment_cost=commitment_cost_bottom, decay=0.99, epsilon=1e-5, name='vq_bottom'
        )
        
        # Top encoder (from bottom features)
        self.enc_top_from_bottom = build_encoder_to_top(bottom_grid, top_grid, num_hiddens_top)
        self.pre_vq_top = tf.keras.layers.Conv2D(embedding_dim_top, 1, name='pre_vq_top')
        self.vq_top = VectorQuantizerEMA(
            num_embeddings_top, embedding_dim_top,
            commitment_cost=commitment_cost_top, decay=0.99, epsilon=1e-5, name='vq_top'
        )
        self.post_vq_top = tf.keras.layers.Conv2D(num_hiddens_top, 1, name='post_vq_top')
        
        # Top decoder (to bottom resolution)
        self.dec_top_to_bottom = build_decoder_top_to_image(bottom_grid, top_grid, num_hiddens_top)
        
        # Merge and decode
        self.post_vq_bottom = tf.keras.layers.Conv2D(num_hiddens_bottom, 1, name='post_vq_bottom')
        self.dec_bottom = build_decoder_bottom_to_image(image_size, bottom_grid, num_hiddens_bottom)
        
        # Regularization
        self.norm_b = tf.keras.layers.LayerNormalization(axis=-1)
        self.drop_b = tf.keras.layers.SpatialDropout2D(0.1)

    def call(self, x, training=None):
        # Bottom path
        hb = self.enc_bottom(x)  # [B, 64, 64, num_hiddens_bottom]
        zb = self.pre_vq_bottom(hb)  # [B, 64, 64, embedding_dim_bottom]
        zbq = self.vq_bottom(zb, training=training)
        
        # Top path (from bottom features)
        ht = self.enc_top_from_bottom(hb)  # [B, 32, 32, num_hiddens_top]
        zt = self.pre_vq_top(ht)  # [B, 32, 32, embedding_dim_top]
        ztq = self.vq_top(zt, training=training)
        ztq = self.post_vq_top(ztq)  # [B, 32, 32, num_hiddens_top]
        
        # Decode top to bottom resolution
        ht_decoded = self.dec_top_to_bottom(ztq)  # [B, 64, 64, 1]
        
        # Merge bottom quantized + top decoded
        zbq = self.post_vq_bottom(zbq)  # [B, 64, 64, num_hiddens_bottom]
        zbq_merged = zbq + ht_decoded  # Residual connection
        zbq_merged = self.norm_b(zbq_merged)
        zbq_merged = self.drop_b(zbq_merged, training=training)
        
        # Final decode
        y = self.dec_bottom(zbq_merged)  # [B, 1024, 1024, 1]
        
        return y

