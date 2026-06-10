#!/usr/bin/env python3
"""
Mode 1: Top-Only VQ-VAE Training (1024× compression)

Usage:
    python train_mode1_toponly.py --data_dir /path/to/images --output_dir ./runs/mode1

This mode provides extreme compression (~1024×) that preserves global structure
without requiring a trained prior. Reconstructions stay stable at extreme bitrate;
fine detail is dropped and tile-boundary discontinuities can appear when stitching patches.
"""

import os
import argparse
import tensorflow as tf

from vqvae_modules.config import get_config, update_config
from vqvae_modules.models import VQVAETopOnly
from vqvae_modules.data_utils import collect_image_files, create_tiled_dataset
from vqvae_modules.eval_utils import export_model_bundle


def ssim_metric(y_true, y_pred):
    """SSIM metric for monitoring."""
    return tf.reduce_mean(tf.image.ssim(y_true + 0.5, y_pred + 0.5, max_val=1.0))


def psnr_metric(y_true, y_pred):
    """PSNR metric for monitoring."""
    return tf.reduce_mean(tf.image.psnr(y_true + 0.5, y_pred + 0.5, max_val=1.0))


def create_callbacks(output_dir):
    """Create training callbacks."""
    os.makedirs(output_dir, exist_ok=True)
    
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_ssim.weights.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_ssim_metric',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_loss.weights.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training.csv')),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'tensorboard')),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def main(args):
    # Load and update config
    config = get_config('top_only')
    config = update_config(
        config,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    
    print("\n" + "="*60)
    print("Mode 1: Top-Only VQ-VAE Training (1024× compression)")
    print("="*60)
    print(f"Image size: {config['image_size']}x{config['image_size']}")
    print(f"Top grid: {config['top_grid']}x{config['top_grid']}")
    print(f"Codebook size: {config['num_embeddings_top']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print("="*60 + "\n")
    
    # Collect image files
    print("Collecting image files...")
    all_files = collect_image_files(args.data_dir)
    print(f"Found {len(all_files)} images")
    
    # Split train/val
    split_idx = int(len(all_files) * config['train_split'])
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds = create_tiled_dataset(
        train_files,
        tile_size=config['tile_size'],
        batch_size=config['batch_size'],
        shuffle=True,
        shuffle_buffer=config['shuffle_buffer']
    )
    
    val_ds = create_tiled_dataset(
        val_files,
        tile_size=config['tile_size'],
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Build model
    print("\nBuilding model...")
    model = VQVAETopOnly(
        image_size=config['image_size'],
        top_grid=config['top_grid'],
        num_hiddens_top=config['num_hiddens_top'],
        embedding_dim_top=config['embedding_dim_top'],
        num_embeddings_top=config['num_embeddings_top'],
        commitment_cost_top=config['commitment_cost_top']
    )
    
    # Compile
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            beta_1=0.9,
            beta_2=0.95
        )
    except Exception:
        # Fallback for older TF versions
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.95
        )
    
    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=[psnr_metric, ssim_metric]
    )
    
    print("Model compiled successfully")
    
    # Create callbacks
    callbacks = create_callbacks(args.output_dir)
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights
    best_weights = os.path.join(args.output_dir, 'best_loss.weights.h5')
    if os.path.exists(best_weights):
        print(f"\nLoading best weights: {best_weights}")
        model.load_weights(best_weights)
    
    # Export bundle
    print("\nExporting model bundle...")
    export_model_bundle(
        model,
        config,
        args.output_dir,
        history=history,
        val_dataset=val_ds,
        num_samples=5
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Mode 1: Top-Only VQ-VAE (1024× compression)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./runs/mode1_toponly',
        help='Output directory for model and results'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-4,
        help='Initial learning rate'
    )
    
    args = parser.parse_args()
    main(args)

