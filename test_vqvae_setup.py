#!/usr/bin/env python3
"""
Test script to verify VQ-VAE setup is working correctly.
Run this after installation to check if everything is configured properly.
"""

import sys
import os

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_imports():
    """Test if all required modules can be imported."""
    print_section("Testing Imports")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        from vqvae_modules import VQVAETopOnly, VQVAETwoLevel
        print("✓ VQ-VAE modules")
    except ImportError as e:
        print(f"✗ VQ-VAE modules: {e}")
        return False
    
    try:
        from vqvae_modules.config import get_config
        print("✓ Configuration module")
    except ImportError as e:
        print(f"✗ Configuration: {e}")
        return False
    
    return True


def test_gpu():
    """Test GPU availability."""
    print_section("Testing GPU Support")
    
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f"✓ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # Test GPU memory
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                c = tf.matmul(a, b)
            print("✓ GPU computation successful")
        except Exception as e:
            print(f"✗ GPU computation failed: {e}")
    else:
        print("⚠ No GPU detected - will use CPU")
        print("  (Training will be slower but still functional)")
    
    return True


def test_model_creation():
    """Test if models can be created."""
    print_section("Testing Model Creation")
    
    import tensorflow as tf
    from vqvae_modules import VQVAETopOnly, VQVAETwoLevel
    from vqvae_modules.config import get_config
    
    try:
        # Test Mode 1
        config1 = get_config('top_only')
        model1 = VQVAETopOnly(
            image_size=config1['image_size'],
            top_grid=config1['top_grid'],
            num_hiddens_top=config1['num_hiddens_top'],
            embedding_dim_top=config1['embedding_dim_top'],
            num_embeddings_top=config1['num_embeddings_top'],
            commitment_cost_top=config1['commitment_cost_top']
        )
        print("✓ Mode 1 (Top-Only) model created")
    except Exception as e:
        print(f"✗ Mode 1 creation failed: {e}")
        return False
    
    try:
        # Test Mode 2
        config2 = get_config('two_level')
        model2 = VQVAETwoLevel(
            image_size=config2['image_size'],
            top_grid=config2['top_grid'],
            bottom_grid=config2['bottom_grid'],
            num_hiddens_top=config2['num_hiddens_top'],
            embedding_dim_top=config2['embedding_dim_top'],
            num_embeddings_top=config2['num_embeddings_top'],
            num_hiddens_bottom=config2['num_hiddens_bottom'],
            embedding_dim_bottom=config2['embedding_dim_bottom'],
            num_embeddings_bottom=config2['num_embeddings_bottom'],
            commitment_cost_top=config2['commitment_cost_top'],
            commitment_cost_bottom=config2['commitment_cost_bottom']
        )
        print("✓ Mode 2 (Two-Level) model created")
    except Exception as e:
        print(f"✗ Mode 2 creation failed: {e}")
        return False
    
    return True


def test_forward_pass():
    """Test if models can perform forward pass."""
    print_section("Testing Forward Pass")
    
    import tensorflow as tf
    from vqvae_modules import VQVAETopOnly
    from vqvae_modules.config import get_config
    
    try:
        config = get_config('top_only')
        model = VQVAETopOnly(
            image_size=config['image_size'],
            top_grid=config['top_grid'],
            num_hiddens_top=config['num_hiddens_top'],
            embedding_dim_top=config['embedding_dim_top'],
            num_embeddings_top=config['num_embeddings_top'],
            commitment_cost_top=config['commitment_cost_top']
        )
        
        # Create dummy input
        dummy_input = tf.zeros([1, config['image_size'], config['image_size'], 1])
        print(f"  Input shape: {dummy_input.shape}")
        
        # Forward pass
        output = model(dummy_input, training=False)
        print(f"  Output shape: {output.shape}")
        
        # Check output
        if output.shape == dummy_input.shape:
            print("✓ Forward pass successful (shapes match)")
        else:
            print(f"✗ Shape mismatch: {output.shape} != {dummy_input.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_data_utils():
    """Test data utilities."""
    print_section("Testing Data Utilities")
    
    try:
        from vqvae_modules.data_utils import natural_sort_key, pad_to_multiple
        import tensorflow as tf
        
        # Test natural sorting
        files = ['image_10.png', 'image_2.png', 'image_1.png']
        sorted_files = sorted(files, key=natural_sort_key)
        expected = ['image_1.png', 'image_2.png', 'image_10.png']
        if sorted_files == expected:
            print("✓ Natural sorting works")
        else:
            print(f"✗ Natural sorting failed: {sorted_files} != {expected}")
            return False
        
        # Test padding
        img = tf.zeros([100, 200, 1])
        padded = pad_to_multiple(img, multiple=128)
        if padded.shape[0] % 128 == 0 and padded.shape[1] % 128 == 0:
            print(f"✓ Padding works: {img.shape} -> {padded.shape}")
        else:
            print(f"✗ Padding failed: {padded.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Data utilities test failed: {e}")
        return False
    
    return True


def print_summary(results):
    """Print test summary."""
    print_section("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your VQ-VAE setup is ready.")
        print("\nNext steps:")
        print("  1. Read QUICK_START.md for usage examples")
        print("  2. Prepare your image dataset")
        print("  3. Run: python train_mode1_toponly.py --help")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Install missing dependencies: pip install -r vqvae_requirements.txt")
        print("  2. Check Python version: python3 --version (need 3.8+)")
        print("  3. Verify you're in the correct directory")


def main():
    """Run all tests."""
    print("="*60)
    print("  VQ-VAE Setup Test")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    results = {}
    
    # Run tests
    results['Import Test'] = test_imports()
    if not results['Import Test']:
        print("\n⚠️  Import test failed. Cannot continue.")
        print("Please install dependencies: pip install -r vqvae_requirements.txt")
        return 1
    
    results['GPU Test'] = test_gpu()
    results['Model Creation Test'] = test_model_creation()
    results['Forward Pass Test'] = test_forward_pass()
    results['Data Utilities Test'] = test_data_utils()
    
    # Print summary
    print_summary(results)
    
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

