"""
CNN Engine - Core Convolutional Neural Network Logic
Provides manual convolution operations, image processing, and visualization
"""

import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import json


def manual_convolution_2d(input_image, filter_kernel, stride=1, padding=0):
    """
    Perform 2D convolution on an image with a given filter.
    
    Args:
        input_image: 2D numpy array (height, width)
        filter_kernel: 2D numpy array (kernel_size, kernel_size)
        stride: Step size for filter movement
        padding: Number of zeros to pad around image
    
    Returns:
        feature_map: 2D output array
    """
    H, W = input_image.shape
    K = filter_kernel.shape[0]
    
    # Add padding if specified
    if padding > 0:
        input_image = np.pad(input_image, padding, mode='constant', constant_values=0)
        H, W = input_image.shape
    
    # Calculate output dimensions
    out_H = (H - K) // stride + 1
    out_W = (W - K) // stride + 1
    
    # Initialize output feature map
    feature_map = np.zeros((out_H, out_W))
    
    # Slide filter across image
    for i in range(out_H):
        for j in range(out_W):
            region = input_image[i*stride:i*stride+K, j*stride:j*stride+K]
            feature_map[i, j] = np.sum(region * filter_kernel)
    
    return feature_map


def max_pooling_2d(feature_map, pool_size=2, stride=None):
    """
    Apply max pooling to reduce feature map dimensions.
    
    Args:
        feature_map: 2D numpy array
        pool_size: Size of pooling window (e.g., 2x2)
        stride: Step size (default = pool_size for non-overlapping pools)
    
    Returns:
        pooled_map: Smaller feature map with max values
    """
    if stride is None:
        stride = pool_size
    
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    pooled_map = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            region = feature_map[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            pooled_map[i, j] = np.max(region)
    
    return pooled_map


def avg_pooling_2d(feature_map, pool_size=2, stride=None):
    """Apply average pooling."""
    if stride is None:
        stride = pool_size
    
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    pooled_map = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            region = feature_map[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            pooled_map[i, j] = np.mean(region)
    
    return pooled_map


def relu(x):
    """ReLU activation function: max(0, x)"""
    return np.maximum(0, x)


def visualize_convolution_operation(test_image, filter_kernel, result):
    """
    Create visualization of convolution operation.
    Returns base64 encoded image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#f8f9fa')
    
    im1 = axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(filter_kernel, cmap='coolwarm')
    axes[1].set_title('Convolution Filter', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(result, cmap='gray')
    axes[2].set_title('Feature Map Output', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def visualize_pooling_comparison(feature_map, max_pool, avg_pool):
    """
    Visualize pooling operations comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#f8f9fa')
    
    im1 = axes[0].imshow(feature_map, cmap='gray')
    axes[0].set_title(f'Original Feature Map\n({feature_map.shape[0]}x{feature_map.shape[1]})', 
                      fontsize=11, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(max_pool, cmap='gray')
    axes[1].set_title(f'Max Pooling (2x2)\n({max_pool.shape[0]}x{max_pool.shape[1]})', 
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(avg_pool, cmap='gray')
    axes[2].set_title(f'Avg Pooling (2x2)\n({avg_pool.shape[0]}x{avg_pool.shape[1]})', 
                      fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def visualize_multiple_filters(image, filters_dict):
    """
    Visualize multiple filter applications on one image.
    Returns base64 encoded image.
    """
    num_filters = len(filters_dict)
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#f8f9fa')
    gs = GridSpec(2, num_filters + 1, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original image
    ax = fig.add_subplot(gs[0, :2])
    im = ax.imshow(image, cmap='gray')
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Each filter
    for idx, (name, (filt, fmap)) in enumerate(filters_dict.items()):
        # Filter visualization
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(filt, cmap='coolwarm')
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)
        plt.colorbar(im, ax=ax)
        
        # Feature map
        ax = fig.add_subplot(gs[0, idx + 2])
        im = ax.imshow(fmap, cmap='gray')
        ax.set_title(f'Feature Map', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def visualize_relu_effect(feature_map_before, feature_map_after):
    """Visualize ReLU activation effect."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#f8f9fa')
    
    im1 = axes[0].imshow(feature_map_before, cmap='RdBu_r')
    axes[0].set_title('Before ReLU (Has negative values)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(feature_map_after, cmap='gray')
    axes[1].set_title('After ReLU (All ≥ 0)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def generate_sample_image(image_type='gradient'):
    """Generate a sample test image."""
    if image_type == 'gradient':
        img = np.linspace(0, 1, 25).reshape(5, 5)
    elif image_type == 'edges':
        img = np.array([
            [1, 2, 3, 0, 1],
            [0, 2, 1, 3, 2],
            [1, 0, 2, 1, 0],
            [2, 1, 0, 2, 3],
            [1, 1, 2, 2, 1]
        ], dtype=float) / 3.0
    elif image_type == 'circle':
        y, x = np.ogrid[-1:1:5j, -1:1:5j]
        img = np.exp(-(x**2 + y**2) / 0.3)
    else:
        img = np.random.rand(5, 5)
    
    return img


def generate_filter_kernels():
    """Generate common filter kernels."""
    kernels = {
        'Horizontal Edges': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
        'Vertical Edges': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
        'Diagonal (\\)': np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=float),
        'Diagonal (/)': np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=float),
        'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float) / 9.0,
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    }
    return kernels


def perform_convolution_demo(image_type, filter_type, stride, padding):
    """Perform convolution demo and return results."""
    try:
        image = generate_sample_image(image_type)
        kernels = generate_filter_kernels()
        
        if filter_type not in kernels:
            return {'error': 'Invalid filter type'}
        
        kernel = kernels[filter_type]
        result = manual_convolution_2d(image, kernel, stride=stride, padding=padding)
        
        # Generate visualization
        viz = visualize_convolution_operation(image, kernel, result)
        
        return {
            'success': True,
            'input_shape': image.shape,
            'output_shape': result.shape,
            'filter': filter_type,
            'image': image_type,
            'visualization': viz,
            'statistics': {
                'input_min': float(image.min()),
                'input_max': float(image.max()),
                'output_min': float(result.min()),
                'output_max': float(result.max()),
                'output_mean': float(result.mean())
            }
        }
    except Exception as e:
        return {'error': str(e)}


def perform_pooling_demo(image_type, pool_type, pool_size):
    """Perform pooling demo and return results."""
    try:
        image = generate_sample_image(image_type)
        kernels = generate_filter_kernels()
        
        # Create feature map by applying a filter
        kernel = kernels['Vertical Edges']
        feature_map = manual_convolution_2d(image, kernel, stride=1, padding=1)
        
        # Apply pooling
        if pool_type == 'max':
            pooled = max_pooling_2d(feature_map, pool_size=pool_size, stride=pool_size)
        else:
            pooled = avg_pooling_2d(feature_map, pool_size=pool_size, stride=pool_size)
        
        # For comparison, create both
        max_pool = max_pooling_2d(feature_map, pool_size=pool_size, stride=pool_size)
        avg_pool = avg_pooling_2d(feature_map, pool_size=pool_size, stride=pool_size)
        
        # Generate visualization
        viz = visualize_pooling_comparison(feature_map, max_pool, avg_pool)
        
        return {
            'success': True,
            'input_shape': feature_map.shape,
            'output_shape': pooled.shape,
            'pool_type': pool_type,
            'pool_size': pool_size,
            'reduction': f"{feature_map.shape[0]*feature_map.shape[1]} → {pooled.shape[0]*pooled.shape[1]} pixels",
            'visualization': viz,
            'statistics': {
                'input_mean': float(feature_map.mean()),
                'max_pool_mean': float(max_pool.mean()),
                'avg_pool_mean': float(avg_pool.mean()),
                'input_std': float(feature_map.std()),
                'max_pool_std': float(max_pool.std()),
                'avg_pool_std': float(avg_pool.std()),
            }
        }
    except Exception as e:
        return {'error': str(e)}


def perform_activation_demo(activation_type):
    """Demonstrate activation functions."""
    try:
        x = np.linspace(-5, 5, 100)
        
        if activation_type == 'relu':
            y = relu(x)
            title = 'ReLU: f(x) = max(0, x)'
        elif activation_type == 'sigmoid':
            y = 1 / (1 + np.exp(-x))
            title = 'Sigmoid: f(x) = 1 / (1 + e^-x)'
        elif activation_type == 'tanh':
            y = np.tanh(x)
            title = 'Tanh: f(x) = tanh(x)'
        else:
            y = x
            title = 'Linear: f(x) = x'
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#f8f9fa')
        
        # Activation function
        axes[0].plot(x, y, linewidth=2.5, color='#3b82f6')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('Input (x)', fontsize=11)
        axes[0].set_ylabel('Output f(x)', fontsize=11)
        axes[0].set_title(f'{title}', fontsize=12, fontweight='bold')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Derivative
        if activation_type == 'relu':
            dy = np.where(x > 0, 1, 0)
            deriv_title = "Derivative: f'(x) = 1 if x > 0 else 0"
        elif activation_type == 'sigmoid':
            y_sig = 1 / (1 + np.exp(-x))
            dy = y_sig * (1 - y_sig)
            deriv_title = "Derivative: f'(x) = f(x) * (1 - f(x))"
        elif activation_type == 'tanh':
            dy = 1 - np.tanh(x)**2
            deriv_title = "Derivative: f'(x) = 1 - tanh²(x)"
        else:
            dy = np.ones_like(x)
            deriv_title = "Derivative: f'(x) = 1"
        
        axes[1].plot(x, dy, linewidth=2.5, color='#ef4444')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('Input (x)', fontsize=11)
        axes[1].set_ylabel("Derivative f'(x)", fontsize=11)
        axes[1].set_title(deriv_title, fontsize=12, fontweight='bold')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'success': True,
            'activation': activation_type,
            'visualization': f"data:image/png;base64,{image_base64}"
        }
    except Exception as e:
        return {'error': str(e)}


# ==================== Image Upload & Processing ====================

def process_image_upload(image_base64, resize_to=28):
    """
    Process uploaded image from base64.
    Returns image array and visualization.
    """
    try:
        # Remove data URI prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        image_pil = Image.open(BytesIO(image_data))
        
        # Convert to grayscale if needed
        if image_pil.mode != 'L':
            image_pil = image_pil.convert('L')
        
        # Resize to target resolution for consistency
        size = int(resize_to)
        if size <= 0:
            size = 28
        image_pil = image_pil.resize((size, size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image_pil) / 255.0
        
        return {
            'success': True,
            'image': image_array,
            'shape': image_array.shape,
            'dtype': str(image_array.dtype)
        }
    except Exception as e:
        return {'error': f'Image processing failed: {str(e)}'}


def visualize_image_tensor(image_array, title="Image Tensor"):
    """
    Visualize image as tensor with pixel values.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#f8f9fa')
        
        # Image
        im = axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title(f'{title}\nShape: {image_array.shape}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0])
        
        # Histogram
        axes[1].hist(image_array.flatten(), bins=50, color='#3b82f6', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Pixel Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Pixel Value Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'success': True,
            'visualization': f"data:image/png;base64,{image_base64}",
            'statistics': {
                'min': float(image_array.min()),
                'max': float(image_array.max()),
                'mean': float(image_array.mean()),
                'std': float(image_array.std())
            }
        }
    except Exception as e:
        return {'error': str(e)}


def visualize_rgb_channels(image_array):
    """
    Visualize RGB channels separately (for RGB images).
    """
    try:
        if len(image_array.shape) == 2:
            # Grayscale - replicate to 3 channels
            rgb_image = np.stack([image_array] * 3, axis=-1)
        else:
            rgb_image = image_array[:, :, :3] if image_array.shape[2] >= 3 else image_array
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.patch.set_facecolor('#f8f9fa')
        
        # Original
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('RGB Combined', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Red
        red = np.zeros_like(rgb_image)
        red[:, :, 0] = rgb_image[:, :, 0]
        axes[0, 1].imshow(red)
        axes[0, 1].set_title('Red Channel', fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Green
        green = np.zeros_like(rgb_image)
        green[:, :, 1] = rgb_image[:, :, 1]
        axes[1, 0].imshow(green)
        axes[1, 0].set_title('Green Channel', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Blue
        blue = np.zeros_like(rgb_image)
        blue[:, :, 2] = rgb_image[:, :, 2]
        axes[1, 1].imshow(blue)
        axes[1, 1].set_title('Blue Channel', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'success': True,
            'visualization': f"data:image/png;base64,{image_base64}"
        }
    except Exception as e:
        return {'error': str(e)}


def apply_convolution_on_upload(image_base64, filter_type, num_filters=1):
    """
    Apply convolution to uploaded image.
    """
    try:
        # Process image
        img_result = process_image_upload(image_base64)
        if 'error' in img_result:
            return img_result
        
        image = img_result['image']
        kernels = generate_filter_kernels()
        
        if filter_type.lower() not in kernels:
            return {'error': 'Invalid filter type'}
        
        # Apply filter
        kernel = kernels[filter_type.lower()]
        result = manual_convolution_2d(image, kernel, stride=1, padding=1)
        result = relu(result)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#f8f9fa')
        
        im1 = axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(kernel, cmap='coolwarm')
        axes[1].set_title(f'{filter_type} Filter', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.2)
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(result, cmap='gray')
        axes[2].set_title('Feature Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        buffer.seek(0)
        viz_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'success': True,
            'visualization': f"data:image/png;base64,{viz_base64}",
            'input_shape': image.shape,
            'output_shape': result.shape,
            'statistics': {
                'output_min': float(result.min()),
                'output_max': float(result.max()),
                'output_mean': float(result.mean()),
                'output_std': float(result.std())
            }
        }
    except Exception as e:
        return {'error': str(e)}


def apply_multiple_filters(image_base64, num_filters=9):
    """
    Apply multiple filters to create feature maps.
    """
    try:
        img_result = process_image_upload(image_base64)
        if 'error' in img_result:
            return img_result
        
        image = img_result['image']
        kernels = list(generate_filter_kernels().items())[:min(num_filters, len(generate_filter_kernels()))]
        
        feature_maps = []
        for name, kernel in kernels:
            fmap = manual_convolution_2d(image, kernel, stride=1, padding=1)
            fmap = relu(fmap)
            feature_maps.append((name, fmap))
        
        # Visualize gallery
        num_cols = min(3, len(feature_maps))
        num_rows = (len(feature_maps) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
        fig.patch.set_facecolor('#f8f9fa')
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for idx, (name, fmap) in enumerate(feature_maps):
            im = axes[idx].imshow(fmap, cmap='gray')
            axes[idx].set_title(f'{name}\nShape: {fmap.shape}', fontsize=10, fontweight='bold')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx])
        
        # Hide extra subplots
        for idx in range(len(feature_maps), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        buffer.seek(0)
        viz_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'success': True,
            'visualization': f"data:image/png;base64,{viz_base64}",
            'num_filters_applied': len(feature_maps),
            'filter_names': [name for name, _ in feature_maps]
        }
    except Exception as e:
        return {'error': str(e)}


def apply_relu_comparison(image_base64):
    """
    Compare before and after ReLU activation.
    """
    try:
        img_result = process_image_upload(image_base64)
        if 'error' in img_result:
            return img_result
        
        image = img_result['image']
        
        # Apply convolution (which can produce negative values)
        kernel = generate_filter_kernels()['Vertical Edges']
        before_relu = manual_convolution_2d(image, kernel, stride=1, padding=1)
        after_relu = relu(before_relu)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#f8f9fa')
        
        im1 = axes[0].imshow(before_relu, cmap='RdBu_r')
        axes[0].set_title('Before ReLU\n(Has negative values)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(after_relu, cmap='gray')
        axes[1].set_title('After ReLU\n(All ≥ 0)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        buffer.seek(0)
        viz_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'success': True,
            'visualization': f"data:image/png;base64,{viz_base64}",
            'before_stats': {
                'min': float(before_relu.min()),
                'max': float(before_relu.max()),
                'mean': float(before_relu.mean()),
                'negative_count': int((before_relu < 0).sum())
            },
            'after_stats': {
                'min': float(after_relu.min()),
                'max': float(after_relu.max()),
                'mean': float(after_relu.mean()),
                'negative_count': int((after_relu < 0).sum())
            }
        }
    except Exception as e:
        return {'error': str(e)}


def apply_pooling_on_upload(image_base64, pool_type='max', pool_size=2):
    """
    Apply pooling to uploaded image with visualization.
    """
    try:
        img_result = process_image_upload(image_base64)
        if 'error' in img_result:
            return img_result
        
        image = img_result['image']
        
        # Create feature map first
        kernel = generate_filter_kernels()['Vertical Edges']
        feature_map = manual_convolution_2d(image, kernel, stride=1, padding=1)
        feature_map = relu(feature_map)
        
        # Apply pooling
        if pool_type.lower() == 'max':
            pooled = max_pooling_2d(feature_map, pool_size=pool_size)
        else:
            pooled = avg_pooling_2d(feature_map, pool_size=pool_size)
        
        # Also create the other type for comparison
        max_pool = max_pooling_2d(feature_map, pool_size=pool_size)
        avg_pool = avg_pooling_2d(feature_map, pool_size=pool_size)
        
        viz = visualize_pooling_comparison(feature_map, max_pool, avg_pool)
        
        return {
            'success': True,
            'visualization': viz,
            'input_shape': feature_map.shape,
            'output_shape': pooled.shape,
            'reduction': f"{feature_map.shape[0]*feature_map.shape[1]} → {pooled.shape[0]*pooled.shape[1]} pixels"
        }
    except Exception as e:
        return {'error': str(e)}


def _to_data_url_gray(image_2d):
    """Convert a 2D float array to a grayscale data URL."""
    arr = np.asarray(image_2d, dtype=float)
    if arr.size == 0:
        arr = np.zeros((4, 4), dtype=float)

    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-9:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        scaled = ((arr - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(scaled, mode='L')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _to_data_url_heatmap(image_2d):
    """Convert a 2D float array to a simple blue-red heatmap data URL."""
    arr = np.asarray(image_2d, dtype=float)
    if arr.size == 0:
        arr = np.zeros((4, 4), dtype=float)

    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-9:
        t = np.zeros_like(arr, dtype=float)
    else:
        t = ((arr - vmin) / (vmax - vmin)).clip(0, 1)

    r = (255 * t).astype(np.uint8)
    g = (90 * (1.0 - t)).astype(np.uint8)
    b = (255 * (1.0 - t)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    img = Image.fromarray(rgb, mode='RGB')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _build_kernel(kernel_mode='edge detection', custom_kernel=None):
    """Create a 3x3 kernel from mode or user input."""
    mode = str(kernel_mode or 'edge detection').strip().lower()

    if mode == 'edge detection':
        return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)
    if mode == 'blur':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float) / 9.0
    if mode == 'sharpen':
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
    if mode == 'random':
        return np.random.uniform(-1.0, 1.0, (3, 3)).astype(float)
    if mode == 'custom':
        if custom_kernel is None:
            return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
        parsed = np.array(custom_kernel, dtype=float)
        if parsed.shape != (3, 3):
            raise ValueError('Custom kernel must be 3x3.')
        return parsed

    return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)


def _apply_preprocess(image_2d, preprocess_type='normalize'):
    """Apply a simple preprocessing transform used in CNN tutorials."""
    mode = str(preprocess_type or 'normalize').strip().lower()
    x = np.asarray(image_2d, dtype=float)

    if mode == 'normalize':
        # Mean/std style normalization used in many vision pipelines.
        return (x - 0.5) / 0.5
    if mode == 'minmax':
        mn = float(x.min())
        mx = float(x.max())
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    if mode == 'invert':
        return 1.0 - x
    if mode == 'threshold':
        return (x > 0.5).astype(float)
    return (x - 0.5) / 0.5


def perform_image_convolution_lab(
    image_base64,
    kernel_mode='edge detection',
    custom_kernel=None,
    stride=1,
    padding=1,
    num_filters=1,
    resize_to=224,
    preprocess_type='normalize',
):
    """Backend computation for an interactive convolution/feature-map teaching lab."""
    try:
        stride = max(1, int(stride))
        padding = max(0, int(padding))
        num_filters = max(1, min(8, int(num_filters)))
        resize_to = int(resize_to)
        if resize_to not in (28, 224):
            resize_to = 224

        image_result = process_image_upload(image_base64, resize_to=resize_to)
        if 'error' in image_result:
            return image_result

        raw_image = np.asarray(image_result['image'], dtype=float)
        preprocessed = _apply_preprocess(raw_image, preprocess_type=preprocess_type)

        base_kernel = _build_kernel(kernel_mode=kernel_mode, custom_kernel=custom_kernel)

        kernels = [base_kernel]
        for _ in range(num_filters - 1):
            kernels.append(np.random.uniform(-1.0, 1.0, (3, 3)).astype(float))

        feature_maps = []
        for k in kernels:
            fmap = manual_convolution_2d(preprocessed, k, stride=stride, padding=padding)
            feature_maps.append(fmap)

        first_map = feature_maps[0]

        # Step trace for animation with a safe upper bound for responsiveness.
        padded = np.pad(preprocessed, padding, mode='constant', constant_values=0) if padding > 0 else preprocessed
        ksize = 3
        out_h, out_w = first_map.shape
        total_steps = int(out_h * out_w)
        max_steps = 220
        step_skip = max(1, int(np.ceil(total_steps / max_steps)))

        trace = []
        index = 0
        for i in range(out_h):
            for j in range(out_w):
                if index % step_skip == 0:
                    sy = i * stride
                    sx = j * stride
                    patch = padded[sy:sy + ksize, sx:sx + ksize]
                    prod = patch * base_kernel
                    s = float(np.sum(prod))
                    terms = [
                        f"{float(patch[r, c]):.3f}*{float(base_kernel[r, c]):.3f}"
                        for r in range(ksize)
                        for c in range(ksize)
                    ]
                    trace.append({
                        'out_y': int(i),
                        'out_x': int(j),
                        'patch_y': int(sy - padding),
                        'patch_x': int(sx - padding),
                        'equation': ' + '.join(terms),
                        'sum': s,
                    })
                index += 1

        maps_payload = []
        for idx, fmap in enumerate(feature_maps):
            maps_payload.append({
                'name': f'Filter {idx + 1}',
                'shape': [int(fmap.shape[0]), int(fmap.shape[1])],
                'image': _to_data_url_gray(fmap),
                'min': float(fmap.min()),
                'max': float(fmap.max()),
                'mean': float(fmap.mean()),
            })

        kernel_name = str(kernel_mode).strip().lower()
        if kernel_name == 'edge detection':
            kernel_focus = 'This kernel responds strongly where intensity changes abruptly, so object boundaries become visible.'
            use_case = 'Useful in OCR, document scanning, and medical edge localization where boundaries matter.'
        elif kernel_name == 'blur':
            kernel_focus = 'This kernel averages neighbors, so noise is reduced and sharp transitions are softened.'
            use_case = 'Useful when denoising noisy camera frames before deeper feature extraction.'
        elif kernel_name == 'sharpen':
            kernel_focus = 'This kernel boosts local contrast and emphasizes fine strokes and details.'
            use_case = 'Useful for highlighting fine structures such as handwriting strokes or surface details.'
        elif kernel_name == 'random':
            kernel_focus = 'A random kernel gives unpredictable responses, similar to untrained filters before learning.'
            use_case = 'Useful for understanding why CNN training is needed to learn meaningful filters.'
        else:
            kernel_focus = 'Custom kernels let you test your own hypothesis about what local pattern should be amplified.'
            use_case = 'Useful for experimentation and intuition-building in filter design.'

        stride_padding = (
            f'Stride {stride} controls scan jump size and padding {padding} controls border context. '
            'Higher stride gives speed but less spatial detail; higher padding preserves edge regions.'
        )

        preprocess_note = {
            'normalize': 'values are centered/scaled so kernels react to relative intensity patterns, not raw brightness.',
            'minmax': 'values are stretched into [0,1] to keep contrast comparable across images.',
            'invert': 'bright and dark regions are flipped to test contrast sensitivity.',
            'threshold': 'pixels become binary (0/1), emphasizing hard boundaries over shades.'
        }.get(str(preprocess_type).strip().lower(), 'a preprocessing transform was applied before convolution.')

        return {
            'success': True,
            'input_shape': [int(raw_image.shape[0]), int(raw_image.shape[1]), 1],
            'output_shape': [int(first_map.shape[0]), int(first_map.shape[1])],
            'stride': stride,
            'padding': padding,
            'num_filters': num_filters,
            'kernel': base_kernel.tolist(),
            'raw_matrix': np.round(raw_image, 6).tolist(),
            'preprocessed_matrix': np.round(preprocessed, 6).tolist(),
            'feature_matrix': np.round(first_map, 6).tolist(),
            'raw_image': _to_data_url_gray(raw_image),
            'preprocessed_image': _to_data_url_gray(preprocessed),
            'feature_image': _to_data_url_gray(first_map),
            'heatmap_image': _to_data_url_heatmap(first_map),
            'feature_maps': maps_payload,
            'trace': trace,
            'explain': {
                'kernel_focus': kernel_focus,
                'stride_padding': stride_padding,
                'texture': 'Patterns create repeated high responses across nearby regions.',
                'preprocess_note': preprocess_note,
                'use_case': use_case,
            },
        }
    except Exception as e:
        return {'error': f'Convolution lab failed: {str(e)}'}
