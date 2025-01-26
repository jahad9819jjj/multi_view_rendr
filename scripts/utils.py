import configparser
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import cv2
import pathlib

def load_config(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load and parse configuration file with type conversion.

    This function handles INI-style configuration files with automatic type
    conversion for numeric and boolean values. Supports nested section structure.

    Args:
        file_path (str): Path to the configuration file. Supported formats:
            - .ini
            - .cfg
            - Other INI-style formats

    Returns:
        Dict[str, Dict[str, Any]]: Nested dictionary containing configuration
            parameters with appropriate data types. Structure:
            {
                'section1': {
                    'param1': value1 (auto-converted type),
                    ...
                },
                ...
            }

    Raises:
        FileNotFoundError: If specified config file does not exist
        ValueError: If config file contains invalid syntax

    Example:
        >>> config = load_config('config.ini')
        >>> print(config['Camera']['focal_length'])
        35.5

    Note:
        Uses eval() for type conversion - only use with trusted config files
        Consider using configparser's built-in type conversion for production
    """
    config = configparser.ConfigParser()
    if not pathlib.Path(file_path).exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    try:
        config.read(file_path)
    except configparser.Error as e:
        raise ValueError(f"Invalid config file: {str(e)}") from e

    params = {}
    for section in config.sections():
        params[section] = {}
        for key, value in config.items(section):
            try:
                # Safe evaluation with restricted namespace
                params[section][key] = eval(value, {"__builtins__": None}, {})
            except:
                # Fallback to string if evaluation fails
                params[section][key] = str(value)
    
    return params


def depth_to_viridis(depth_image: np.ndarray) -> np.ndarray:
    """Convert depth map to Viridis color-mapped visualization.

    Applies min-max normalization followed by Viridis colormap conversion
    suitable for scientific visualization.

    Args:
        depth_image (np.ndarray): Input depth map with shape (H, W).
            Expected dtype: float32 or float64
            Valid value range: Any finite float values

    Returns:
        np.ndarray: Color-mapped image in BGR format with shape (H, W, 3)
            dtype: uint8 (0-255 range)

    Raises:
        ValueError: If input is not 2D array or contains invalid values

    Example:
        >>> depth = np.random.rand(480, 640).astype(np.float32)
        >>> colorized = depth_to_viridis(depth)
        >>> cv2.imwrite('depth_vis.png', colorized)

    Note:
        Uses matplotlib's Viridis colormap - ensure matplotlib is installed
        Output format is OpenCV-compatible BGR for direct visualization
    """
    if depth_image.ndim != 2:
        raise ValueError(f"Invalid input shape: {depth_image.shape}. Expected 2D array")
    
    if not np.isfinite(depth_image).all():
        raise ValueError("Depth image contains NaN or infinite values")

    # Robust normalization handling edge cases
    d_min, d_max = np.nanmin(depth_image), np.nanmax(depth_image)
    if d_max - d_min < 1e-6:  # Handle uniform depth case
        depth_norm = np.zeros_like(depth_image)
    else:
        depth_norm = (depth_image - d_min) / (d_max - d_min)

    # Apply colormap and convert to OpenCV BGR format
    color_map = plt.cm.get_cmap('viridis')
    viridis_rgb = (color_map(depth_norm)[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(viridis_rgb, cv2.COLOR_RGB2BGR)