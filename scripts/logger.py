"""
3D Rendering Pipeline Configuration Logger Module

This module provides logging functionalities for capturing critical parameters 
in 3D computer vision workflows, including camera configurations, depth data statistics,
and algorithmic parameters.
"""

import logging
import json
from typing import List, Union, Dict
import open3d as o3d
import numpy as np

class Logger:
    """Abstract base class defining the logger interface"""
    def __init__(self) -> None:
        raise NotImplementedError("Logger is an abstract class")
    
    def log(self):
        """Abstract method for logging operations"""
        raise NotImplementedError("Must implement log method")
    
    
class SettingLogger(Logger):
    """
    Concrete logger implementation for 3D rendering configurations
    
    Attributes:
        path (str): Log file output path
        logger (logging.Logger): Configured logger instance
        kwargs (dict): Additional configuration parameters
    """
    
    def __init__(self, 
                 file_path: str, 
                 **kwargs: Union[Dict, List]) -> None:
        """
        Initialize logging system with dual file/console outputs

        Args:
            file_path (str): Path to save log file
            **kwargs: Additional configuration parameters for extensibility

        Raises:
            PermissionError: If log file path is not writable
        """
        self.path = file_path
        self.kwargs = kwargs
        
        # Configure hierarchical logging system
        self.logger = logging.getLogger('3DRenderingLogger')
        self.logger.setLevel(logging.DEBUG)  # Capture all debug information

        # File handler configuration for persistent storage
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)

        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # ISO 8601 timestamp format for auditability
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%SZ'
        )
        
        # Apply consistent formatting across handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Prevent duplicate handlers in Jupyter environments
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    
    def log(self,
            camera_params: List[o3d.camera.PinholeCameraParameters],
            depths: List[np.ndarray],
            virtual_view_point: Dict,
            surface_reconstruction: Dict = None,
            ) -> None:
        """
        Central logging method for capturing entire pipeline state

        Args:
            camera_params: List of Open3D camera parameter objects
            depths: List of depth maps as numpy arrays
            virtual_view_point: Virtual camera configuration dictionary
            surface_reconstruction: Surface reconstruction parameters (optional)

        Examples:
            >>> logger.log(cameras, depths, view_config, recon_params)
        """
        self.log_camera_params(camera_params)
        self.log_depth_value(depths)
        self.log_virtual_view_point(virtual_view_point)
        
        if surface_reconstruction is not None:
            self.log_surface_reconstruction(surface_reconstruction)
    
            
    def log_virtual_view_point(self, virtual_view_point: Dict) -> None:
        """Log virtual viewpoint generation parameters"""
        self.logger.info(
            "Virtual Viewpoint Configuration:\n%s", 
            json.dumps(virtual_view_point, indent=2, default=str)
        )
        
    def log_surface_reconstruction(self, surface_reconstruction: Dict) -> None:
        """Log surface reconstruction algorithm parameters"""
        self.logger.info(
            "Surface Reconstruction Parameters:\n%s", 
            json.dumps(surface_reconstruction, indent=2, default=str)
        )
    
    def log_camera_params(self, 
                         camera_params: List[o3d.camera.PinholeCameraParameters]
                         ) -> None:
        """Log camera intrinsic/extrinsic parameters in structured format"""
        
        def parse_camera_param_to_dict(
            camera_param: o3d.camera.PinholeCameraParameters
        ) -> Dict:
            """
            Convert Open3D camera parameters to serializable dictionary
            
            Args:
                camera_param: Open3D camera parameter object
                
            Returns:
                Dictionary containing:
                - Intrinsic: focal length (fx,fy), principal point (cx,cy)
                - Extrinsic: rotation matrix (r11-r33), translation vector (t1-t3)
                - Image dimensions derived from principal point position
            """
            intrinsic = camera_param.intrinsic.intrinsic_matrix
            extrinsic = camera_param.extrinsic
            
            return {
                # Intrinsic parameters
                "fx": intrinsic[0, 0],
                "fy": intrinsic[1, 1],
                "principal_point": (intrinsic[0, 2], intrinsic[1, 2]),
                "image_dimensions": (
                    int(intrinsic[0, 2] * 2),  # Width derived from principal point
                    int(intrinsic[1, 2] * 2)   # Height derived from principal point
                ),
                
                # Extrinsic parameters (R|t matrix)
                "rotation_matrix": [
                    [extrinsic[0, 0], extrinsic[0, 1], extrinsic[0, 2]],
                    [extrinsic[1, 0], extrinsic[1, 1], extrinsic[1, 2]],
                    [extrinsic[2, 0], extrinsic[2, 1], extrinsic[2, 2]]
                ],
                "translation_vector": [
                    extrinsic[0, 3], 
                    extrinsic[1, 3], 
                    extrinsic[2, 3]
                ]
            }
        
        for idx, camera in enumerate(camera_params):
            camera_data = parse_camera_param_to_dict(camera)
            self.logger.info(
                "Camera %d Parameters:\n%s",
                idx,
                json.dumps(camera_data, indent=2)
            )
        
    def log_depth_value(self, depths: List[np.ndarray]) -> None:
        """Log statistical summary of depth maps"""
        for idx, depth in enumerate(depths):
            self.logger.info(
                "Depth Map %d Statistics: min=%.3f, max=%.3f, mean=%.3f",
                idx,
                depth.min(),
                depth.max(),
                depth.mean()
            )