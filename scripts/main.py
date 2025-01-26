import open3d as o3d
import cv2
import numpy as np
import os
import argparse
import pathlib
from typing import List
from utils import load_config, depth_to_viridis
from renderer import Renderer, CameraPoseGenerator
from logger import SettingLogger

def main(
    input_path: str,
    camera_config_path: str,
    output_dir: str,
    debug: bool = False
) -> None:
    """Main processing pipeline for multi-view 3D rendering and data generation.

    This pipeline handles the complete workflow from data loading to output generation:
    1. Configuration parsing and validation
    2. Point cloud data loading and verification
    3. Virtual camera setup and pose generation
    4. Multi-view RGB-D rendering
    5. Output data saving and logging

    Typical usage flow:
    - Load camera configuration file
    - Generate virtual camera trajectories
    - Render color/depth images from multiple viewpoints
    - Save results with associated camera parameters

    Args:
        input_path (str): Path to input 3D point cloud file (PLY/PCD format)
        camera_config_path (str): Path to YAML/JSON configuration file containing:
            - camera_pose: Viewpoint generation parameters
            - camera_intrinsic: Camera sensor specifications
        output_dir (str): Output directory path for saving:
            - Rendered images (color_*.png, depth_*.tiff)
            - Camera parameters (camera_params.npz)
            - Configuration log (config.log)
        debug (bool, optional): Enable debug mode with additional visualizations.
            When True, shows:
            - Camera coordinate frames overlaid on point cloud
            - Depth map visualizations during rendering
            Defaults to False.

    Raises:
        ValueError: If camera configuration is missing required parameters
        FileNotFoundError: If input point cloud file does not exist
        ValueError: If loaded point cloud contains no valid points

    Example:
        >>> main(
        ...     input_path="data/object.ply",
        ...     camera_config_path="config/camera_config.yaml",
        ...     output_dir="output",
        ...     debug=True
        ... )
    """
    # Configuration loading
    config = load_config(camera_config_path)
    if not config.get('camera_pose') or not config.get('camera_intrinsic'):
        raise ValueError("Invalid camera configuration")

    # Input validation
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Data preparation
    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        raise ValueError("Failed to load point cloud or empty point cloud")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Camera setup
    pose_gen = CameraPoseGenerator(config['camera_pose'])
    camera_poses = pose_gen.generate()
    
    # Camera parameters initialization
    intrinsic_params = config['camera_intrinsic']
    cameras = []
    for pose in camera_poses:
        cam = o3d.camera.PinholeCameraParameters()
        cam.extrinsic = pose
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic_params.get('width', 1920),
            height=intrinsic_params.get('height', 1080),
            fx=intrinsic_params.get('fx', 1372.45),
            fy=intrinsic_params.get('fy', 1350.36),
            cx=intrinsic_params.get('cx', 1049.30),
            cy=intrinsic_params.get('cy', 593.64)
        )
        cameras.append(cam)

    # Debug visualization
    if debug:
        axes = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(pose)
                for pose in camera_poses]
        o3d.visualization.draw_geometries([pcd, *axes])

    # Rendering pipeline
    renderer = Renderer(pcd, cameras)
    colors, depths = [], []

    # Capture views
    for idx, cam in enumerate(cameras):
        # Color rendering
        color = renderer.color(cam.extrinsic)
        color = (color * 255).astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR format
        colors.append(color)
        cv2.imwrite(os.path.join(output_dir, f"color_{idx:03d}.png"), color)

        # Depth rendering
        depth = renderer.depth(cam.extrinsic)
        depths.append(depth)
        cv2.imwrite(os.path.join(output_dir, f"depth_{idx:03d}.tiff"), depth)

        # Debug depth visualization
        if debug:
            vis_depth = depth_to_viridis(depth)
            cv2.imshow(f"Depth View {idx}", vis_depth)
            cv2.waitKey(100)

    # Save camera parameters
    np.savez(
        os.path.join(output_dir, "camera_params.npz"),
        intrinsics=[cam.intrinsic.intrinsic_matrix for cam in cameras],
        extrinsics=[cam.extrinsic for cam in cameras]
    )

    # Save configuration log
    logger = SettingLogger(os.path.join(output_dir, "config.log"))
    logger.log(
        camera_params=cameras,
        depths=depths,
        virtual_view_point=config
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-view Point Cloud Renderer - Generate RGB-D views from 3D point cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        help="Input point cloud/mesh file path",
        type=pathlib.Path,
        default="./data/bunny/reconstruction/bun_zipper.ply"
    )
    parser.add_argument(
        "-c", "--config",
        help="Camera configuration file path",
        type=pathlib.Path,
        default='./config/example.config'
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory path",
        type=pathlib.Path,
        default='./output'
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional visualizations",
        default=False
    )

    args = parser.parse_args()

    try:
        main(
            input_path=str(args.input.resolve()),
            camera_config_path=str(args.config.resolve()),
            output_dir=str(args.output.resolve()),
            debug=args.debug
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        exit(1)