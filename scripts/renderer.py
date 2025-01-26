import numpy as np
import open3d as o3d
import cv2
from typing import Union, Optional


def sampling_farthest_points(points, npoint):
    """ Sampling by farthest.

    Args:
        points (np.array): NxD
        npoint (int): M

    Returns:
        np.array : MxD
    """
    np.random.seed(0)
    N, D = points.shape
    xyz = points[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points = points[centroids.astype(np.int32)]
    return points


class CameraPoseGenerator:
    def __init__(self,
                 params:Optional[dict]=None
                 ) -> None:
        self.params = params
        
    def __generate_xyz_coordinates(self) -> np.array:
        """
        Generates a set of xyz coordinates based on the method and parameters

        Raises:
            NotImplementedError: _description_

        Returns:
            np.array: _description_
        """
        if self.params['method'] == 'sphere':
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.params['radius'])
            # randomly sampling(support steady randomness due to seed)
            xyz_coordinates = sampling_farthest_points(
                np.asarray(sphere.vertices),
                npoint=self.params['number_of_points']
            )
        elif self.params['method'] == 'polyhedron':
            pass
        else:
            raise NotImplementedError(f'Not implemented method: {self.method}')
        return xyz_coordinates
        
    def __generate_lookat_matrices(self, xyz_coords:np.array) -> np.array:
        """
        Generates a look-at transformation matrix for each point in points,
        where each point is assumed to be looking at the origin.

        Args:
        points (np.ndarray): A (N, 3) array of points.

        Returns:
        list: A list of 4x4 transformation matrices.
        """
        def normalize(v):
            """ Normalize a vector """
            return v / np.linalg.norm(v)

        matrices = []
        up = np.array([0, 1, 0])  # Up vector
        
        for point in xyz_coords:
            # Compute the look-at vector (z-axis)
            z_axis = -normalize(point)
            
            # Compute the right vector (x-axis)
            x_axis = normalize(np.cross(up, z_axis))
            
            # Compute the actual up vector (y-axis)
            y_axis = np.cross(z_axis, x_axis)

            # Create the transformation matrix
            matrix = np.eye(4)
            matrix[:3, 0] = x_axis
            matrix[:3, 1] = y_axis
            matrix[:3, 2] = z_axis
            matrix[:3, 3] = point

            matrices.append(matrix)
        return matrices
    
    def generate(self) -> np.array:
        """
        Generates a set of look-at transformation matrices based on the method and parameters

        Raises:
            NotImplementedError: _description_

        Returns:
            np.array: _description_
        """
        xyz_coords = self.__generate_xyz_coordinates()
        matrices = self.__generate_lookat_matrices(xyz_coords)
        return matrices
        

class Renderer():
    def __init__(self, 
                 geometry:Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh], 
                 intrinsic:Union[o3d.camera.PinholeCameraParameters, o3d.camera.PinholeCameraIntrinsic],
                 pixel_step:int=1
                 ):
        self.geometry = geometry
        if type(intrinsic[0]) == o3d.camera.PinholeCameraParameters:
            self.intrinsic = intrinsic[0].intrinsic
        else:
            self.intrinsic = intrinsic
        self.w:int = (self.intrinsic.width - 1) // pixel_step + 1
        self.h:int = (self.intrinsic.height - 1) // pixel_step + 1
        self.fx:float = self.intrinsic.intrinsic_matrix[0, 0]
        self.fy:float = self.intrinsic.intrinsic_matrix[1, 1]
        self.cx:float = self.intrinsic.intrinsic_matrix[0, 2]
        self.cy:float = self.intrinsic.intrinsic_matrix[1, 2]
        
        self.pixel_step = pixel_step
        self.image_width:int = int((self.cx if self.cx * 2 > self.w
                               else self.image_width - self.cx)
                                * 2 / self.pixel_step)
        self.image_height:int = int((self.cy if self.cy * 2 > self.h
                                else self.image_height - self.cy)
                                * 2 / self.pixel_step)
        
        self.vis:o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.image_width, height=self.image_height, visible=False)
        self.vis.add_geometry(self.geometry)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.get_render_option().mesh_show_back_face = True
        # self.vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        
        # TODO: Research ctrl's fov type.
        # const(default) : https://github.com/isl-org/Open3D/blob/4214a0d8f10ec46f1fd1787e76fa56027fe7f7ed/cpp/open3d/visualization/visualizer/ViewControl.cpp#L26C1-L29C52
        # calculation method: https://github.com/isl-org/Open3D/blob/4214a0d8f10ec46f1fd1787e76fa56027fe7f7ed/cpp/open3d/visualization/visualizer/ViewControl.cpp#L149-L153
        self.ctr:o3d.visualization.ViewControl = self.vis.get_view_control()
        
        self.param = self.ctr.convert_to_pinhole_camera_parameters()
        self.param.intrinsic = self.intrinsic
        
    # TODO: merge render(color and depth) due to unconxiuous about extrinsic
    
    def depth(self, pose):
        # open3dでレンダリングされるdepthはfloat32
        # self.param.extrinsic = np.linalg.inv(pose).astype(np.float32)
        # self.param.extrinsic = np.linalg.inv(pose).astype(np.float64)
        self.param.extrinsic = np.linalg.inv(pose)
        self.ctr.convert_from_pinhole_camera_parameters(self.param, True)
        self.vis.poll_events()
        self.vis.update_renderer()
        depth = np.asarray(self.vis.capture_depth_float_buffer(do_render=False))
        if self.cy * 2 > self.image_height:
            depth = depth[:self.h, :]
        else:
            depth = depth[depth.shape[0] - self.h:, :]
        if self.cx * 2 > self.image_width:
            depth = depth[:, :self.w]
        else:
            depth = depth[:, depth.shape[1] - self.w:]
        return depth
    
    def depth_cv(self, pose):
        # if depth is expressed by 3d point indicies
        raise NotImplementedError

    def color(self, pose):
        # self.param.extrinsic = np.linalg.inv(pose).astype(np.float32)
        # self.param.extrinsic = np.linalg.inv(pose).astype(np.float64)
        self.param.extrinsic = np.linalg.inv(pose)
        self.ctr.convert_from_pinhole_camera_parameters(self.param, True)
        self.vis.poll_events()
        self.vis.update_renderer()
        color = np.asarray(self.vis.capture_screen_float_buffer(do_render=False))
        if self.cy * 2 > self.image_height:
            color = color[:self.h, :]
        else:
            color = color[color.shape[0] - self.h:, :]
        if self.cx * 2 > self.image_width:
            color = color[:, :self.w]
        else:
            color = color[:, color.shape[1] - self.w:]
        return color

if __name__ == "__main__":
    pass