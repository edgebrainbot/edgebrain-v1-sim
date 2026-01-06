#!/usr/bin/env python3
"""
Vision Processor Module
Processes point cloud data to vision features
"""

import torch
import numpy as np
from sensor_msgs_py import point_cloud2


class VisionProcessor:
    """
    Processes LiDAR point cloud to vision features using polynomial features
    """
    
    def __init__(self, device, feature_dim=24):
        """
        Initialize Vision Processor
        
        Args:
            device: torch device
            feature_dim: Vision feature dimension (must be divisible by 3)
        """
        self.device = device
        self.feature_dim = feature_dim
        
        # Matrix for processing lidar
        self.mat_base_vec = torch.arange(0.1, 144.1, 0.1, dtype=torch.float32).to(device) * 50
        x = torch.log(self.mat_base_vec) * 0.0001
        
        # Polynomial feature matrix
        self.vis_feature_mat = torch.vander(x, N=feature_dim // 3, increasing=True).T.to(device)
    
    def parse_pointcloud(self, msg):
        """
        Parse PointCloud2 message to extract XYZ coordinates
        
        Args:
            msg: PointCloud2 message
            
        Returns:
            np.ndarray: Point cloud array (N, 3)
        """
        try:
            points_list = []
            
            # Read point cloud data (x, y, z fields)
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False):
                clean_point = np.nan_to_num([point[0], point[1], point[2]], 
                                           nan=0.0, posinf=0.0, neginf=0.0)
                points_list.append(clean_point.tolist())
            
            return np.array(points_list)
            
        except Exception as e:
            return None
    
    def process_pointcloud(self, points):
        """
        Process point cloud to vision features
        
        Args:
            points: Point cloud array (N, 3)
            
        Returns:
            torch.Tensor: Vision features (feature_dim,)
        """
        if points is None:
            return torch.zeros(self.feature_dim, device=self.device)
        
        points_tensor = torch.tensor(points, dtype=torch.float32).to(self.device)
        res = self.vis_feature_mat @ points_tensor
        res *= 0.01
        
        return res.flatten()
