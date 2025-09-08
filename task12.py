import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def transform_points(points, tx, ty, tz, rx, ry, rz, use_htm=True):
    rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)
    
    if use_htm: #i made it a general function with this switch to choose between HTM and quaternion
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(rx), -np.sin(rx), 0],
                       [0, np.sin(rx), np.cos(rx), 0],
                       [0, 0, 0, 1]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                       [0, 1, 0, 0],
                       [-np.sin(ry), 0, np.cos(ry), 0],
                       [0, 0, 0, 1]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                       [np.sin(rz), np.cos(rz), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        # Translation matrix
        T = np.array([[1, 0, 0, tx],
                      [0, 1, 0, ty],
                      [0, 0, 1, tz],
                      [0, 0, 0, 1]])
        
        H = T @ Rz @ Ry @ Rx
        ones = np.ones((points.shape[0], 1))
        pts_h = np.hstack((points, ones))
        transformed = (H @ pts_h.T).T[:, :3]
    
    else: # Quaternion
        rotation = Rotation.from_euler('xyz', [rx, ry, rz])
        rotated_points = rotation.apply(points)
        transformed = rotated_points + np.array([tx, ty, tz])
    
    return transformed

points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

tx, ty, tz = 2, 1, 3 #parameters
rx, ry, rz = 30, 45, 60 #parameters

transformed_htm = transform_points(points, tx, ty, tz, rx, ry, rz, use_htm=True)
transformed_quat = transform_points(points, tx, ty, tz, rx, ry, rz, use_htm=False)


fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='Original')
ax1.scatter(transformed_htm[:, 0], transformed_htm[:, 1], transformed_htm[:, 2], c='r', label='HTM')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('HTM')
ax1.legend()


ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='Original')
ax2.scatter(transformed_quat[:, 0], transformed_quat[:, 1], transformed_quat[:, 2], c='g', label='Quaternion')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Quaternion')
ax2.legend()

plt.tight_layout()
plt.show()