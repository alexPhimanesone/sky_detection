import sys
import os
import numpy as np
import yaml as yml
sys.path.append("../fisheye_to_equirectangular_v3")
from camera_coords_to_image_intrinsic import camera_coords_to_image_intrinsic

data_dir = "../data/"
config_dir = "./calib_config/"


def estimate_radius(calibration_path):
    """
    Save principal point and radius at pprad_path.
    """
    pprad_path = os.path.join(data_dir, "pprad.yml")

    # PARAMETERS
    fov_steps_factor = 10

    with open(os.path.join(config_dir, "fov.txt"), 'r') as f:
        fov_limits = f.readlines()
    fov_min = eval(fov_limits[0].strip())
    fov_max = eval(fov_limits[1].strip())

    with open(calibration_path) as f:
        data = yml.load(f, Loader=yml.SafeLoader)
    poly_incident_angle_to_radius = data['poly_incident_angle_to_radius']
    principal_point = data['principal_point']

    # METHODE DE CAO
    fov_steps = (fov_max - fov_min) * fov_steps_factor
    fov_test_theta = np.linspace(fov_min, fov_max, fov_steps) * np.pi/180
    x_prime, y_prime  = np.tan(fov_test_theta), np.zeros(len(fov_test_theta))
    fov_limit = camera_coords_to_image_intrinsic(np.array([x_prime,y_prime]).T.tolist(),
                                                 poly_incident_angle_to_radius,
                                                 principal_point)
    index_of_max = np.argmax(np.transpose(fov_limit - principal_point))
    estimated_fov = fov_min + index_of_max / fov_steps_factor
    distance_to_fov = fov_limit[index_of_max][0] - principal_point[0]
    radius = round(distance_to_fov) + 1

    print(f"Estimated FOV: {estimated_fov}")
    data = {'principal_point': principal_point, 'radius': radius}
    with open(pprad_path, 'w') as f:
        yml.dump(data, f)


def crop_around_disk(pprad_path, img):
    """
    Returns:
    Minimal square image with black corners.
    """

    # Load pp and rad
    with open(pprad_path, 'r') as f:
        data = yml.load(f, Loader=yml.SafeLoader)
    principal_point = data['principal_point']
    radius = data['radius']

    # Calculate the bounding box for the disk region
    cx, cy = map(round, principal_point)
    x_min = cx - radius
    y_min = cy - radius
    x_max = cx + radius
    y_max = cy + radius

    # Calculate the coordinates of the disk points within the cropped image
    y_coords, x_coords = np.meshgrid(np.arange(y_min, y_max+1), np.arange(x_min, x_max+1))
    distances = (x_coords - cx)**2 + (y_coords - cy)**2
    disk_mask = distances <= radius**2

    # Crop the image using the bounding box
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]

    # Apply the mask to the cropped image
    cropped_img = np.where(disk_mask[..., np.newaxis], cropped_img, 0)

    return cropped_img
