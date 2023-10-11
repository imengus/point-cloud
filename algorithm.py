import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

import sklearn.linear_model as lm
from sklearn.cluster import KMeans
from skimage.transform import probabilistic_hough_line
from skimage import draw
import cv2

import warnings
warnings.filterwarnings("ignore")

import sys


def read_xyz(PATH):
    """Create df and matrix from data"""
    df = pd.read_csv(PATH, sep=' ')

    # Any extra columns detected are irrelevant
    df = df.drop(df.columns[3:], axis=1)
    mat = df.to_numpy()
    return mat, df


def find_ceiling(col_num, df, n_divs):
    """Finds the coordinates of the ceiling points"""
    points = np.zeros((1, 3))

    col = df.columns[col_num]
    n_iter = len(df[col]) // n_divs
    ordf = df.sort_values(by=[col])

    # Reversed to capture ceiling rather than wall
    for i in reversed(range(n_iter)):
        # Interval of points
        temp = ordf[i*n_divs:(i+1)*n_divs]

        # The index that gives the highest z value
        idx = temp[df.columns[2]].idxmax()

        # Extracts the [x y z] array from dataframe
        current = df.iloc[idx].to_numpy().copy()

        # If current point isn't increasing,
        # it isn't recorded
        if np.max(points[:,2]) < current[2]:
            # Append to list
            points = np.vstack((points,current))
    
    # Remove 0s at the start that are appended to
    points = points[1:,:]
    return points


def fit_plane2ceil(points):
    """Fit a plane to the ceiling"""
    X_train = points[:, :2]
    y_train = points[:, 2]

    model = lm.LinearRegression()
    model.fit(X_train, y_train)

    return model.coef_, model.intercept_


def rotate2norm(coefs, mat):
    """Fix orientation of point cloud"""
    # Normal of fitted plane
    norm = -np.array([coefs[0], coefs[1], -1])

    # Normalise the normal
    norm = norm / np.linalg.norm(norm)

    basis_x = np.array([1, 0, 0])

    # Find axis of rotation perpendicular to norm and basis_x
    axis = np.cross(norm, basis_x)
    axis = axis / np.linalg.norm(axis)
    angle = -np.arccos(np.dot(norm, basis_x))

    # Use Rodrigues rotation formula to calculate rotation matrix
    C = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
    rot_mat =  np.identity(3) + np.sin(angle)*C + (1-np.cos(angle))*np.dot(C, C)

    # rot_mat : basis_x --> normal. Inverse reverses this
    inv_rot = np.linalg.inv(rot_mat)
    mat2d = np.dot(inv_rot, mat.T).T

    # Only select the highest third of points to remove unnecessary objects
    # before flattening
    return mat2d[mat2d[:, 2].argsort()[:80000]][:,:2]


def create_img(ceil):
    fig, ax = plt.subplots(1)
    ax.scatter(ceil[:,0], ceil[:, 1], color='k', alpha=0.002)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    # Writes the changes
    fig.canvas.draw()

    # Supress popup
    plt.ioff()

    # Turn plot into a np matrix
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_edge(img):
    """Find edges in ceiling"""
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    sigma = np.std(blur)
    mean = np.mean(blur)
    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))

    # Utilising Canny edge detection
    return cv2.Canny(blur, lower, upper)


def _kmeans(grad, n_clusters=2):
    km = KMeans(n_clusters=n_clusters)
    km.fit(grad)
    km.predict(grad)
    return km.labels_


def detect_lines(edge):
    """Detect any lines in the wall contour"""
    lines = probabilistic_hough_line(edge, threshold=2, line_length=20, line_gap=1)

    # The first point that the line joines, ie the start
    starts = np.array([[i[0][0],i[0][1]] for i in lines])

    # Vector gradient of each line
    grad = np.array([[np.abs(i[1][0]-i[0][0]),np.abs(i[1][1]-i[0][1])] for i in lines])
    
    # Normalise the gradients for each line
    norms = np.sqrt(grad[:,0]**2 + grad[:,1]**2)[:, np.newaxis]
    norms = np.hstack((norms, norms)) + 0.001
    return grad / norms, starts


def detect_img_corners(grad, starts):
    """Detect the points lying at the corners of the contour"""
    # Classify each gradient into two groups
    labels = _kmeans(grad)

    # Stack gradients and starts over their respective labels
    grad_classified = np.hstack((grad, labels[:, np.newaxis]))
    starts_classified = np.hstack((starts, labels[:, np.newaxis]))

    # Split into two groups
    starts0 = starts_classified[starts_classified[:,2] == 0.][:,:2]
    starts1 = starts_classified[starts_classified[:,2] == 1.][:,:2]

    # Find orthonormal basis
    b0 = np.mean(grad_classified[grad_classified[:,2] == 0.][:,:2], axis=0)
    b1 = np.array([b0[1], b0[0]])

    # Further split the lines to differentiate between opposite walls
    label0 = _kmeans(starts0)
    label1 = _kmeans(starts1)

    starts0 = np.hstack((starts0, label0[:, np.newaxis]))
    starts1 = np.hstack((starts1, label1[:, np.newaxis]))

    # Opposite walls
    s00 = np.mean(starts0[starts0[:,2] == 0.][:,:2], axis=0)
    s01 = np.mean(starts0[starts0[:,2] == 1.][:,:2], axis=0)

    s10 = np.mean(starts1[starts1[:,2] == 0.][:,:2], axis=0)
    s11 = np.mean(starts1[starts1[:,2] == 1.][:,:2], axis=0)


    # Solve for the intersections of these lines
    points = np.zeros((1, 2))
    a = np.vstack((b0, -b1)).T
    for i0 in [s00, s01]:
        for i1 in [s10, s11]:
            b = i0[:, np.newaxis] - i1[:, np.newaxis]
            x = np.linalg.solve(a, b)
            x = i1 - x[1] * b1
            points = np.vstack((points, x[np.newaxis, :]))
    return points[1:, :]


def calc_scale_factor(ceil, starts):
    """Scale from image dimensions to metres"""
    xmin_c, xmax_c = np.min(ceil[:, 0]), np.max(ceil[:, 0])
    ymin_c, ymax_c = np.min(ceil[:, 1]), np.max(ceil[:, 1])
    shape_c = xmax_c - xmin_c, ymax_c - ymin_c

    xmin_s, xmax_s = np.min(starts[:, 0]), np.max(starts[:, 0])
    ymin_s, ymax_s = np.min(starts[:, 1]), np.max(starts[:, 1])
    shape_s = xmax_s - xmin_s, ymax_s - ymin_s

    factor_x = shape_s[0] / shape_c[0]
    factor_y = shape_s[1] / shape_c[1]
    return np.mean((factor_y, factor_x))


def normalise_corners(corners):
    """Create cornes that are perpendicular to each other"""
    corner_arr = corners.flatten()
    temp = np.vstack((corner_arr, np.zeros_like(corner_arr))).T

    labels = _kmeans(temp, 4)

    d = {}
    for i in range(4):
        d[i] = np.min(corner_arr[labels == i])
    for i, j in enumerate(labels):
        corner_arr[i] = d[j]
    return corner_arr.reshape((4,2))


def detect_corners(PATH):
    """Detect positions of corners relative to each other"""
    mat, df = read_xyz(PATH)
    points = find_ceiling(0, df, 500)
    coefs, _ = fit_plane2ceil(points)
    ceil = rotate2norm(coefs, mat)
    img = create_img(ceil)
    edge = get_edge(img)
    grad, starts = detect_lines(edge)
    points = detect_img_corners(grad, starts)
    factor = calc_scale_factor(ceil, starts)
    corners = (points - points.min()) / factor
    return normalise_corners(corners)


def main():
    if len(sys.argv) > 1:
        PATH = sys.argv[1]
    else:
        PATH = 'a.xyz'
    print(detect_corners(PATH))


if __name__ == "__main__":
    main()
