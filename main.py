import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from skimage.transform import probabilistic_hough_line
import cv2

import warnings
import sys
warnings.filterwarnings("ignore")

class Side:
    def __init__(self) -> None:
        # Initial vector gradient
        self.b = None
        
        # Start and end points of each line in a matrix
        self.points = None

        # Points in 2d rotated
        self.rotated = None

        # New orthonormal basis (either [1 0] or [0 1])
        self.ob = None

        # 1D representation of points
        self.oned = None
        
        # Classified point dictionary
        self.d = {}

        # Number of clusters obtained
        self.n = None


def read_xyz(PATH):
    """Create df and matrix from data"""
    df = pd.read_csv(PATH, sep=' ')

    # Any extra columns detected are irrelevant
    df = df.drop(df.columns[3:], axis=1)
    mat = df.to_numpy()

    mu = np.mean(mat, axis=0)
    mat = mat - mu

    # swapping the y and z axes means the ceiling is oriented correctly
    mat[:, [1, 2]] = mat[:, [2, 1]]

    return mat


def to_edge(mat):
    # Sorting in terms of height of points, we take the top 60% to avoid noise
    ceil = mat[mat[:, 2].argsort()][int(mat.shape[0]*0.6):,:]

    # Only take x and y dimensions
    ceil = ceil[:,:2]

    # To implement line detection, we must split thedata into voxels. 
    # The easiest way to do this is to plot a graph
    fig, ax = plt.subplots(1)
    ax.scatter(ceil[:,0], ceil[:, 1], color='k')
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    # Writes the changes
    fig.canvas.draw()

    # Supress popup
    plt.ioff()

    # Turn plot into a np matrix
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Blur any noise using a 9x9 kernel
    blur = cv2.GaussianBlur(img, (9, 9), 0)

    # Take only the sharpest edges
    return cv2.Canny(blur, 200, 300)


def classify_perpendicular(edge):
    # Using the Hough transform, we can detect the lines in this image.
    # These lines are represented by two (x, y) pairs
    lines = probabilistic_hough_line(edge, threshold=0, line_length=1, line_gap=10)

    # Find the vector gradients of the lines 
    # (e.g. y = [1 2] + y * [0 1] for a vertical line passing through [1 2])
    grad = np.array([[(i[1][0]-i[0][0]),(i[1][1]-i[0][1])] for i in lines])

    # Use l2 normalisation for each vector gradient
    norms = np.linalg.norm(grad, axis=1)[:, np.newaxis]
    norms = np.tile(norms, (1, 2))

    # Divide both components of the vector elementwise
    grad = grad / norms

    # Using K-Means we split the lines into two groups for perpendicular walls
    km = KMeans(n_clusters=2).fit(grad)

    # Returns a list of 0/1s corresponding to the group each gradient is assigned
    labels = km.labels_

    # Append to each (x, y) pair the labels. 
    labelled_grad = np.hstack((grad, labels[:, np.newaxis]))
    # The first and second new orthonormal basis vector (compared to [1 0] and [0 1])
    b0 = np.median(labelled_grad[labelled_grad[:,2] == 0.][:,:2], axis=0)
    b1 = np.array([1, -b0[0] / b0[1]])
    b1 = b1 / np.linalg.norm(b1)


    return b0, b1, lines, labels


def classify_points(lines, labels, b0, b1):
    # First and second point that make up each line
    arr0 = np.array(([[i[0][0],i[0][1]] for i in lines]))
    arr1 = np.array([[i[1][0],i[1][1]] for i in lines])

    # Stack them into a (len(arr0) + len(arr1)) x 2 matrix
    points = np.vstack((arr0, arr1))
    p_mean = np.mean(points, axis=0)
    points = points - p_mean

    # The start and end points of the lines will have the same classification,
    # so the labels are repeated
    labelled_points = np.hstack((points, np.hstack((labels,labels))[:, np.newaxis]))

    s0, s1 = Side(), Side()
    d = [s0, s1]
    d[0].b, d[1].b = b0, b1

    # Classify the points into two groups
    d[0].points, d[1].points = (labelled_points[labelled_points[:,2] == i][:,:2] for i in range(2))
    return d


def rotate3d(b0, mat):
    # Now that we know the bases/ eigenvectors, we can rotate the 3d point cloud
    # axis of rotation = z axis
    # normal of wall (without loss of generality) = [b0[0] b0[1] 0]

    basis = np.array([1, 0, 0])

    # Normal of fitted plane
    norm = np.array([b0[0], b0[1], 0])
    # Normalise the normal
    norm = norm / np.linalg.norm(norm)

    angle = -np.arccos(np.dot(norm, basis))

    # Find axis of rotation perpendicular to norm and basis_x
    axis = np.array([0, 0, 1])
    # axis = axis / np.linalg.norm(axis)

    # Use Rodrigues rotation formula to calculate rotation matrix
    C = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
    rot_mat =  np.identity(3) + np.sin(angle)*C + (1-np.cos(angle))*np.dot(C, C)
    # Rotate each point
    new_mat = np.dot(rot_mat, mat.T).T
    return new_mat, angle


def rotate2d(d, angle):
    rot_mat_2d = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
    for i in range(2):

        # Cardinal bases in the form of [1 0], [0 1]
        d[i].ob = np.uint8(np.round(np.dot(d[i].b, rot_mat_2d)))

        # Along which axes do the points vary
        idx = np.argmin(d[i].ob)

        # Transform points
        rotated = np.dot(d[i].points, rot_mat_2d)

        # Sort the rotated points
        d[i].rotated = rotated[rotated[:, idx].argsort()]

        # Remove extra dimension
        d[i].oned = d[i].rotated[:, idx]
    return d


def cluster_1d(d, eps=2, n_thresh=2):

    # For each side
    for n in range(2):

        # Points of current side
        curr_points = d[n].oned
        curr_label = 0
        labels = np.zeros_like(curr_points)

        for i, j in enumerate(curr_points[:-1]):

            # If the next point is closer than eps units, it receives the same label
            if j + eps >= curr_points[i+1]:
                labels[i+1] = curr_label
            else:
                curr_label += 1
                labels[i+1] = curr_label
        n_labels = 0

        # Separate into clusters by label
        for i in range(curr_label+1):
            cluster = d[n].rotated[labels == float(i),:]
            
            # Only clusters with n_points greater than threshold remain to filter out noise
            if cluster.shape[0] > n_thresh:
                d[n].d[f'{n_labels}'] = cluster
                s = np.mean(cluster, axis=0)
                d[n].d[f's{n_labels}'] = s
                n_labels += 1
        d[n].n = n_labels
    return d


def calc_corners(d):
    # For appending to
    corners = np.zeros((1, 2))

    # Linearly solve to find all possible intersections
    a = np.vstack((d[0].ob, -d[1].ob))
    for i in range(d[0].n):
        for j in range(d[1].n):
            # a * lam = b
            b = d[0].d[f's{i}'] - d[1].d[f's{j}']

            # vector of parameters
            lam = np.linalg.solve(a, b[:, np.newaxis])

            # Reconstruct Cartesian point
            x0 = d[0].d[f's{i}'] - lam[0] * d[0].ob

            # Append to previous corners
            corners = np.vstack((corners, x0[np.newaxis, :]))
    return corners[1:,:]


def scale_corners(corners, new_mat):

    # Minimum corner has value 0, so they are relative to each other
    norm_corn = (corners - np.min(corners))

    n_bins = 100
    dims = np.zeros(2)

    x, y = new_mat[:, 0], new_mat[:, 1]

    for i, j in enumerate([x, y]):

        # Interval in which bins are located
        interval = np.min(j), np.max(j)
        bins = np.linspace(*interval, n_bins)

        # Take midpoint of each bin
        midpoints = (bins[1:] + bins[:-1]) / 2

        # Find which midpoints are greater than 0
        pos = midpoints > 0

        # Count per bin
        counts, _ = np.histogram(j, bins, density=True)

        # For each side of the number line
        ub, lb = midpoints[pos][np.argmax(counts[pos])], midpoints[~pos][np.argmax(counts[~pos])]
        dims[i] = ub - lb

    # Find the factor to be divided by, taking into account calculations from both dims
    factor = np.mean(np.ptp(norm_corn, axis=0) / dims)

    # Round to 2dp
    return np.round(norm_corn / factor, 2)


def detect_corners(PATH):
    """Detect positions of corners relative to each other"""
    mat = read_xyz(PATH)
    edge = to_edge(mat)
    b0, b1, lines, labels = classify_perpendicular(edge)
    new_mat, angle = rotate3d(b0, mat)
    d = classify_points(lines, labels, b0, b1)
    d = rotate2d(d, angle)
    d = cluster_1d(d)
    corners = calc_corners(d)
    return scale_corners(corners, new_mat)


def main():
    if len(sys.argv) > 1:
        PATH = sys.argv[1]
    else:
        PATH = 'data/examples/a.xyz'
    print(detect_corners(PATH))

if __name__ == "__main__":
    main()
