import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sys

def read_xyz(PATH):
    """Create df and matrix from data"""
    df = pd.read_csv(PATH, sep=' ')

    # Any extra columns detected are irrelevant
    df = df.drop(df.columns[3:], axis=1)
    mat = df.to_numpy()
    return mat, df


def find_ceiling(col_num, df, n_divs, rev=True):
    """Finds the coordinates of the ceiling points"""
    points = np.zeros((1, 3))

    col = df.columns[col_num]
    n_iter = len(df[col]) // n_divs
    ordf = df.sort_values(by=[col])
    
    rng = range(n_iter)
    if rev:
        rng = reversed(range(n_iter))
    # Reversed to capture ceiling rather than wall
    for i in rng:
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
    matv2 = np.dot(inv_rot, mat.T).T

    # Only select the highest third of points to remove unnecessary objects
    # before flattening
    matv3 = matv2[matv2[:, 1].argsort()]#[int(len(mat[:,0])//(1.5)):]]
    matv3[:, [1, 2]] = matv3[:, [2, 1]]
    # return matv2[matv2[:, 1].argsort()]
    return matv3


def normalise(PATH):
    """Detect positions of corners relative to each other"""
    mat, df = read_xyz(PATH)
    points = find_ceiling(0, df, 2000)
    points_r = find_ceiling(0, df, 2000, False)
    if len(points) < len(points_r):
        print('Yes')
        points = points_r
    coefs, _ = fit_plane2ceil(points)
    ceil = rotate2norm(coefs, mat)
    np.savetxt(f'{PATH[:-4]}.csv', ceil, delimiter=',')


def main():
    if len(sys.argv) > 1:
        PATH = sys.argv[1]
    else:
        PATH = 'a.xyz'
    normalise(PATH)

if __name__ == "__main__":
    main()