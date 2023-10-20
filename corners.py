from main import *

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
        PATH = 'a.xyz'
    print(detect_corners(PATH))

if __name__ == "__main__":
    main()
