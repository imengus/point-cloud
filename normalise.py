from pc import *

def normalise(PATH):
    """Detect positions of corners relative to each other"""
    mat = read_xyz(PATH)
    edge = to_edge(mat)
    b0, b1, _, _ = classify_perpendicular(edge)
    new_mat, _ = rotate3d(b0, mat)
    np.savetxt(f'{PATH[:-4]}.csv', new_mat, delimiter=',')


def main():
    if len(sys.argv) > 1:
        PATH = sys.argv[1]
    else:
        PATH = 'a.xyz'
    normalise(PATH)

if __name__ == "__main__":
    main()