import numpy as np
from dft2cc.utils.mol import MASS


def get_barycenter(molecular):
    """
    Get the barycenter
    """
    barycenter = np.array([0, 0, 0], dtype=np.float64)
    mass = 0.0
    for mol in molecular:
        mass += MASS[mol[0]]
        barycenter[0] += mol[1] * MASS[mol[0]]
        barycenter[1] += mol[2] * MASS[mol[0]]
        barycenter[2] += mol[3] * MASS[mol[0]]
    return barycenter / mass


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if np.abs(s) < 1e-12:
        print(vec1, vec2)
        return np.eye(3)
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def get_inertia_moment(molecular):
    """
    Get the moment of inertia
    """
    I = np.zeros((3, 3), dtype=np.float64)
    for mol in molecular:
        I[0, 0] += MASS[mol[0]] * (mol[2] ** 2 + mol[3] ** 2)
        I[1, 1] += MASS[mol[0]] * (mol[1] ** 2 + mol[3] ** 2)
        I[2, 2] += MASS[mol[0]] * (mol[1] ** 2 + mol[2] ** 2)
        I[0, 1] -= MASS[mol[0]] * mol[1] * mol[2]
        I[0, 2] -= MASS[mol[0]] * mol[1] * mol[3]
        I[1, 2] -= MASS[mol[0]] * mol[2] * mol[3]
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
    return I


def rotate(molecular):
    """
    Rotate the MOs to the canonical basis
    """
    # Get the barycenter
    barycenter = get_barycenter(molecular)

    for mol in molecular:
        mol[1] -= barycenter[0]
        mol[2] -= barycenter[1]
        mol[3] -= barycenter[2]

    I = get_inertia_moment(molecular)
    eig_val, eig_vec = np.linalg.eig(I)
    index1 = np.argsort(eig_val)[2]
    index2 = np.argsort(eig_val)[1]
    if np.abs((eig_val[index1] - eig_val[index2])) > 1e-12:
        list_max_eig = eig_vec[:, index1]
        rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [0, 0, 1])
        # rotate the molecule
        for mol in molecular:
            x_array = np.array(mol[1:])
            x_array = rotation_matrix @ x_array
            mol[1] = x_array[0]
            mol[2] = x_array[1]
            mol[3] = x_array[2]

        I = get_inertia_moment(molecular)
        eig_val, eig_vec = np.linalg.eig(I)
        index2 = np.argsort(eig_val)[1]
        list_max_eig = eig_vec[:, index2]
        rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [0, 1, 0])
        for mol in molecular:
            x_array = np.array(mol[1:])
            x_array = rotation_matrix @ x_array
            mol[1] = x_array[0]
            mol[2] = x_array[1]
            mol[3] = x_array[2]
    else:
        index1 = np.argsort(eig_val)[0]
        list_max_eig = eig_vec[:, index1]

        if (np.sqrt(list_max_eig[1] ** 2 + list_max_eig[2] ** 2)) > 1e-6:
            rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [1, 0, 0])

            # rotate the molecule
            for mol in molecular:
                x_array = np.array(mol[1:])
                x_array = rotation_matrix @ x_array
                mol[1] = x_array[0]
                mol[2] = x_array[1]
                mol[3] = x_array[2]

        for mol in molecular:
            x_array = np.array(mol[1:])
            index_ = np.argsort(np.abs(x_array))[-1]
            mol[1] = x_array[index_]

    if molecular[0][1] > 0:
        for i_mol, mol in enumerate(molecular):
            mol[1] = -mol[1]
    if molecular[0][2] > 0:
        for i_mol, mol in enumerate(molecular):
            mol[2] = -mol[2]
    if molecular[0][3] > 0:
        for i_mol, mol in enumerate(molecular):
            mol[3] = -mol[3]
