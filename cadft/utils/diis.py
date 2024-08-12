import numpy as np

class DIIS:
    """
    DIIS for the Fock matrix.
    """

    def __init__(self, nao, n=50):
        self.n = n
        self.errors = np.zeros((n, nao, nao))
        self.mat_fock = np.zeros((n, nao, nao))
        self.step = 0

    def add(self, mat_fock_, error):
        """
        Add the new Fock matrix and error.
        """
        # rolling back [_, _, 1, 2, 3] -> [_, 1, 2, 3, _]
        self.mat_fock = np.roll(self.mat_fock, -1, axis=0)
        self.errors = np.roll(self.errors, -1, axis=0)
        self.mat_fock[-1, :, :] = mat_fock_
        self.errors[-1, :, :] = error

    def hybrid(self):
        """
        Return the hybrid Fock matrix.
        """
        self.step += 1
        mat = np.zeros((self.n + 1, self.n + 1))
        mat[:-1, :-1] = np.einsum("inm,jnm->ij", self.errors, self.errors)
        mat[-1, :] = -1
        mat[:, -1] = -1
        mat[-1, -1] = 0
        b = np.zeros(self.n + 1)
        b[-1] = -1

        if self.step < self.n:
            c = np.linalg.solve(
                mat[-(self.step + 1) :, -(self.step + 1) :], b[-(self.step + 1) :]
            )
            mat_fock = np.sum(
                c[:-1, np.newaxis, np.newaxis] * self.mat_fock[-self.step :], axis=0
            )
            return mat_fock
        else:
            c = np.linalg.solve(mat, b)
            mat_fock = np.sum(c[:-1, np.newaxis, np.newaxis] * self.mat_fock, axis=0)
            return mat_fock
