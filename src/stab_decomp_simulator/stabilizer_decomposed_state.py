import numpy as np
import numpy.typing as npt

from .stabilizer_state_ch_form import StabilizerStateChForm


class StabilizerDecomposedState:
    def __init__(
        self, n_qubits: int,
        stabilizers: list[StabilizerStateChForm],
        coeffs: list[complex]
    ):

        """Initialize the decomposed state.

        Args:
            n_qubits: The number of qubits in the state.
            stabilizers: The list of stabilizers.
            coeffs: The coefficients for each stabilizer.
        """
        self.n_qubits = n_qubits
        self.stabilizers = stabilizers
        self.coeffs = coeffs
        self._validate()

    def _validate(self):
        """Validate the decomposed state.

        Raises:
            ValueError: If the number of stabilizers and coefficients do not
            match.
        """
        if len(self.stabilizers) != len(self.coeffs):
            raise ValueError(
                "Number of stabilizers must match number of coefficients.")
        # TODO: Add more validation checks as needed

    def inner(self, other: 'StabilizerDecomposedState') -> complex:
        """Compute the inner product ⟨self|other⟩."""
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                "Cannot compute inner product: number of qubits differ.")
        # Sum over all pairs of stabilizer components
        total: complex = 0 + 0j
        for c1, stab1 in zip(self.coeffs, self.stabilizers):
            for c2, stab2 in zip(other.coeffs, other.stabilizers):
                total += c1.conjugate() * c2 * stab1.inner(stab2)
        return total

    def inner_product_naive(
        self,
        other: 'StabilizerDecomposedState'
    ) -> complex:
        """Compute the inner product using the state vectors."""
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                "Cannot compute inner product: number of qubits differ.")
        # Compute the state vectors
        vec1 = self.state_vector
        vec2 = other.state_vector
        # Compute the inner product
        return vec1.conjugate().dot(vec2)

    @property
    def state_vector(self) -> npt.NDArray[np.complex128]:
        """Compute the state vector from the stabilizer decomposition."""
        # Initialize the state vector
        state = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        for coeff, stab in zip(self.coeffs, self.stabilizers):
            state += coeff * stab.state_vector()
        return state

    def tensor_product(
        self,
        other: 'StabilizerDecomposedState'
    ) -> 'StabilizerDecomposedState':
        """Compute the tensor product of two decomposed states."""
        new_n_qubits = self.n_qubits + other.n_qubits
        new_stabilizers = []
        new_coeffs = []
        for c1, stab1 in zip(self.coeffs, self.stabilizers):
            for c2, stab2 in zip(other.coeffs, other.stabilizers):
                new_stabilizers.append(stab1.kron(stab2))
                new_coeffs.append(c1 * c2)
        return StabilizerDecomposedState(
            new_n_qubits, new_stabilizers, new_coeffs)

    def exp_value(self, observable: str) -> complex:
        """Compute the expectation value of an observable.
        The observable is given as a Pauli string (e.g., 'XXY').
        """
        if len(observable) != self.n_qubits:
            raise ValueError("Observable length must match number of qubits.")
        if not all(c in 'XYZI' for c in observable):
            raise ValueError(
                "Observable must be a valid Pauli string (X, Y, Z, I)."
                )

        exp_value = 0 + 0j

        n = len(self.stabilizers)

        # cross terms
        for i in range(n - 1):
            for j in range(i + 1, n):
                factor = np.conj(self.coeffs[j]) * self.coeffs[i]
                stab_i_copy = self.stabilizers[i].copy()
                stab_j_copy = self.stabilizers[j].copy()
                stab_i_copy.apply_pauli_string(observable)
                exp_value += 2 * np.real(
                    factor * stab_j_copy.inner(stab_i_copy))

        # diagonal terms
        for i in range(n):
            coeff = self.coeffs[i]
            factor = coeff.real ** 2 + coeff.imag ** 2
            stab_copy = self.stabilizers[i].copy()
            stab_copy.apply_pauli_string(observable)
            exp_value += factor * self.stabilizers[i].inner(stab_copy)

        return exp_value
