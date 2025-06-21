import numpy as np
import numpy.typing as npt
import pytest
from src.t_tensor_state import construct_T_tensor_state


def get_T_tensor_vector(num_qubits: int) -> npt.NDArray[np.complex128]:
    """Statevector of |T>^⊗n for verification."""
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    plus = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
    T_plus = T @ plus
    state = np.array([1 + 0j], dtype=np.complex128)
    for _ in range(num_qubits):
        state: npt.NDArray[np.complex128] = np.kron(
            state, T_plus)  # type: ignore[assignment]
    return state


@pytest.mark.parametrize(
    "num_qubits",
    range(2, 16),
    ids=[f"{i} qubits" for i in range(2, 16)]
)
def test_construct_T_tensor_state(num_qubits: int):
    state = construct_T_tensor_state(num_qubits)
    state_vector = state.state_vector
    expected_vector = get_T_tensor_vector(num_qubits)
    assert np.allclose(state_vector, expected_vector), (
        f"Failed for {num_qubits} qubits: {state_vector} != {expected_vector}"
    )
