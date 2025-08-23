import numpy as np

from .stabilizer_decomposed_state import StabilizerDecomposedState
from .stabilizer_state_ch_form import StabilizerStateChForm


def construct_toffoli_state() -> StabilizerDecomposedState:
    """Constructs the Toffoli state |Toffoli⟩ = (|000⟩ + |100⟩ + |010⟩ + |111⟩) / 2
    as a stabilizer-decomposed state:
     |Toffoli⟩ = (|0+0⟩ + |1,Bell⟩) / √2
    """
    # |0+0> part
    stab1 = StabilizerStateChForm(num_qubits=3)
    stab1.apply_h(1)

    # |1,Bell> part
    stab2 = StabilizerStateChForm(num_qubits=3)
    stab2.apply_x(0)
    stab2.apply_h(1)
    stab2.apply_cx(1, 2)

    coeffs = [1 / np.sqrt(2), 1 / np.sqrt(2)]

    return StabilizerDecomposedState(3, [stab1, stab2], coeffs)


def construct_toffoli_tensor_state(num_tensors: int) -> StabilizerDecomposedState:
    """Constructs |Toffoli⟩^⊗m as a StabilizerDecomposedState"""
    if num_tensors < 1:
        raise ValueError("Number of tensors must be at least 1.")

    if num_tensors == 1:
        return construct_toffoli_state()

    # Recursively build the tensor product
    smaller_tensor = construct_toffoli_tensor_state(num_tensors - 1)
    toffoli_state = construct_toffoli_state()

    new_stabilizers = []
    new_coeffs: list[complex] = []

    for stab1, coeff1 in zip(smaller_tensor.stabilizers, smaller_tensor.coeffs):
        for stab2, coeff2 in zip(toffoli_state.stabilizers, toffoli_state.coeffs):
            new_stab = stab1.kron(stab2)
            new_stabilizers.append(new_stab)
            new_coeffs.append(coeff1 * coeff2)

    return StabilizerDecomposedState(
        smaller_tensor.n_qubits + 3,
        new_stabilizers,
        new_coeffs
    )
