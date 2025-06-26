import numpy as np
import numpy.typing as npt

from .stabilizer_decomposed_state import StabilizerDecomposedState
from .stabilizer_state_ch_form import StabilizerStateChForm


def apply_pauli_string(state: StabilizerStateChForm, pauli_string: str):
    """Apply a Pauli string to the stabilizer state."""
    for i, char in enumerate(pauli_string):
        if char == 'X':
            state.apply_x(i)
        elif char == 'Y':
            state.apply_y(i)
        elif char == 'Z':
            state.apply_z(i)
        elif char == 'I':
            continue  # Identity does nothing
        else:
            raise ValueError(
                f"Unsupported Pauli character '{char}' at position {i} "
                f"in '{pauli_string}'.")


def zero_minus_i_one_state(num_qubits: int) -> StabilizerStateChForm:
    """|0^n> - i|1^n> as a single stabilizer CH form."""
    state = StabilizerStateChForm(num_qubits=num_qubits)
    state.apply_h(0)
    for i in range(1, num_qubits):
        state.apply_cx(0, i)
    # S† on first qubit (phase -i for |1>)
    state.apply_z(0, exponent=-0.5)
    return state


def even_parity_state(num_qubits: int) -> StabilizerStateChForm:
    """1/√N ∑_{|x| even} |x>"""
    state = StabilizerStateChForm(num_qubits=num_qubits)
    for i in range(num_qubits - 1):
        state.apply_h(i)
    for i in range(num_qubits - 1):
        state.apply_cx(i, num_qubits - 1)
    return state


def even_parity_phase_flip_state(num_qubits: int) -> StabilizerStateChForm:
    """Even parity plus phase flip by CZ on all pairs."""
    state: StabilizerStateChForm = StabilizerStateChForm(num_qubits=num_qubits)
    # build even parity
    for i in range(num_qubits - 1):
        state.apply_h(i)
    for i in range(num_qubits - 1):
        state.apply_cx(i, num_qubits - 1)
    # apply CZ between each pair
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            state.apply_cz(i, j)
    return state


def append_X_S(state: StabilizerStateChForm, qubit: int) -> None:
    """Apply X then S on target qubit."""
    state.apply_x(axis=qubit)
    state.apply_z(axis=qubit, exponent=0.5)


def construct_cat_1_state() -> StabilizerDecomposedState:
    """|cat_1> = |0>"""
    stab = StabilizerStateChForm(num_qubits=1)
    return StabilizerDecomposedState(1, [stab], [1 + 0j])


def construct_cat_2_state() -> StabilizerDecomposedState:
    """|cat_2> = (|00> + i|11>)/√2"""
    stab = StabilizerStateChForm(num_qubits=2)
    stab.apply_h(0)
    stab.apply_cx(0, 1)
    stab.apply_z(1, exponent=0.5)
    return StabilizerDecomposedState(2, [stab], [1 + 0j])


def construct_cat_4_state() -> StabilizerDecomposedState:
    """|cat_4> as superposition of two stabilizers."""
    st1 = zero_minus_i_one_state(4)
    st2 = even_parity_state(4)
    coeffs = [(1 - 1j) / 2, 1j]
    return StabilizerDecomposedState(4, [st1, st2], coeffs)


def construct_cat_6_state() -> StabilizerDecomposedState:
    """|cat_6> as superposition of three stabilizers."""
    st1 = zero_minus_i_one_state(6)
    st2 = even_parity_state(6)
    st3 = even_parity_phase_flip_state(6)
    coeffs = [0.5, (-1 + 1j) / 2, (-1 - 1j) / 2]
    return StabilizerDecomposedState(6, [st1, st2, st3], coeffs)


def construct_cat_state(num_qubits: int) -> StabilizerDecomposedState:
    """Dispatch generic cat_m factory for m=1,2,4,6."""
    if num_qubits == 1:
        return construct_cat_1_state()
    if num_qubits == 2:
        return construct_cat_2_state()
    if num_qubits == 3:
        cat_4 = construct_cat_4_state()
        return reduced_cat_state(cat_4)
    if num_qubits == 4:
        return construct_cat_4_state()
    if num_qubits == 5:
        cat_6 = construct_cat_6_state()
        return reduced_cat_state(cat_6)
    if num_qubits == 6:
        return construct_cat_6_state()
    elif num_qubits >= 7:
        cat_pair = construct_cat_state(num_qubits - 4).tensor_product(
            construct_cat_state(6))
        projected = project_cat_2(cat_pair, [num_qubits - 5, num_qubits - 4])
        return projected
    raise NotImplementedError(
        f"cat state for {num_qubits} qubits not implemented.")


def reduced_cat_state(
    state: StabilizerDecomposedState
) -> StabilizerDecomposedState:
    new_stabilizers = []
    for stab in state.stabilizers:
        copy = stab.copy()
        copy.project_0(state.n_qubits - 1)
        # new_stabilizers.append(remove_qubit(copy, state.n_qubits - 1))
        new_stabilizers.append(copy.remove_qubit(state.n_qubits - 1))
    return StabilizerDecomposedState(
        state.n_qubits - 1,
        new_stabilizers,
        state.coeffs
    )


def _project_cat_2_on_ch_form(
    state: StabilizerStateChForm,
    qubits: list[int]
) -> StabilizerStateChForm:
    """Project state onto |cat_2> subspace."""
    # Project onto the |00> + i|11> subspace
    state.apply_z(qubits[0], exponent=-0.5)  # Apply S† on qubit 0
    state.apply_cx(qubits[0], qubits[1])  # Apply CNOT from qubit 0 to qubit 1
    state.apply_h(qubits[0])  # Apply Hadamard on qubit 0
    state.project_0(qubits[0])  # Project qubit 0 onto |0>
    state.project_0(qubits[1])  # Project qubit 1 onto |0>
    # Remove qubit 0 and 1
    removed = state.remove_qubit(qubits[1]).remove_qubit(qubits[0])

    return removed


def project_cat_2(
    state: StabilizerDecomposedState,
    qubits: list[int]
) -> StabilizerDecomposedState:
    new_stabilizers = []
    new_coeffs = []
    n_qubits_orig = state.n_qubits
    for stab, coeff in zip(state.stabilizers, state.coeffs):
        try:
            projected = _project_cat_2_on_ch_form(stab.copy(), qubits)
            new_stabilizers.append(projected)
            new_coeffs.append(coeff)
        except ValueError:
            print("Failed to project stabilizer")
            continue

    return StabilizerDecomposedState(
        n_qubits_orig - 2,
        new_stabilizers,
        new_coeffs
    )


def construct_T_tensor_state(num_qubits: int) -> StabilizerDecomposedState:
    """
    Construct |T⟩^⊗m as a StabilizerDecomposedState
    |T⟩^⊗m = √2(|T><T|⊗I)|cat_m>
    """
    cat_state = construct_cat_state(num_qubits)

    new_stabilizers = [stab.copy() for stab in cat_state.stabilizers]
    for stab in cat_state.stabilizers:
        new_stab = stab.copy()
        append_X_S(new_stab, 0)
        new_stabilizers.append(new_stab)

    new_coeffs: list[complex] = []
    for coeff in cat_state.coeffs:
        new_coeffs.append(coeff / np.sqrt(2))
    for coeff in cat_state.coeffs:
        new_coeffs.append(coeff * (1 - 1j) / 2)

    return StabilizerDecomposedState(num_qubits, new_stabilizers, new_coeffs)


def get_T_tensor_vector(num_qubits: int) -> npt.NDArray[np.complex128]:
    """Statevector of |T>^⊗n for verification."""
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    plus = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128)
    T_plus = T @ plus
    state = np.array([1 + 0j], dtype=np.complex128)
    for _ in range(num_qubits):
        state: npt.NDArray[np.complex128] = np.kron(
            state, T_plus)  # type: ignore
    return state


def get_T_bottom_tensor_vector(num_qubits: int) -> npt.NDArray[np.complex128]:
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    T_bottom_plus = Z @ get_T_tensor_vector(1)
    state = np.array([1 + 0j], dtype=np.complex128)
    for _ in range(num_qubits):
        state: npt.NDArray[np.complex128] = np.kron(
            state, T_bottom_plus)  # type: ignore
    return state
