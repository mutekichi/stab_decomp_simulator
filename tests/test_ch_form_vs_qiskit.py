import random

import numpy as np
import pytest
from qiskit.circuit.library import (CXGate, CZGate, HGate, SGate, XGate, YGate,
                                    ZGate)
# Import necessary modules from Qiskit
from qiskit.quantum_info import (DensityMatrix, Operator, Statevector,
                                 partial_trace)
# Import the class to be tested
from src.stab_decomp_simulator.stabilizer_state_ch_form import \
    StabilizerStateChForm

# --- Helper Functions ---


def reorder_to_little_endian(vec: np.ndarray, num_qubits: int) -> np.ndarray:
    """
    Reorders a state vector from a big-endian convention to Qiskit's
    little-endian convention by reversing the bit order of the indices.
    """
    new_vec = np.zeros_like(vec)
    for i in range(len(vec)):
        # Format index to a binary string, padded with zeros
        binary_str = format(i, f'0{num_qubits}b')
        # Reverse the bit string to convert endianness
        reversed_binary_str = binary_str[::-1]
        # The new index is the integer value of the reversed bit string
        new_index = int(reversed_binary_str, 2)
        new_vec[new_index] = vec[i]
    return new_vec


def vectors_are_equivalent(
    ch_vec: np.ndarray, qiskit_vec: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """
    Checks if two state vectors are equivalent up to a global phase.
    If they are not, it prints the vectors for debugging.
    """
    # Check for zero vectors
    is_ch_zero = np.allclose(ch_vec, 0, atol=atol)
    is_qiskit_zero = np.allclose(qiskit_vec, 0, atol=atol)
    if is_ch_zero and is_qiskit_zero:
        return True
    if is_ch_zero or is_qiskit_zero:
        print("\n--- Mismatch: One vector is zero while the other is not. ---")
        print(f"CH Form Vector (norm={np.linalg.norm(ch_vec)}):\n{ch_vec}")
        print(f"Qiskit Vector  (norm={np.linalg.norm(qiskit_vec)}):\n{qiskit_vec}")
        return False

    # Find the first non-zero element to align the global phase
    try:
        idx1 = np.flatnonzero(np.abs(ch_vec) > atol)[0]
        phase1 = ch_vec[idx1] / np.abs(ch_vec[idx1])

        idx2 = np.flatnonzero(np.abs(qiskit_vec) > atol)[0]
        phase2 = qiskit_vec[idx2] / np.abs(qiskit_vec[idx2])

        qiskit_vec_aligned = qiskit_vec * (phase1 / phase2)
    except IndexError:
        # Fallback for corner cases, though zero-vector check should handle this
        return False

    # Compare the vectors
    is_close = np.allclose(ch_vec, qiskit_vec_aligned, rtol=rtol, atol=atol)
    if not is_close:
        # Print vectors for analysis when test fails (visible with 'pytest -s')
        print("\n--- State Vector Mismatch ---")
        np.set_printoptions(precision=6, suppress=True)
        print(f"CH Form Vector (norm={np.linalg.norm(ch_vec)}):")
        print(ch_vec)
        print(f"\nQiskit Vector (phase-aligned, norm={np.linalg.norm(qiskit_vec_aligned)}):")
        print(qiskit_vec_aligned)
        print("\n(For reference) Original Qiskit Vector:")
        print(qiskit_vec)
        print("---------------------------")
    return is_close


def statevector_from_density_matrix(dm: DensityMatrix) -> np.ndarray:
    """
    Extracts a state vector from a pure state density matrix.
    For rho = |psi><psi|, any non-zero column of rho is proportional to |psi>.
    """
    dm_data = dm.data
    # Find the first non-zero column and normalize it
    for i in range(dm_data.shape[1]):  # type: ignore
        col = dm_data[:, i]
        norm = np.linalg.norm(col)
        if not np.isclose(norm, 0):
            return col / norm
    # Handle the zero matrix case
    return np.zeros(dm_data.shape[0], dtype=np.complex128)


# --- Test Functions ---

@pytest.mark.parametrize("test_run", range(50))  # Run 50 random tests
def test_gate_applications_vs_qiskit(test_run: int, num_qubits: int = 4):
    """
    Tests if the state after applying a random gate sequence matches Qiskit's result.
    """
    n = num_qubits
    rng = random.Random(test_run)  # Use test_run as the seed for reproducibility

    # 1. Prepare initial states
    ch_state = StabilizerStateChForm(n, initial_state=0)
    qiskit_sv = Statevector.from_int(0, 2**n)

    # 2. Apply a sequence of random gates
    gate_set = ['x', 'y', 'z', 'h', 'cx', 'cz', 's']
    for _ in range(15):  # Number of gates to apply
        gate_name = rng.choice(gate_set)

        if gate_name in ['x', 'y', 'z', 'h']:
            q = rng.randint(0, n - 1)
            if gate_name == 's':
                # Special case for S gate, which is a Z gate with exponent 0.5
                ch_state.apply_z(q, exponent=0.5)
            else:
                getattr(ch_state, f'apply_{gate_name}')(q)
            gate_map = {'x': XGate(), 'y': YGate(), 'z': ZGate(), 'h': HGate(), 's': SGate()}
            qiskit_sv = qiskit_sv.evolve(gate_map[gate_name], [q])

        elif gate_name in ['cx', 'cz']:
            q1, q2 = rng.sample(range(n), 2)
            getattr(ch_state, f'apply_{gate_name}')(q1, q2)
            gate_map = {'cx': CXGate(), 'cz': CZGate()}
            qiskit_sv = qiskit_sv.evolve(gate_map[gate_name], [q1, q2])

    # 3. Compare the results
    ch_vector_be = ch_state.state_vector()
    qiskit_vector_le = qiskit_sv.data

    # FIX: Reorder CH vector from big-endian to little-endian for comparison
    ch_vector_le = reorder_to_little_endian(ch_vector_be, n)

    assert vectors_are_equivalent(ch_vector_le, qiskit_vector_le), \
        f"Test run {test_run} failed for random gate applications."


def test_postselection_and_removal_vs_qiskit(num_qubits: int = 4):
    """
    Tests if post-selecting to |0> and removing a qubit matches Qiskit's result.
    """
    n = num_qubits

    # 1. Prepare a non-trivial state (GHZ state) for testing
    ch_state = StabilizerStateChForm(n)
    qiskit_sv = Statevector.from_int(0, 2**n)
    ch_state.apply_h(0)
    qiskit_sv = qiskit_sv.evolve(HGate(), [0])
    for i in range(n - 1):
        ch_state.apply_cx(i, i + 1)
        qiskit_sv = qiskit_sv.evolve(CXGate(), [i, i + 1])

    qubit_to_remove = 0

    # --- Qiskit-side calculation ---
    # 2. Create the projection operator P_0 = |0><0| for the target qubit
    p0_op = Operator(np.array([[1, 0], [0, 0]]))
    ident_op = Operator(np.eye(2))
    # Qiskit's qubit order is little-endian (q_n-1, ..., q_0)
    op_list = [ident_op] * n
    op_list[n - 1 - qubit_to_remove] = p0_op
    full_proj_op = Operator.from_label('I' * n).tensor(op_list[0])  # A bit tricky, let's build it up
    full_proj_op = op_list[0]
    for i in range(1, n):
        full_proj_op = full_proj_op.tensor(op_list[i])

    # 3. Calculate the projection probability P(0) = <psi| P_0 |psi>
    qiskit_prob_0 = qiskit_sv.expectation_value(full_proj_op).real

    # 4. If projection is impossible, check that the user's code raises an error
    if np.isclose(qiskit_prob_0, 0):
        with pytest.raises(ValueError):
            ch_state.project_0(qubit_to_remove)
        return

    # 5. Calculate the post-projection state and remove the target qubit
    post_sv = qiskit_sv.evolve(full_proj_op)
    post_dm = DensityMatrix(post_sv)

    # FIX: Create a list of the single qubit TO DISCARD.
    qubits_to_trace_out = [qubit_to_remove]
    qiskit_final_dm = partial_trace(post_dm, qubits_to_trace_out)

    # Normalize the final density matrix and then extract the state vector
    qiskit_final_dm_normalized = qiskit_final_dm / qiskit_prob_0
    qiskit_final_vector_le = statevector_from_density_matrix(qiskit_final_dm_normalized)

    # --- StabilizerStateChForm-side calculation ---
    ch_prob_0 = ch_state.project_0(qubit_to_remove)
    ch_final_state = ch_state.remove_qubit(qubit_to_remove)
    ch_final_vector_be = ch_final_state.state_vector()

    # FIX: Reorder final CH vector to little-endian
    ch_final_vector_le = reorder_to_little_endian(ch_final_vector_be, n - 1)

    # --- Comparison ---
    assert np.isclose(qiskit_prob_0, ch_prob_0), "Projection probabilities do not match."
    assert vectors_are_equivalent(ch_final_vector_le, qiskit_final_vector_le), \
        "State vectors after qubit removal do not match."
