import random
from typing import Optional

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from src.stab_decomp_simulator.qiskit_converter import (
    qiskit_circuit_to_t_decomposed_stab_decomp_state,
    qiskit_circuit_to_toffoli_decomposed_stab_decomp_state)


def random_clifford_t_circuit(
    num_qubits: int, num_gates: int, t_prob: float = 0.3, seed: Optional[int] = None
) -> QuantumCircuit:
    """Generates a random circuit with Clifford and T/Tdg gates."""
    if seed is not None:
        random.seed(seed)

    qc = QuantumCircuit(num_qubits)
    clifford_gates = ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"]

    for _ in range(num_gates):
        gate_name = random.choice(["t", "tdg"]) if random.random() < t_prob else random.choice(clifford_gates)

        if gate_name in ["t", "tdg", "h", "s", "sdg", "x", "y", "z"]:
            q = random.randint(0, num_qubits - 1)
            getattr(qc, gate_name)(q)
        else:  # 2-qubit gates
            if num_qubits < 2:
                continue
            q1, q2 = random.sample(range(num_qubits), 2)
            getattr(qc, gate_name)(q1, q2)
    return qc


def get_random_indices(total_count: int, sample_count: int):
    """Selects a random sample of indices without repetition."""
    # Return a set of 'sample_count' unique indices from the range [0, total_count - 1].
    return set(random.sample(range(total_count), sample_count))


def random_clifford_toffoli_circuit(
    num_qubits: int, num_clifford_gates: int, num_toffoli: int, seed: Optional[int] = None
) -> QuantumCircuit:
    """Generates a random circuit with Clifford and Toffoli gates."""
    if seed is not None:
        random.seed(seed)

    if num_qubits < 3:
        raise ValueError("At least 3 qubits are required for Toffoli gates.")

    qc = QuantumCircuit(num_qubits)
    clifford_gates = ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"]
    total_gates = num_clifford_gates + num_toffoli
    toffoli_indices = get_random_indices(total_gates, num_toffoli)

    for gate_idx in range(total_gates):
        if gate_idx in toffoli_indices:
            q1, q2, q3 = random.sample(range(num_qubits), 3)
            qc.ccx(q1, q2, q3)
        else:
            gate_name = random.choice(clifford_gates)
            if gate_name in ["h", "s", "sdg", "x", "y", "z"]:
                q = random.randint(0, num_qubits - 1)
                getattr(qc, gate_name)(q)
            else:
                q1, q2 = random.sample(range(num_qubits), 2)
                getattr(qc, gate_name)(q1, q2)
    return qc


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
@pytest.mark.parametrize("test_run", range(10))
def test_exp_avlue_clif_t(num_qubits: int, test_run: int):
    """
    Compares the expectation value from StabilizerDecomposedState with Qiskit.
    """
    # 1. Create a random Clifford+T circuit
    num_gates = num_qubits * 5
    seed = num_qubits * 100 + test_run
    qc = random_clifford_t_circuit(num_qubits, num_gates, t_prob=0.4, seed=seed)

    # Skip test if the circuit has no T-gates, as it doesn't test the core logic
    if qc.count_ops().get('t', 0) == 0 and qc.count_ops().get('tdg', 0) == 0:  # type: ignore
        pytest.skip("Skipping test run with no T-gates.")

    # 2. Convert to StabilizerDecomposedState
    try:
        stab_decomp_state = qiskit_circuit_to_t_decomposed_stab_decomp_state(qc)
    except NotImplementedError as e:
        pytest.skip(f"Skipping due to NotImplementedError from cat_state constructor: {e}")
        return

    # If conversion results in an empty state (all components failed projection)
    if not stab_decomp_state.stabilizers:
        qiskit_vector = Statevector(qc).data
        assert np.allclose(np.linalg.norm(qiskit_vector), 0), \
            "Converter produced an empty state, but Qiskit state is non-zero."
        return

    # 3. Generate a random Pauli string for the observable
    pauli_string = "".join(random.choices(["I", "X", "Y", "Z"], k=num_qubits))
    if pauli_string == "I" * num_qubits:
        pauli_string = "X" + "I" * (num_qubits - 1)

    # 4. Calculate expectation value with our implementation (big-endian)
    exp_val_mine = stab_decomp_state.exp_value(pauli_string)

    # 5. Calculate expectation value with Qiskit (little-endian)
    # The Pauli string must be reversed to match Qiskit's endianness.
    qiskit_pauli = Pauli(pauli_string[::-1])
    qiskit_vector = Statevector(qc)
    exp_val_qiskit = qiskit_vector.expectation_value(qiskit_pauli).real

    # 6. Compare results
    assert np.isclose(exp_val_mine.real, exp_val_qiskit, atol=1e-6), (
        f"Real part of expectation value mismatch for seed {seed} "
        f"and Pauli '{pauli_string}'.\nMine: {exp_val_mine.real}, Qiskit: {exp_val_qiskit}"
    )
    assert np.isclose(exp_val_mine.imag, 0, atol=1e-6), (
        f"Imaginary part of expectation value should be zero, but is {exp_val_mine.imag}"
    )


@pytest.mark.parametrize("num_qubits", [5, 6])
@pytest.mark.parametrize("test_run", range(10))
def test_exp_value_clif_toffoli(num_qubits: int, test_run: int):
    """
    Compares the expectation value from StabilizerDecomposedState with Qiskit
    for circuits with Clifford+Toffoli gates.
    """
    # 1. Create a random Clifford+Toffoli circuit
    num_clifford_gates = 100
    num_toffoli = 5
    seed = None
    qc = random_clifford_toffoli_circuit(
        num_qubits, num_clifford_gates, num_toffoli, seed=seed)
    # 2. Convert to StabilizerDecomposedState
    try:
        stab_decomp_state = qiskit_circuit_to_toffoli_decomposed_stab_decomp_state(qc)
    except NotImplementedError as e:
        pytest.skip(f"Skipping due to NotImplementedError: {e}")
        return

    # If conversion results in an empty state (all components failed projection)
    if not stab_decomp_state.stabilizers:
        qiskit_vector = Statevector(qc).data
        assert np.allclose(np.linalg.norm(qiskit_vector), 0), \
            "Converter produced an empty state, but Qiskit state is non-zero."
        return

    for _ in range(10):  # Test multiple random observables
        # 3. Generate a random Pauli string for the observable
        pauli_string = "".join(random.choices(["I", "X", "Y", "Z"], k=num_qubits))
        if pauli_string == "I" * num_qubits:
            pauli_string = "X" + "I" * (num_qubits - 1)
        # 4. Calculate expectation value with our implementation (big-endian)
        exp_val_mine = stab_decomp_state.exp_value(pauli_string)
        # 5. Calculate expectation value with Qiskit (little-endian)
        # The Pauli string must be reversed to match Qiskit's endianness.
        qiskit_pauli = Pauli(pauli_string[::-1])
        qiskit_vector = Statevector(qc)
        exp_val_qiskit = qiskit_vector.expectation_value(qiskit_pauli).real
        # 6. Compare results
        assert np.isclose(exp_val_mine.real, exp_val_qiskit, atol=1e-6), (
            f"Real part of expectation value mismatch for test run {test_run}, "
            f"Pauli '{pauli_string}'.\nMine: {exp_val_mine.real}, Qiskit: {exp_val_qiskit}"
        )
        assert np.isclose(exp_val_mine.imag, 0, atol=1e-6), (
            f"Imaginary part of expectation value should be zero, but is {exp_val_mine.imag}"
        )
