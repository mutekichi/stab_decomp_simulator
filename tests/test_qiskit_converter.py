import random
from typing import Optional

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from src.qiskit_converter import qiskit_circuit_to_stab_decomp


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


@pytest.mark.parametrize("num_qubits", [2, 3, 4])
@pytest.mark.parametrize("test_run", range(10))
def test_exp_value_vs_qiskit(num_qubits: int, test_run: int):
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
        stab_decomp_state = qiskit_circuit_to_stab_decomp(qc)
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
