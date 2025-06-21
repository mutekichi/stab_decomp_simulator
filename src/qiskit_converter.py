import numpy as np
from qiskit import QuantumCircuit

from .stabilizer_decomposed_state import StabilizerDecomposedState
from .stabilizer_state_ch_form import StabilizerStateChForm
from .t_tensor_state import construct_T_tensor_state


def _apply_gate_ch(state: StabilizerStateChForm, gate_name: str, qubits: list[int]):
    """Helper function to apply a gate to the CH-form stabilizer state."""
    if gate_name == "h":
        state.apply_h(qubits[0])
    elif gate_name == "s":
        state.apply_z(qubits[0], exponent=0.5)
    elif gate_name == "sdg":
        state.apply_z(qubits[0], exponent=-0.5)
    elif gate_name == "x":
        state.apply_x(qubits[0])
    elif gate_name == "y":
        state.apply_y(qubits[0])
    elif gate_name == "z":
        state.apply_z(qubits[0])
    elif gate_name == "cx":
        state.apply_cx(qubits[0], qubits[1])
    elif gate_name == "cz":
        state.apply_cz(qubits[0], qubits[1])
    elif gate_name == "swap":
        q1, q2 = qubits[0], qubits[1]
        state.apply_cx(q1, q2)
        state.apply_cx(q2, q1)
        state.apply_cx(q1, q2)
    else:
        raise ValueError(f"Unsupported gate '{gate_name}' for CH form application.")


def qiskit_circuit_to_stab_decomp(qc: QuantumCircuit) -> StabilizerDecomposedState:
    """
    Given a Qiskit circuit C composed of only Clifford+T gates,
    returns the state |ψ⟩ = C|0⟩ as a stabilizer-decomposed state.
    """
    num_qubits_orig = qc.num_qubits
    num_t_gates = 0
    clifford_ops: list[tuple[str, list[int]]] = []

    gate_map_ch = {
        "h": "h", "s": "s", "sdg": "sdg", "x": "x", "y": "y",
        "z": "z", "cx": "cx", "cz": "cz", "swap": "swap",
    }

    # First pass to parse the circuit and separate T-gates from Cliffords
    for instruction in qc.data:
        instr = instruction.operation
        qregs = instruction.qubits
        qubits = [q._index for q in qregs]
        name = instr.name.lower()
        if name in ("t", "tdg"):
            ancilla_idx = num_qubits_orig + num_t_gates
            clifford_ops.append(("cx", [qubits[0], ancilla_idx]))
            if name == "tdg":
                clifford_ops.append(("sdg", [qubits[0]]))
            num_t_gates += 1
        elif name in gate_map_ch:
            clifford_ops.append((gate_map_ch[name], qubits))
        else:
            raise ValueError(f"Unsupported gate '{name}' in circuit.")

    # If there are no T-gates, the circuit is purely Clifford.
    if num_t_gates == 0:
        final_stab = StabilizerStateChForm(num_qubits_orig)
        for gate, qubits in clifford_ops:
            _apply_gate_ch(final_stab, gate, qubits)
        return StabilizerDecomposedState(num_qubits_orig, [final_stab], [1.0 + 0j])

    # For circuits with T-gates, construct the initial |T> state tensor product
    t_tensor_state = construct_T_tensor_state(num_t_gates)

    final_stabilizers = []
    final_coeffs = []

    # Process each stabilizer component of the |T> state
    for stab, coeff in zip(t_tensor_state.stabilizers, t_tensor_state.coeffs):
        # Create the combined system: |0...0> (data) ⊗ |stab> (ancillas)
        full_stab_state = StabilizerStateChForm(num_qubits_orig).kron(stab)

        # Apply the Clifford operations
        for gate, qubits in clifford_ops:
            _apply_gate_ch(full_stab_state, gate, qubits)

        # Attempt to post-select all ancilla qubits to |0>
        can_postselect_all = True
        prob_success = 1.0
        for i in range(num_t_gates - 1, -1, -1):
            ancilla_qubit = num_qubits_orig + i
            try:
                p0 = full_stab_state.project_0(ancilla_qubit)
                prob_success *= p0
            except ValueError:
                can_postselect_all = False
                break

        # If post-selection succeeded, save the resulting state
        if can_postselect_all:
            final_stab = full_stab_state
            for i in range(num_t_gates - 1, -1, -1):
                final_stab = final_stab.remove_qubit(num_qubits_orig + i)

            final_stabilizers.append(final_stab)
            new_coeff = coeff * np.sqrt(prob_success * (2**num_t_gates))
            final_coeffs.append(new_coeff)

    # Create the final decomposed state
    final_state = StabilizerDecomposedState(num_qubits_orig, final_stabilizers, final_coeffs)

    # Normalize the final state vector as some components may have been discarded
    norm = np.sqrt(final_state.inner(final_state).real)
    if norm > 1e-9:
        final_state.coeffs = [c / norm for c in final_state.coeffs]

    return final_state
