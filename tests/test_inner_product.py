import random

import numpy as np
from src.stab_decomp_simulator.stabilizer_state_ch_form import \
    StabilizerStateChForm


# --- Test functions ---
def test_inner_method_random(num_qubits: int = 4, num_tests: int = 50) -> None:
    """
    Tests the StabilizerStateChForm.inner method by comparing it to the
    naive state_vector method on random Clifford states.
    A smaller num_qubits is used by default due to state_vector memory constraints.
    """
    print(f"--- Running {num_tests} random tests for {num_qubits} qubits ---")

    def apply_random_cliffords(
        state: StabilizerStateChForm, num_ops: int = 20
    ) -> None:
        n = state.n

        def choose(n: int) -> int:
            return random.randrange(n)

        for _ in range(num_ops):
            gate = random.choice(['h', 'x', 'y', 'z', 's', 'cx', 'cz'])
            if gate == 'h':
                state.apply_h(choose(n))
            elif gate in ('x', 'y', 'z', 's'):
                exp = random.choice([0.5, 1.0, 1.5]) if gate != 's' else 1.0
                if gate == 's':
                    state.apply_z(choose(n), exponent=0.5)
                else:
                    getattr(state, f"apply_{gate}")(choose(n), exponent=exp)
            else:  # cx, cz
                q0, q1 = choose(n), choose(n)
                while q1 == q0:
                    q1 = choose(n)
                getattr(state, f"apply_{gate}")(q0, q1)

    for i in range(num_tests):
        state1 = StabilizerStateChForm(num_qubits=num_qubits)
        state2 = StabilizerStateChForm(num_qubits=num_qubits)
        apply_random_cliffords(state1)
        apply_random_cliffords(state2)

        # Naive comparison using full state vectors
        sv1 = state1.state_vector()
        sv2 = state2.state_vector()
        expected = np.vdot(sv1, sv2)

        # Calculation using the new inner method
        result = state1.inner(state2)

        assert np.allclose(expected, result), f"Mismatch in test {i}:\nExpected: {expected}\nResult:   {result}"
    print(f"--- All {num_tests} random tests passed. ---")
