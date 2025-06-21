import random

from src.stabilizer_state_ch_form import StabilizerStateChForm
from src.utils import apply_random_cliffords, are_equivalent_states


def test_project_and_remove():
    for _ in range(100):
        n = 6
        state = StabilizerStateChForm(num_qubits=n)
        apply_random_cliffords(state, num_ops=100, seed=None)
        target = random.randint(0, n - 1)
        try:
            state.project_0(target)
            copy = state.copy()
            removed = copy.remove_qubit(target)
            recovered = removed.kron(StabilizerStateChForm(0))
            assert are_equivalent_states(state, recovered), (
                f"Failed to recover state after project and remove on qubit {target}."
            )
        except ValueError:
            pass
