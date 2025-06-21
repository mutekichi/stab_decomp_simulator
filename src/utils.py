import random
from typing import Optional

import numpy as np

from .stabilizer_state_ch_form import StabilizerStateChForm


def are_equivalent_states(
    state1: StabilizerStateChForm,
    state2: StabilizerStateChForm
) -> bool:
    return np.isclose(np.abs(state1.inner(state2)), 1.0, atol=1e-8)


def apply_random_cliffords(
    state: StabilizerStateChForm,
    num_ops: int = 1000,
    qubit_selector: Optional[callable] = None,  # type: ignore
    seed: Optional[int] = None,  # seed パラメータを追加
) -> None:
    """`state` に対してランダムな Clifford 操作を `num_ops` 回適用します。

    - H は exponent=1 で適用
    - X, Y, Z は exponent を 0.5, 1, 1.5 のいずれかからランダムに選択
    - CX, CZ は exponent を 1 または 3 のいずれかからランダムに選択（mod 2 で 1 になるように）

    Args:
        state: cirq.StabilizerStateChForm のインスタンス
        num_ops: 適用する操作の総数
        qubit_selector: optional なコールバック関数で、`state.n` を引数に取り
                        `0 <= q < state.n` の整数を返すようにすると、
                        qubit の選択をカスタマイズできます。
                        省略時は `random.randrange` を用います。
        seed: ランダムジェネレータのシード値。None の場合はシードを設定しません。
    """
    if seed is not None:
        random.seed(seed)  # ★ シードを設定

    n = state.n
    choose_qubit = qubit_selector or (lambda num_qubits_in_state: random.randrange(num_qubits_in_state))

    for _ in range(num_ops):
        gate = random.choice(['H', 'X', 'Y', 'Z', 'CX', 'CZ'])
        # print(f"Op {i+1}/{num_ops}: Chosen gate: {gate}") # デバッグ用

        if gate in ('H', 'X', 'Y', 'Z'):
            q = choose_qubit(n)
            if gate == 'H':
                # Hadamard は exponent=1 のみ許容
                state.apply_h(q, exponent=1)
            elif gate == 'X':
                exp = random.choice([0.5, 1.0, 1.5])
                state.apply_x(q, exponent=exp)
            elif gate == 'Y':
                exp = random.choice([0.5, 1.0, 1.5])
                state.apply_y(q, exponent=exp)
            elif gate == 'Z':
                exp = random.choice([0.5, 1.0, 1.5])
                state.apply_z(q, exponent=exp)

        else:  # CX or CZ
            q0 = choose_qubit(n)
            q1 = choose_qubit(n)
            # 異なる qubit を選ぶ
            while q1 == q0:
                q1 = choose_qubit(n)
            exp = random.choice([1, 3])  # mod 2 で 1 になるように
            if gate == 'CX':
                state.apply_cx(q0, q1, exponent=exp)
            else:  # 'CZ'
                state.apply_cz(q0, q1, exponent=exp)
