import random

import numpy as np
from qiskit import QuantumCircuit
from src.qiskit_converter import qiskit_circuit_to_stab_decomp


# このテストファイル内で定義されていると仮定
def apply_decomposed_toffoli(qc: QuantumCircuit, control1: int, control2: int, target: int):
    qc.h(target)
    qc.cx(control2, target)
    qc.tdg(target)
    qc.cx(control1, target)
    qc.t(target)
    qc.cx(control2, target)
    qc.tdg(target)
    qc.cx(control1, target)
    qc.t(control2)
    qc.t(target)
    qc.h(target)
    qc.cx(control1, control2)
    qc.tdg(control2)
    qc.cx(control1, control2)
    qc.t(control1)
    qc.s(target)


def add_random_cliffords(qc: QuantumCircuit, qubit_indices: list[int], num_gates: int):
    """指定された量子ビットにランダムなCliffordゲートを追加する"""
    if len(qubit_indices) == 0:
        return
    clifford_gates_1q = ["h", "s", "x", "z"]
    clifford_gates_2q = ["cx", "cz"]
    for _ in range(num_gates):
        if len(qubit_indices) > 1 and random.random() < 0.5:
            q1, q2 = random.sample(qubit_indices, 2)
            getattr(qc, random.choice(clifford_gates_2q))(q1, q2)
        else:
            q = random.choice(qubit_indices)
            getattr(qc, random.choice(clifford_gates_1q))(q)


def test_complex_tensor_product_analytical_example():
    """
    複数の独立した部分系のテンソル積からなる、より複雑な回路を検証する。
    """
    NUM_QUBITS = 30
    # 理論期待値 = 0.5 (A) * 0.5 (B) * 1.0 (C) = 0.25
    THEORETICAL_VALUE = 0.25

    qc = QuantumCircuit(NUM_QUBITS)

    # --- サブシステムAの構築 (qubits 0, 1, 2) ---
    qc.h(0)
    qc.h(1)
    apply_decomposed_toffoli(qc, 0, 1, 2)
    # --- サブシステムBの構築 (qubits 3, 4, 5) ---
    qc.h(3)
    qc.h(4)
    apply_decomposed_toffoli(qc, 3, 4, 5)

    # --- サブシステムCの構築 (qubits 6 to 29) ---
    spectator_qubits = list(range(6, NUM_QUBITS))
    add_random_cliffords(qc, spectator_qubits, num_gates=100)

    # オブザーバブルの定義 P = (XXX)_A (ZZZ)_B (I...)_C
    pauli_list = ['I'] * NUM_QUBITS
    pauli_list[0:3] = ['X', 'X', 'X']
    pauli_list[3:6] = ['Z', 'Z', 'Z']
    pauli_string = "".join(pauli_list)

    print(f"\n--- 複雑なテンソル積状態の厳密比較テスト (N={NUM_QUBITS}) ---")
    print(f"オブザーバブル: {pauli_string}")
    print(f"理論期待値: {THEORETICAL_VALUE}")

    # シミュレータで全体の期待値を計算
    stab_decomp_state = qiskit_circuit_to_stab_decomp(qc)

    sim_value = stab_decomp_state.exp_value(pauli_string)
    print(f"シミュレータ計算値: {sim_value.real:.8f}")

    # 理論値とシミュレータの計算結果を比較
    assert np.isclose(sim_value.real, THEORETICAL_VALUE, atol=1e-7), \
        f"計算値が理論値と一致しません！ Got: {sim_value.real}, Expected: {THEORETICAL_VALUE}"

    print("✅ 検証成功！ 複雑な系のシミュレーション結果は理論値と厳密に一致しました。")
