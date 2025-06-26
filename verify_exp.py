# benchmarks/run_expectation_benchmark.py

import cProfile
import pstats
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, random_clifford
# 修正したコンバータをインポート
# from src.qiskit_converter import qiskit_circuit_to_stab_decomp
# 以下のコードは、上記インポートが正しく動作する前提で記述されています。
# ユーザーの環境に合わせて、実際のインポートパスに修正してください。
from src.qiskit_converter import qiskit_circuit_to_stab_decomp


def generate_clifford_t_layers_circuit(num_qubits: int, num_layers: int) -> QuantumCircuit:
    """
    各層がランダムなCliffordユニタリとそれに続くTゲートで構成される回路を生成する。
    Cliffordオブジェクトをcomposeすることで、回路は自動的に基本ゲートに展開される。
    """
    if num_qubits < 1:
        raise ValueError("Tゲートを適用するため、量子ビット数は1以上である必要があります。")

    main_circuit = QuantumCircuit(num_qubits)
    for _ in range(num_layers):
        clifford_op = random_clifford(num_qubits)
        clifford_circuit_layer = clifford_op.to_circuit()
        main_circuit.compose(clifford_circuit_layer, qubits=range(num_qubits), inplace=True)
        main_circuit.t(0)
    return main_circuit


def verify_expectation_values(num_qubits: int, num_layers: int, num_trials: int):
    """
    スタビライザー分解シミュレータの期待値計算をQiskitのStatevectorと比較検証する。
    各回路に対し、期待値が非ゼロになるオブザーバブルが見つかるまで試行する。
    """
    print(f"--- Verification Start: {num_qubits} qubits, {num_layers} T-gates, {num_trials} trials ---")

    for i in range(num_trials):
        trial_str = f"Trial {i+1}/{num_trials}:"
        print(f"{trial_str:<15}", end="")
        start_time = time.time()

        # 各Trialで一つ回路（状態）を生成
        circ = generate_clifford_t_layers_circuit(num_qubits, num_layers)

        try:
            # 状態ベクトルと分解状態は一度だけ計算
            state_vec = Statevector(circ)
            decomposed_state = qiskit_circuit_to_stab_decomp(circ)

            # 非ゼロの期待値が見つかるまでオブザーバブルを試行するループ
            MAX_PAULI_ATTEMPTS = 500  # 無限ループを避けるための上限
            non_zero_found = False
            for attempt in range(MAX_PAULI_ATTEMPTS):
                # ランダムな単一量子ビットPauli演算子をオブザーバブルとして生成
                pauli_list = ['I'] * num_qubits
                pauli_qubit = np.random.randint(0, num_qubits)
                pauli_char = np.random.choice(['X', 'Y', 'Z'])
                pauli_list[pauli_qubit] = pauli_char
                pauli_string = "".join(pauli_list)

                # まずQiskitで期待値を計算し、非ゼロか確認
                observable = SparsePauliOp(pauli_string)
                exp_qiskit = state_vec.expectation_value(observable).real

                # 期待値が非ゼロの場合のみ、詳細な検証に進む
                if not np.isclose(exp_qiskit, 0, atol=1e-7):
                    non_zero_found = True

                    # 自作シミュレータで期待値を計算
                    pauli_string_big_endian = pauli_string[::-1]
                    exp_mine = decomposed_state.exp_value(pauli_string_big_endian).real

                    # 結果の比較
                    if np.isclose(exp_qiskit, exp_mine, atol=1e-6):
                        print(f"OK (Found non-zero exp={exp_qiskit:.4f} for {pauli_string} at attempt {attempt+1}) ({time.time() - start_time:.2f}s)")
                    else:
                        print(f"\nMISMATCH! Qiskit: {exp_qiskit:.6f}, Mine: {exp_mine:.6f} for {pauli_string}")

                    break  # 非ゼロの期待値が見つかったので、このTrialは終了

            if not non_zero_found:
                print(f"WARNING: Could not find a non-zero expectation value after {MAX_PAULI_ATTEMPTS} attempts. ({time.time() - start_time:.2f}s)")

        except Exception as e:
            print(f"ERROR: {e}")

    print("--- Verification Finished ---")


def profile_decomposition(num_qubits: int, num_layers: int):
    """
    大規模な系に対して、シミュレータの性能をプロファイリングする。
    """
    print(f"\n--- Profiling Start: {num_qubits} qubits, {num_layers} T-gates ---")

    profiler = cProfile.Profile()

    circ = generate_clifford_t_layers_circuit(num_qubits, num_layers)
    pauli_string = 'Z' + 'I' * (num_qubits - 1)
    pauli_string_big_endian = pauli_string[::-1]

    profiler.enable()
    decomposed_state = qiskit_circuit_to_stab_decomp(circ)
    exp_decomposed = decomposed_state.exp_value(pauli_string_big_endian)
    profiler.disable()

    print(f"Decomposition and expectation calculation finished. Result: {exp_decomposed.real:.6f}")

    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.strip_dirs()
    stats.print_stats()
    print("--- Profiling Finished ---")


if __name__ == "__main__":
    # verify_expectation_values(num_qubits=8, num_layers=20, num_trials=10)

    # プロファイリング (20量子ビット、10Tゲート)
    # 必要に応じてコメントを解除して実行
    profile_decomposition(num_qubits=50, num_layers=10)
