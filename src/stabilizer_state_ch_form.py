# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from its original version in Google Cirq.
# Modifications include enhanced type hinting, removal of Cirq dependencies,
# and addition of an 'inner' method based on external logic.

from __future__ import annotations

import random
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


# Helper functions to remove dependency on Cirq
def big_endian_int_to_digits(
    integer: int, digit_count: int, base: int
) -> List[int]:
    """Converts a big-endian integer to a list of digits in a given base."""
    if integer < 0:
        raise ValueError("Integer must be non-negative.")
    if integer >= base**digit_count:
        raise ValueError("Integer is too large for the given digit count.")

    digits = []
    temp_integer = integer
    for _ in range(digit_count):
        digits.append(temp_integer % base)
        temp_integer //= base
    return digits[::-1]

def parse_random_state(
    seed: Optional[Union[int, np.random.RandomState]]
) -> np.random.RandomState:
    """Turn a seed into a np.random.RandomState instance."""
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)

def _phase(exponent: float, global_shift: float) -> complex:
    """Calculates the phase factor."""
    return np.exp(1j * np.pi * global_shift * exponent)


class StabilizerStateChForm:
    r"""A representation of stabilizer states using the CH form.

    The state is represented as:
        $|\psi> = \omega U_C U_H |s>$

    This representation keeps track of the overall phase and is based on
    the formalism described in https://arxiv.org/abs/1808.00128.

    This class is a standalone implementation with enhanced type hinting,
    derived from the original implementation in Google's Cirq library.
    """

    def __init__(self, num_qubits: int, initial_state: int = 0) -> None:
        """Initializes the StabilizerStateChForm.

        Args:
            num_qubits: The number of qubits in the system.
            initial_state: The computational basis state as a big-endian
                integer, which the stabilizer state is initialized to.
        """
        if num_qubits < 0:
            raise ValueError("Number of qubits must be non-negative.")
        self.n: int = num_qubits

        # The state is represented by a set of binary matrices and vectors,
        # as described in Section IVa of Bravyi et al.
        self.G: np.ndarray = np.eye(self.n, dtype=bool)
        self.F: np.ndarray = np.eye(self.n, dtype=bool)
        self.M: np.ndarray = np.zeros((self.n, self.n), dtype=bool)
        self.gamma: np.ndarray = np.zeros(self.n, dtype=int)

        self.v: np.ndarray = np.zeros(self.n, dtype=bool)
        self.s: np.ndarray = np.zeros(self.n, dtype=bool)

        self.omega: complex = 1.0

        # Apply X gates for every non-zero bit in the initial_state
        if initial_state != 0:
            initial_bits = big_endian_int_to_digits(
                initial_state, digit_count=num_qubits, base=2
            )
            for i, bit in enumerate(initial_bits):
                if bit:
                    self.apply_x(i)

    def copy(self) -> StabilizerStateChForm:
        """Creates a deep copy of the StabilizerStateChForm object."""
        new_state = StabilizerStateChForm(self.n)
        new_state.G = self.G.copy()
        new_state.F = self.F.copy()
        new_state.M = self.M.copy()
        new_state.gamma = self.gamma.copy()
        new_state.v = self.v.copy()
        new_state.s = self.s.copy()
        new_state.omega = self.omega
        return new_state

    def __str__(self) -> str:
        """Returns the state vector string representation of the state."""
        return str(self.state_vector())

    def __repr__(self) -> str:
        """Returns a string representation of the StabilizerStateChForm."""
        return f"StabilizerStateChForm(num_qubits={self.n!r})"

    def __eq__(self, other: Any) -> bool:
        """Checks for equality with another StabilizerStateChForm."""
        if not isinstance(other, StabilizerStateChForm):
            return NotImplemented
        return (
            self.n == other.n
            and np.all(self.G == other.G)
            and np.all(self.F == other.F)
            and np.all(self.M == other.M)
            and np.all(self.gamma == other.gamma)
            and np.all(self.v == other.v)
            and np.all(self.s == other.s)
            and self.omega == other.omega
        )

    def inner_product_of_state_and_x(self, x: int) -> complex:
        """Returns the amplitude of the x-th element of the state vector.

        This calculates <x|psi>.

        Args:
            x: The computational basis state index as a big-endian integer.

        Returns:
            The complex amplitude of the state vector at index x.
        """
        y = np.array(big_endian_int_to_digits(x, digit_count=self.n, base=2), dtype=bool)

        mu = int(sum(y * self.gamma))

        u = np.zeros(self.n, dtype=bool)
        for p in range(self.n):
            if y[p]:
                u ^= self.F[p, :]
                mu += 2 * (sum(self.M[p, :] & u) % 2)

        is_zero = not np.all(self.v | (u == self.s))

        return (
            self.omega
            * (2 ** (-sum(self.v) / 2))
            * (1j**mu)
            * ((-1) ** sum(self.v & u & self.s))
            * (not is_zero)
        )

    def state_vector(self) -> np.ndarray:
        """Computes the full state vector (wave function)."""
        if self.n > 16:
            # This is a reasonable limit to prevent excessive memory usage.
            raise MemoryError(
                f"Cannot generate state vector for {self.n} qubits. "
                "The vector would have 2**{self.n} elements."
            )
        size = 2**self.n
        wf = np.zeros(size, dtype=np.complex128)
        for x in range(size):
            wf[x] = self.inner_product_of_state_and_x(x)
        return wf

    def _S_right(self, q: int) -> None:
        """Right-multiplication of an S gate on qubit q."""
        self.M[:, q] ^= self.F[:, q]
        self.gamma[:] = (self.gamma[:] - self.F[:, q]) % 4

    def _CZ_right(self, q: int, r: int) -> None:
        """Right-multiplication of a CZ gate on qubits q and r."""
        self.M[:, q] ^= self.F[:, r]
        self.M[:, r] ^= self.F[:, q]
        self.gamma[:] = (self.gamma[:] + 2 * self.F[:, q] * self.F[:, r]) % 4

    def _CNOT_right(self, q: int, r: int) -> None:
        """Right-multiplication of a CNOT gate on control q and target r."""
        self.G[:, q] ^= self.G[:, r]
        self.F[:, r] ^= self.F[:, q]
        self.M[:, q] ^= self.M[:, r]

    def update_sum(self, t: np.ndarray, u: np.ndarray, delta: int = 0, alpha: int = 0) -> None:
        """Implements the transformation (Proposition 4 in Bravyi et al).

        i^alpha U_H (|t> + i^delta |u>) = omega W_C W_H |s'>
        """
        if np.all(t == u):
            self.s = t
            self.omega *= (1 / np.sqrt(2)) * ((-1) ** alpha) * (1 + 1j**delta)
            return

        set0 = np.where((~self.v) & (t ^ u))[0]
        set1 = np.where(self.v & (t ^ u))[0]

        q: int
        if len(set0) > 0:
            q = set0[0]
            for i in set0:
                if i != q:
                    self._CNOT_right(q, i)
            for i in set1:
                self._CZ_right(q, i)
        elif len(set1) > 0:
            q = set1[0]
            for i in set1:
                if i != q:
                    self._CNOT_right(i, q)
        else: # t == u, handled at the start
            return

        e = np.zeros(self.n, dtype=bool)
        e[q] = True

        y: np.ndarray
        z: np.ndarray
        if t[q]:
            y = u ^ e
            z = u
        else:
            y = t
            z = t ^ e

        omega, a, b, c = self._H_decompose(bool(self.v[q]), bool(y[q]), bool(z[q]), delta)

        self.s = y
        self.s[q] = c
        self.omega *= ((-1)**alpha) * omega

        if a:
            self._S_right(q)
        self.v[q] = b


    def _H_decompose(self, v: bool, y: bool, z: bool, delta: int) -> Tuple[complex, bool, bool, bool]:
        """Decomposes H^v (|y> + i^delta |z>) = omega S^a H^b |c> for a single qubit.

        Args:
            v, y, z: Boolean inputs for the single qubit state.
            delta: Integer (mod 4) for the phase.

        Returns:
            A tuple (omega, a, b, c) where omega is a complex number, and a, b, c are booleans.

        Raises:
            ValueError: if y == z.
        """
        if y == z:
            raise ValueError("Input states |y> and |z> cannot be the same.")

        omega: complex
        a: bool
        b: bool
        c: bool

        if not v:
            omega = (1j) ** (delta * int(y))
            delta2 = ((-1) ** y * delta) % 4
            c = bool(delta2 >> 1)
            a = bool(delta2 & 1)
            b = True
        else:
            if not (delta & 1): # delta is even
                a = False
                b = False
                c = bool(delta >> 1)
                omega = (-1) ** (c & y)
            else: # delta is odd
                omega = (1 / np.sqrt(2)) * (1 + 1j**delta)
                b = True
                a = True
                c = not ((delta >> 1) ^ y)

        return omega, a, b, c

    def _measure(self, q: int, prng: np.random.RandomState) -> int:
        """Measures the q-th qubit.

        This is a helper function for the public `measure` method.
        It simulates a single qubit measurement and projects the state.
        See Section 4.1 "Simulating measurements" in the reference paper.

        Args:
            q: The qubit index to measure.
            prng: A NumPy random number generator.

        Returns:
            The measurement outcome (0 or 1).
        """
        w = self.s.copy()
        for i, v_i in enumerate(self.v):
            if v_i:
                w[i] = bool(prng.randint(2))

        measurement_outcome = int(sum(w & self.G[q, :]) % 2)

        # Project the state to the measurement outcome.
        self.project_Z(q, measurement_outcome)
        return measurement_outcome

    def project_Z(self, q: int, z: int) -> None:
        """Applies a Z projector on the q-th qubit.

        Updates the state to be a normalized state where Z_q |psi> = z |psi>.

        Args:
            q: The qubit index.
            z: The measurement outcome (0 or 1).
        """
        t = self.s.copy()
        u = (self.G[q, :] & self.v) ^ self.s
        delta = (2 * sum((self.G[q, :] & (~self.v)) & self.s) + 2 * z) % 4

        if np.all(t == u):
            self.omega /= np.sqrt(2)

        self.update_sum(t, u, delta=delta)

    def kron(self, other: StabilizerStateChForm) -> StabilizerStateChForm:
        """Computes the tensor product of this state with another."""
        n_total = self.n + other.n
        new_state = StabilizerStateChForm(n_total)

        new_state.G[:self.n, :self.n] = self.G
        new_state.G[self.n:, self.n:] = other.G
        new_state.F[:self.n, :self.n] = self.F
        new_state.F[self.n:, self.n:] = other.F
        new_state.M[:self.n, :self.n] = self.M
        new_state.M[self.n:, self.n:] = other.M

        new_state.gamma = np.concatenate([self.gamma, other.gamma])
        new_state.v = np.concatenate([self.v, other.v])
        new_state.s = np.concatenate([self.s, other.s])

        new_state.omega = self.omega * other.omega
        return new_state

    def reindex(self, axes: Sequence[int]) -> StabilizerStateChForm:
        """Permutes the qubits."""
        if len(axes) != self.n:
            raise ValueError(f"Number of axes {len(axes)} must equal number of qubits {self.n}.")

        new_state = StabilizerStateChForm(self.n)
        new_state.G = self.G[np.ix_(axes, axes)]
        new_state.F = self.F[np.ix_(axes, axes)]
        new_state.M = self.M[np.ix_(axes, axes)]
        new_state.gamma = self.gamma[axes]
        new_state.v = self.v[axes]
        new_state.s = self.s[axes]
        new_state.omega = self.omega
        return new_state

    def apply_x(self, axis: int, exponent: float = 1.0, global_shift: float = 0.0) -> None:
        """Applies an X gate to a specific qubit."""
        if exponent % 0.5 != 0.0:
            raise ValueError("X gate exponent must be a half-integer.")

        if exponent % 2 != 0:
            self.apply_h(axis)
            self.apply_z(axis, exponent)
            self.apply_h(axis)
        self.omega *= _phase(exponent, global_shift)

    def apply_y(self, axis: int, exponent: float = 1.0, global_shift: float = 0.0) -> None:
        """Applies a Y gate to a specific qubit."""
        if exponent % 0.5 != 0.0:
            raise ValueError("Y gate exponent must be a half-integer.")

        shift = _phase(exponent, global_shift)
        eff_exp = exponent % 2

        if eff_exp == 0:
            self.omega *= shift
        elif eff_exp == 0.5:
            self.apply_z(axis)
            self.apply_h(axis)
            self.omega *= shift * (1 + 1j) / np.sqrt(2)
        elif eff_exp == 1.0:
            self.apply_z(axis)
            self.apply_h(axis)
            self.apply_z(axis)
            self.apply_h(axis)
            self.omega *= shift * 1j
        elif eff_exp == 1.5:
            self.apply_h(axis)
            self.apply_z(axis)
            self.omega *= shift * (1 - 1j) / np.sqrt(2)

    def apply_z(self, axis: int, exponent: float = 1.0, global_shift: float = 0.0) -> None:
        """Applies a Z gate to a specific qubit."""
        if exponent % 0.5 != 0.0:
            raise ValueError("Z gate exponent must be a half-integer.")

        if exponent % 2 != 0:
            # S gate is Z**0.5. Apply S for each 0.5 in exponent.
            effective_exponent = exponent % 2
            for _ in range(int(effective_exponent * 2)):
                # Left-multiplication of S gate
                # (See end of Proposition 4 in the reference paper)
                self.M[axis, :] ^= self.G[axis, :]
                self.gamma[axis] = (self.gamma[axis] - 1) % 4
        self.omega *= _phase(exponent, global_shift)

    def apply_h(self, axis: int, exponent: float = 1.0, global_shift: float = 0.0) -> None:
        """Applies an H gate to a specific qubit."""
        if exponent % 1 != 0:
            raise ValueError("H gate exponent must be an integer.")

        if exponent % 2 != 0:
            # Left-multiplication of H gate
            # (See Equations 48, 49 and Proposition 4 in the reference paper)
            t = self.s ^ (self.G[axis, :] & self.v)
            u = self.s ^ (self.F[axis, :] & (~self.v)) ^ (self.M[axis, :] & self.v)
            alpha = sum(self.G[axis, :] & (~self.v) & self.s) % 2

            beta = sum(self.M[axis, :] & (~self.v) & self.s)
            beta += sum(self.F[axis, :] & self.v & self.M[axis, :])
            beta += sum(self.F[axis, :] & self.v & self.s)
            beta %= 2

            delta = (self.gamma[axis] + 2 * (alpha + beta)) % 4
            self.update_sum(t, u, delta=delta, alpha=alpha)
        self.omega *= _phase(exponent, global_shift)

    def apply_cz(
        self, control_axis: int, target_axis: int, exponent: float = 1.0, global_shift: float = 0.0
    ) -> None:
        """Applies a CZ gate."""
        if exponent % 1 != 0:
            raise ValueError("CZ exponent must be an integer.")

        if exponent % 2 != 0:
            # Left-multiplication of CZ gate
            self.M[control_axis, :] ^= self.G[target_axis, :]
            self.M[target_axis, :] ^= self.G[control_axis, :]
        self.omega *= _phase(exponent, global_shift)

    def apply_cx(
        self, control_axis: int, target_axis: int, exponent: float = 1.0, global_shift: float = 0.0
    ) -> None:
        """Applies a CNOT gate."""
        if exponent % 1 != 0:
            raise ValueError("CX gate exponent must be an integer.")

        if exponent % 2 != 0:
            # Left-multiplication of CX gate
            # (See end of Proposition 4 in the reference paper)
            term1 = self.gamma[control_axis]
            term2 = self.gamma[target_axis]
            term3 = 2 * (sum(self.M[control_axis, :] & self.F[target_axis, :]) % 2)
            self.gamma[control_axis] = (term1 + term2 + term3) % 4

            self.G[target_axis, :] ^= self.G[control_axis, :]
            self.F[control_axis, :] ^= self.F[target_axis, :]
            self.M[control_axis, :] ^= self.M[target_axis, :]
        self.omega *= _phase(exponent, global_shift)

    def apply_global_phase(self, coefficient: complex) -> None:
        """Applies a global phase to the state."""
        self.omega *= coefficient

    def measure(
        self, axes: Sequence[int], seed: Optional[Union[int, np.random.RandomState]] = None
    ) -> List[int]:
        """Measures the given qubits in the computational basis.

        Args:
            axes: The sequence of qubit indices to measure.
            seed: A seed for the random number generator or a RandomState object.

        Returns:
            A list of measurement outcomes (0 or 1) for each measured qubit.
        """
        prng = parse_random_state(seed)
        return [self._measure(axis, prng) for axis in axes]

    def phase(self, exponent: float, global_shift: float = 0.0) -> complex:
        """A public-facing method to calculate a phase factor.

        This is a wrapper around the internal _phase helper function.
        """
        return _phase(exponent, global_shift)

    # --- NEWLY ADDED METHOD AND HELPERS ---

    def inner(self, other: StabilizerStateChForm) -> complex:
        """Calculates the inner product <self|other> with another stabilizer state.

        This method works by finding a sequence of Clifford operations that
        transforms the state |self> into the |0...0> state, and then applying
        the same sequence of operations to |other>. The inner product is then
        derived from the resulting state's amplitude at the |0...0> basis state.

        Args:
            other: The other StabilizerStateChForm object.

        Returns:
            The complex value of the inner product <self|other>.

        Raises:
            ValueError: If the number of qubits in the two states do not match.
        """
        if self.n != other.n:
            raise ValueError("Cannot compute inner product between states with different numbers of qubits.")

        # 1. Find the sequence of operations to transform |self> to |0...0>
        ops = self._get_normalize_to_zero_ops()

        # 2. Apply these operations to a copy of |self> to get the final phase
        phi1_norm = self.copy()
        self._apply_ops_to_state(phi1_norm, ops)
        omega1 = phi1_norm.omega

        # 3. Apply the same operations to a copy of |other>
        phi2_prime = other.copy()
        self._apply_ops_to_state(phi2_prime, ops)

        # 4. The inner product <0|U|other> is the 0-th amplitude of the new state
        amplitude = phi2_prime.inner_product_of_state_and_x(0)

        # 5. The full inner product is <self|other> = <0|U_dag U|other> = omega1* <0|U|other>
        return np.conjugate(omega1) * amplitude

    def _get_basis_normalization_ops(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Generates operations to transform the state to (G,F)=I, M=0."""
        st = self.copy()
        ops: List[Tuple[str, Tuple[int, ...]]] = []
        n = st.n

        # --- Part 1: Convert G to identity matrix using left CNOTs ---
        for j in range(n):
            if not st.G[j, j]:
                # Find a pivot
                try:
                    k = next(idx for idx in range(j + 1, n) if st.G[idx, j])
                except StopIteration:
                    raise RuntimeError("Failed to find pivot; G matrix is singular.")
                # Swap rows j and k using CNOTs
                for ctrl, tgt in [(k, j), (j, k), (k, j)]:
                    st.apply_cx(ctrl, tgt)
                    ops.append(('cnot', (ctrl, tgt)))
            # Elimination step
            for i in range(n):
                if i != j and st.G[i, j]:
                    st.apply_cx(j, i)
                    ops.append(('cnot', (j, i)))

        assert np.array_equal(st.G, np.eye(n, dtype=bool)), "G matrix is not identity after CNOTs."
        assert np.array_equal(st.F, np.eye(n, dtype=bool)), "F matrix is not identity after CNOTs."

        # --- Part 2: Convert off-diagonal M to zero using right CZs ---
        for r in range(n):
            for c in range(r + 1, n):
                if st.M[r, c]:
                    st._CZ_right(r, c)
                    ops.append(('cz', (r, c)))
                    assert not st.M[r, c] and not st.M[c, r], "M matrix is not zero after CZs."

        # --- Part 3: Convert diagonal M to zero using right Ss ---
        for q in range(n):
            if st.M[q, q]:
                st._S_right(q)
                ops.append(('s', (q,)))
                assert not st.M[q, q], "M matrix is not zero after S gates."
        assert not st.M.any(), "M matrix is not zero after all operations."

        return ops

    def _get_normalize_to_zero_ops(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Generates operations to transform the state to omega|0...0>."""
        st = self.copy()

        # First, get ops to normalize the basis to (G,F)=I, M=0
        ops = st._get_basis_normalization_ops()
        self._apply_ops_to_state(st, ops)

        # Now, st is in the form omega' U_H |s'>
        # We need to eliminate v and s

        for i, vi in enumerate(st.v):
            if vi:
                st.apply_h(i)
                ops.append(('h', (i,)))
                assert not st.v[i], f"v[{i}] should be zero after H gate."

        for i, si in enumerate(st.s):
            if si:
                st.apply_x(i)
                ops.append(('x', (i,)))

        assert not st.v.any() and not st.s.any()

        return ops

    @staticmethod
    def _apply_ops_to_state(
        state: StabilizerStateChForm,
        ops: List[Tuple[str, Tuple[int, ...]]]
    ) -> None:
        """Applies a list of abstract operations to a state."""
        for op_name, qubits in ops:
            if op_name == 'cnot':
                state.apply_cx(qubits[0], qubits[1])
            elif op_name == 'h':
                state.apply_h(qubits[0])
            elif op_name == 'x':
                state.apply_x(qubits[0])
            elif op_name == 's':
                state.apply_z(qubits[0], exponent=0.5)
            elif op_name == 'cz':
                state.apply_cz(qubits[0], qubits[1])
            else:
                raise ValueError(f"Unsupported internal operation: {op_name}")
