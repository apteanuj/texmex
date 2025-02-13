# This code is part of Mthree.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Independent bit-array generator"""
import numpy as np


class IndependentGenerator:
    """Independent bit-array generator"""

    def __init__(self, num_qubits):
        """Generator of arrays corresponding to a single x-gate on
        one qubit at a time.

        This is primarily for use with simulators.

        Parameters:
            num_qubits (int): Number of qubits

        Attributes:
            num_qubits (int): Number of qubits / length of arrays
            length (int): Total number of generated arrays, default=None (infinite)
            seed (int): Seed used for RNG
        """
        self.name = "independent"
        self.num_qubits = num_qubits
        self.length = num_qubits
        self._iter_index = 0

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < self.length:
            out = np.zeros(self.num_qubits, dtype=np.uint8)
            out[self.num_qubits - self._iter_index - 1] = 1
            self._iter_index += 1
            return out
        else:
            raise StopIteration
