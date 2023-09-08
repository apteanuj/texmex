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
"""
Calibration utility functions
-----------------------------

.. autosummary::
   :toctree: ../stubs/

   m3_legacy_cals

"""
import numpy as np
from qiskit.result import marginal_counts
from mthree.utils import expval


def m3_legacy_cals(m3_cals, calibration):
    """Convert new M3 calibration (flat array of diagonals)
    to the old list format of full matrices

    Parameters:
        m3_cals (ndarray): Array of single-qubit A-matrices (diagonals only)
        calibration (Calibration): A Calibration object

    Returns:
        list: Single-qubit A-matrices where index labels the physical qubit
    """
    num_qubits = calibration.backend_info['num_qubits']
    out_cals = [None] * num_qubits
    for bit, qubit in calibration.bit_to_physical_mapping.items():
        A = np.zeros((2, 2), dtype=float)
        A[0, 0] = m3_cals[2*bit]
        A[1, 1] = m3_cals[2*bit+1]
        A[1, 0] = 1 - A[0, 0]
        A[0, 1] = 1 - A[1, 1]
        out_cals[qubit] = A

    return out_cals

def permute_bitstring(bitstring, shuffle_pattern):
    """
    Permute a bit-string based on the given shuffle pattern.

    Parameters:
        bitstring (str): The input bit-string to be permuted.
        shuffle_pattern (list): The shuffle pattern specifying the new order of bits.

    Returns:
        str: Permuted bit-string.
    """
    permuted_bitstring = ""

    for index in shuffle_pattern:
        permuted_bitstring += bitstring[index]

    return permuted_bitstring

def list_to_dict(input_list):
    result_dict = {}
    for idx, value in enumerate(input_list):
        result_dict[idx] = value
    return result_dict

def marginalize_counts(counts, qubits, bit_to_physical_mapping):
    final_measurement_mapping = list_to_dict(qubits)
    final_mapping_interchanged = {v: k for k, v in final_measurement_mapping.items()}
    physical_to_bit_mapping = {v: k for k, v in bit_to_physical_mapping.items()}
    
    indices_to_keep = [physical_to_bit_mapping[qubit] for qubit in qubits]

    shuffled_marginalized_counts = marginal_counts(result=counts, indices=indices_to_keep)
    
    indices_to_shuffle = [final_mapping_interchanged[bit_to_physical_mapping[index]] for index in indices_to_keep]
    
    marginalized_counts = {permute_bitstring(key, indices_to_shuffle): val for key, val in shuffled_marginalized_counts.items()}
    
    return marginalized_counts

def rel_variance_from_expval(x, shots):
    """
    Relative Variance of a quantity estimated by summing up {-1,+1} for each shot based on the 
    observed bistring 

    Parameters:
        x (float): Input Expectation value 
        shots (int): Total Number of shots

    Returns:
        float: variance 
    """
    return (1-x**2)/(shots * x**2)


def mitig_expval_std(counts, qubits, operator, calibration_counts, bit_to_physical_mapping):
    """
    Use the counts data to compute mitigated expectation value of an operator for a given circuit 

    Parameters:
        counts (dict): dictionary of counts
        qubits (list): list of qubits measured at the end of the circuit
        operator (str or dict or list): String or dict representation of diagonal 
                        qubit operators used in computing the expectation
                        value.
        calibration_counts (dict): dictionary of counts
        bit_to_physical_mapping (dict): dictionary containing mapping of measured qubits for calibration

    Returns:
        list: list of results for the mitigated expectation value and uncertainity estimate
    """
    # This is needed because counts is a Counts object in Qiskit not a dict.
    counts = dict(counts)
    calibration_counts = dict(calibration_counts)

    # Find number of shots for circuit and calibration
    shots = sum(counts.values())
    calib_shots = sum(calibration_counts.values())

    # marginalize calibration data 
    marginalized_calibration_counts = marginalize_counts(calibration_counts, qubits, bit_to_physical_mapping)

    # find the expectation value from the circuit data and the associated uncertainity
    expvalue  = expval(items=counts, exp_ops=operator)
    #expvalue_rel_variance = rel_variance_from_expval(expvalue, shots)

    # find the expectation value from the calibration data and the associated uncertainity
    calib_expval  = expval(items=marginalized_calibration_counts, exp_ops=operator)
    #calib_rel_variance = rel_variance_from_expval(calib_expval, calib_shots)

    # divide by the calibration expectation value to obtain mitigated expectation value 
    mitigated_expval = expvalue/calib_expval
    #mitigated_rel_variance = expvalue_rel_variance + calib_rel_variance
    #mitigated_std = mitigated_expval*np.sqrt(mitigated_rel_variance)
    mitigated_std = (1/calib_expval)*np.sqrt(1/shots + 1/calib_shots)

    return (mitigated_expval, mitigated_std)

def mitig_expval(counts, qubits, operator, calibration_counts, bit_to_physical_mapping):
    """
    Use the counts data to compute mitigated expectation value of an operator for a given circuit 

    Parameters:
        counts (dict): dictionary of counts
        qubits (list): list of qubits measured at the end of the circuit
        operator (str or dict or list): String or dict representation of diagonal 
                        qubit operators used in computing the expectation
                        value.
        calibration_counts (dict): dictionary of counts
        bit_to_physical_mapping (dict): dictionary containing mapping of measured qubits for calibration

    Returns:
        float: result for the mitigated expectation value
    """
    # This is needed because counts is a Counts object in Qiskit not a dict.
    counts = dict(counts)
    calibration_counts = dict(calibration_counts)

    # Find number of shots for circuit and calibration
    shots = sum(counts.values())
    calib_shots = sum(calibration_counts.values())

    # marginalize calibration data 
    marginalized_calibration_counts = marginalize_counts(calibration_counts, qubits, bit_to_physical_mapping)

    # find the expectation value from the circuit data and the associated uncertainity
    expvalue  = expval(items=counts, exp_ops=operator)
    #expvalue_rel_variance = rel_variance_from_expval(expvalue, shots)

    # find the expectation value from the calibration data and the associated uncertainity
    calib_expval  = expval(items=marginalized_calibration_counts, exp_ops=operator)
    #calib_rel_variance = rel_variance_from_expval(calib_expval, calib_shots)

    # divide by the calibration expectation value to obtain mitigated expectation value 
    mitigated_expval = expvalue/calib_expval
    #mitigated_rel_variance = expvalue_rel_variance + calib_rel_variance
    #mitigated_std = mitigated_expval*np.sqrt(mitigated_rel_variance)
    mitigated_std = (1/calib_expval)*np.sqrt(1/shots + 1/calib_shots)

    return mitigated_expval