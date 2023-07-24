import threading
import warnings
import datetime
from dateutil import tz
import numpy as np

from qiskit.result import marginal_counts

from mthree.exceptions import M3Error
from mthree.utils import expval, final_measurement_mapping

def vals_from_dict(dict):
    """
    Obtain the values stored in a dictionary 

    Parameters:
        dict (dictionary): Input dictionary

    Returns:
        list: List containing values from dictionary
    """
    values = []
    for key,value in dict.items():
        values.append(value)
    return values


def undo_twirling(dict_list, generator):
    """
    Undo the effect of twirling from the array_list to the converted form of the dictionaries in the dict_list.

    Parameters:
        dict_list (list): List of dictionaries.
        generator (list): List of binary strings from the generator

    Returns:
        list: List of dictionaries with untwirled keys.
    Raises Error if the length of dict_list does not match length of generator
    """
    if len(dict_list)!=len(list(generator)):
        raise ValueError('Length of dict_list must match length of generator')
    result = []
    for dictionary, array in zip(dict_list, generator):
        undone_dict = {}
        for key, value in dictionary.items():
            array_int = [int(bit) for bit in array]
            undone_key_as_list = [(array_int[i] + int(bit)) % 2 for i, bit in enumerate(key)]
            undone_key = "".join([str(x) for x in undone_key_as_list])
            undone_dict[undone_key] = value
        result.append(undone_dict)
    return result

def merge_dictionaries(dict_list):
    """
    Merge dictionaries in the given list into a single dictionary,
    adding values based on the keys.

    Parameters:
        dict_list (list): List of dictionaries.

    Returns:
        dict: Merged dictionary with added values based on keys.
    """
    merged_dict = {key: sum(d.get(key, 0) for d in dict_list) for d in dict_list for key in d}
    return merged_dict

def undo_twirling_and_merge(dict_list, generator):
    """
    Undo the effect of twirling from the generator for the count dictionaries in the dict_list and merge
    the resulting list of dictionaries

    Parameters:
        dict_list (list): List of dictionaries.
        generator (list): List of binary strings from the generator

    Returns:
        dict: Merged dictionary
    Raises Error if the length of dict_list does not match length of generator
    """
    return merge_dictionaries(undo_twirling(dict_list, generator))

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

def key_val_interchange(dict):
    return {val: key for key, val in dict.items()}

def marginalize_calibration_counts(final_mapping, calibration_mapping, counts):
    """
    Process the counts dictionary based on the measurement mapping dictionaries by marginalizing over non-relevant bits.

    Parameters:
        final_mapping (dict): Final measurement mapping dictionary.
        calibration_mapping (dict): Physical-to-bit mapping dictionary.
        counts (dict): Counts dictionary.

    Returns:
        dict: Processed counts dictionary.
    """
    final_mapping_interchanged = key_val_interchange(final_mapping)
    bit_to_physical_mapping = key_val_interchange(calibration_mapping)

    indices_to_keep = []
    for circ_logical_bit, circ_physical_bit in final_mapping.items():
        calib_logical_bit = calibration_mapping[circ_physical_bit]
        indices_to_keep.append(calib_logical_bit)

    shuffled_marginalized_counts = marginal_counts(result=counts, indices=indices_to_keep)

    # marginalization does not care about the ordering of bits so 
    # we must restore the appropriate bit-ordering 
    # marginalized_counts = marginal_counts(result=counts, indices=indices_to_keep)

    indices_to_shuffle = []
    for index in indices_to_keep:
        calib_physical_bit = bit_to_physical_mapping[index]
        circ_logical_bit = final_mapping_interchanged[calib_physical_bit]
        indices_to_shuffle.append(circ_logical_bit)

    # now we unshuffle the shuffle marginalized counts to get marginalized counts 
    marginalized_counts = {permute_bitstring(key, indices_to_shuffle):val for key,val in shuffled_marginalized_counts.items()}
    
    return marginalized_counts

def minimal_qubits_to_calib(qc_list):
    """
    Find the minimal set of qubits that need to be calibrated for given list of quantum circuits.

    Parameters:
        qc_list (list): list of transpiled circuit
    Returns:
        list: list of qubits which is the union of all the qubits that are measured in the transpiled circuits
    """
    qubits = set()
    for qc in qc_list:
        for key,value in final_measurement_mapping(qc).items():
            qubits.add(value)
    return list(qubits)

def sort_counts(dict):
    # sort dictionary of counts with the most common occurence in the beginning
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}

def convert_to_probabilities(counts):
    total_counts = sum(counts.values())
    probabilities = {bitstring: count / total_counts for bitstring, count in counts.items()}
    return probabilities

def total_variation_distance(probabilities1, probabilities2):
    all_keys = set(probabilities1.keys()).union(set(probabilities2.keys()))
    tv_distance = 0.0
    for bitstring in all_keys:
        prob1 = probabilities1.get(bitstring, 0.0)
        prob2 = probabilities2.get(bitstring, 0.0)
        tv_distance += abs(prob1 - prob2)
    return tv_distance / 2.0

def util_expval_and_stddev(counts, operator, mapping, calibration_counts, calibration_mapping):
    """
    Use the counts data to compute corrected expectation value of an operator for a given circuit 

    Parameters:
        counts (dict): dictionary of counts
        operator (str or dict or list): String or dict representation of diagonal 
                                        qubit operators used in computing the expectation
                                        value.
        mapping (dict): dictionary containing mapping of measured qubits for the given circuit
        calibration_counts (dict): dictionary of counts
        calibration_mapping (dict): dictionary containing mapping of measured qubits for calibration

    Returns:
        list: list of results for the corrected expectation value and uncertainity estimate
    """
    # This is needed because counts is a Counts object in Qiskit not a dict.
    counts = dict(counts)
    shots = sum(counts.values())

    # calibration data is stored in calibration_counts
    calibration_counts = dict(calibration_counts)
    calibration_shots = sum(calibration_counts.values())

    # marginalize calibration data 
    marginalized_calibration_counts = marginalize_calibration_counts(mapping, calibration_mapping, calibration_counts)

    # find the expectation value from the circuit data and the associated uncertainity
    uncorrected_expval  = expval(items=counts, exp_ops=operator)
    uncorrected_expval_uncertainity = np.sqrt(2*np.log(4/0.05)/shots)

    # find the expectation value from the calibration data and the associated uncertainity
    calibration_expval  = expval(items=marginalized_calibration_counts, exp_ops=operator)
    calibration_expval_uncertainity = np.sqrt(2*np.log(4/0.05)/calibration_shots)

    # divide by the calibration expectation value to obtain corrected expectation value 
    corrected_expval = uncorrected_expval/calibration_expval
    uncertainity_corrected_expval = (uncorrected_expval_uncertainity+calibration_expval_uncertainity)/(np.abs(calibration_expval)-calibration_expval_uncertainity)

    #result_dict = {'uncorrected_expval': uncorrected_expval, 'expval': corrected_expval, 'uncertainity': uncertainity_corrected_expval}

    return (corrected_expval, uncertainity_corrected_expval)

def util_expval(counts, operator, mapping, calibration_counts, calibration_mapping):
    """
    Use the counts data to compute corrected expectation value of an operator for a given circuit 

    Parameters:
        counts (dict): dictionary of counts
        operator (str or dict or list): String or dict representation of diagonal 
                                        qubit operators used in computing the expectation
                                        value.
        mapping (dict): dictionary containing mapping of measured qubits for the given circuit
        calibration_counts (dict): dictionary of counts
        calibration_mapping (dict): dictionary containing mapping of measured qubits for calibration

    Returns:
        float: corrected expectation value
    """
    # This is needed because counts is a Counts object in Qiskit not a dict.
    counts = dict(counts)

    # calibration data is stored in calibration_counts
    calibration_counts = dict(calibration_counts)

    # marginalize calibration data 
    marginalized_calibration_counts = marginalize_calibration_counts(mapping, calibration_mapping, calibration_counts)

    # find the expectation value from the circuit data and the associated uncertainity
    uncorrected_expval  = expval(items=counts, exp_ops=operator)

    # find the expectation value from the calibration data and the associated uncertainity
    calibration_expval  = expval(items=marginalized_calibration_counts, exp_ops=operator)

    # divide by the calibration expectation value to obtain corrected expectation value 
    corrected_expval = uncorrected_expval/calibration_expval

    #result_dict = {'uncorrected_expval': uncorrected_expval, 'expval': corrected_expval}

    return corrected_expval