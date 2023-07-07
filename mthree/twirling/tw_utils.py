import threading
import warnings
import datetime
from dateutil import tz
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.result import marginal_counts

from mthree.exceptions import M3Error
from mthree._helpers import system_info
from mthree.generators import HadamardGenerator
from mthree.utils import final_measurement_mapping, expval

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

def chop_dict_list(dict_list, list_of_generators):
    """
    Chop up a list of dictionaries into a list of list of dictionaries based on the
    lengths of the generators in the list_of_generators.

    Parameters:
        list_of_generators (list): List of generators specifying the lengths of sublists.
        dict_list (list): List of dictionaries.

    Returns:
        list: List of list of dictionaries.
    """
    lengths = [len(sublist) for sublist in list_of_generators]
    result = []
    remaining_dicts = dict_list[:]
    for length in lengths:
        sublist = remaining_dicts[:length]
        result.append(sublist)
        remaining_dicts = remaining_dicts[length:]
    return result

def chop_undo_twirling_and_merge(dict_list, list_of_generators):
    """
    Chop the list of dictionaries based on list_of_generators and then
    undo the effect of twirling from the generator for the count dictionaries in the dict_list and merge
    the resulting list of dictionaries

    Parameters:
        dict_list (list): List of dictionaries.
        generator (list): List of binary strings from the generator

    Returns:
        dict: Merged dictionary
    Raises Error if the length of dict_list does not match length of generator
    """
    chopped_counts = chop_dict_list(dict_list, list_of_generators)
    return [undo_twirling_and_merge(list_of_counts, generator) for list_of_counts, generator in zip(chopped_counts, list_of_generators)]

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

def marginalize_tw_calib_counts(final_measurement_mapping, physical_to_bit_mapping, counts):
    """
    Process the counts dictionary based on the measurement mapping dictionaries by marginalizing over non-relevant bits.

    Parameters:
        final_measurement_mapping (dict): Final measurement mapping dictionary.
        physical_to_bit_mapping (dict): Physical-to-bit mapping dictionary.
        counts (dict): Counts dictionary.

    Returns:
        dict: Processed counts dictionary.
    """
    final_measurement_mapping_interchanged = key_val_interchange(final_measurement_mapping)
    bit_to_physical_mapping = key_val_interchange(physical_to_bit_mapping)

    indices_to_keep = []
    for circ_logical_bit, circ_physical_bit in final_measurement_mapping.items():
        calib_logical_bit = physical_to_bit_mapping[circ_physical_bit]
        indices_to_keep.append(calib_logical_bit)

    shuffled_marginalized_counts = marginal_counts(result=counts, indices=indices_to_keep)

    # marginalization does not care about the ordering of bits so 
    # we must restore the appropriate bit-ordering 
    # marginalized_counts = marginal_counts(result=counts, indices=indices_to_keep)

    indices_to_shuffle = []
    for index in indices_to_keep:
        calib_physical_bit = bit_to_physical_mapping[index]
        circ_logical_bit = final_measurement_mapping_interchanged[calib_physical_bit]
        indices_to_shuffle.append(circ_logical_bit)

    # now we unshuffle the shuffle marginalized counts to get marginalized counts 
    marginalized_counts = {permute_bitstring(key, indices_to_shuffle):val for key,val in shuffled_marginalized_counts.items()}
    
    return marginalized_counts

def shots_from_dict(counts):
    """
    Find the total number of shots in a given dictionary of counts. 

    Parameters:
        counts (dict): Counts dictionary.

    Returns:
        shots: Total number of shots
    """
    return sum(counts.values())

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

def flatten_with_sum(lsts):
    # flatten a list using the sum function 
    return sum(lsts, [])

def tw_expval(quantum_circ, operator, tw_circuits, tw_calibration):
    """
    Use the twirled circuits and calibration data to compute expectation value of an operator for a given circuit

    Parameters:
        quantum_circ (circuit): transpiled circuit whose output is used to find expectation value
        operator (str or dict or list): String or dict representation of diagonal
                                        qubit operators used in computing the expectation
                                        value.
        tw_circuits (object): Twirled Circuit object containing circuit measurement data
        tw_calibration (object): Twirled Calibration object containing calibration data

    Returns:
        list: list of results for the unmitigated expectation value, mitigated expectation value and uncertainity estimate
    """
    # first we check if measurements have been performed for the tw_circuits and tw_calibration objects
    circuits_data = tw_circuits.to_tw_circuits_data()
    calibrations_data = tw_calibration.to_tw_calibration()

    if len(circuits_data)==0:
        raise ValueError('Measurement Data has not been acquired for Twirled Circuits.')
    if len(calibrations_data)==0:
        raise ValueError('Measurement Data has not been acquired for Twirled Calibration.')
    
    # find circuit measurement mapping and calibration measurement mapping 
    calibration_mapping = tw_calibration.physical_to_bit_mapping
    circuit_measurement_mapping  = final_measurement_mapping(quantum_circ)

    # We find the corresponding data for the circuit stored inside tw_circuits object
    # and then use it to compute mitigated expectation value. 

    # If the same transpiled circuit is ran multiple times then the all the expectation values will be returned 
    # For this to occur even the physical measured qubits and the mapping to hardware has to be exact

    # find list of all circuits that are twirled in the tw_circuits object
    circuits_list = tw_circuits.circuits 
    #print(circuits_list)

    if quantum_circ not in circuits_list:
        raise ValueError('The Twirled Circuits object does not contain the given circuit.')

    expval_results = []
    for circuit,circuit_data in zip(circuits_list, circuits_data):
        if circuit==quantum_circ:
            # find the expectation value from the circuit data and the associated uncertainity
            circuit_data_counts = shots_from_dict(circuit_data)
            #print(f"Total shots for Circuit data: {circuit_data_counts}")
            expval_circuit = expval(items=circuit_data, exp_ops=operator)
            expval_circuit_uncertainity = np.sqrt(2*np.log(4/0.05)/circuit_data_counts)

            # find the expectation value from the calibration data and the associated uncertainity
            # first we must marginalize over the qubits that are not relevant to the given circuit qc
            calib_data = marginalize_tw_calib_counts(final_measurement_mapping=circuit_measurement_mapping, physical_to_bit_mapping=calibration_mapping, counts=calibrations_data)
            calib_data_counts = shots_from_dict(calib_data)
            #print(f"Total shots for Calibration data: {calib_data_counts}")
            expval_calib = expval(items=calib_data, exp_ops=operator)
            expval_calib_uncertainity = np.sqrt(2*np.log(4/0.05)/calib_data_counts)

            # now we can compute the mitigated expectation value and the overall uncertainity
            # note that the uncertainity estimate only depends on the number of circuit and calibration counts
            # and the expectation value from the calibration data and is an upper bound derived from Hoeffiding's inequality
            unmitigated_expval = expval_circuit
            mitigated_expval = expval_circuit/expval_calib
            uncertainity_mitigated_expval = (expval_circuit_uncertainity+expval_calib_uncertainity)/(np.abs(expval_calib)-expval_calib_uncertainity)

            # store the resulting expectation values and uncertainity
            result = [unmitigated_expval, mitigated_expval, uncertainity_mitigated_expval]
            expval_results.append(result)

    return expval_results