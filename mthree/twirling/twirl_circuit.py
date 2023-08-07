"""Twirled Circuit object"""
import warnings
import datetime
from dateutil import tz
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister

from mthree.exceptions import M3Error
from mthree._helpers import system_info
from mthree.generators import HadamardGenerator
from mthree.utils import final_measurement_mapping
from mthree.calibrations.src import calibration_to_texmex, calibration_to_texmex_counts
from mthree.twirling.tw_utils import vals_from_dict


class Tw_Circuit:
    """Twirled Circuit object"""

    def __init__(self, backend, circuit=None, generator=None,
                 num_random_circuits=None, seed=None):
        """Twirled Circuit object

        Parameters:
            backend (Backend): Target backend 
            circuit (circuit): Transpiled circuit to twirl 
            generator (Generator): Generator for twirling circuits
            num_random_circuits (int): Number of random circuits if
                                       using random generator, default=16
            seed (int): Seed for random circuit generation, default=None
        """
        self.backend = backend
        self.backend_info = system_info(backend)

        if circuit is None:
            raise ValueError('There must be a circuit to twirl.')
        else:
            self.circuit = circuit

        self.bit_to_physical_mapping = final_measurement_mapping(circuit)

        self.qubits = vals_from_dict(self.bit_to_physical_mapping)
        self.num_qubits = len(self.qubits)
        
        if seed is None:
            seed = np.random.seed()

        if generator is None:
            gen = HadamardGenerator(self.num_qubits)
        elif 'Random' in generator.__name__:
            # For random and random-compliment generators
            if num_random_circuits is None:
                num_random_circuits = 16
            gen = generator(self.num_qubits, num_arrays=num_random_circuits, seed=seed)
        else:
            gen = generator(self.num_qubits)
            if num_random_circuits is not None or seed is not None:
                warnings.warn(f'random generator settings not applicable for {gen.name} generator')
        self.generator = gen
        self.tw_data = None
        self.num_circuits = self.generator.length
        self.shots_per_circuit = None

    def tw_circuits(self):
        """Twirled circuits from underlying generator

        Returns:
            list: Twirled circuits
        """
        out_circuits = []
        # obtain circuits after twirling by X gates based on generator 
        for string in self.generator:
            qc_twirled = self.circuit.remove_final_measurements(inplace=False)
            qc_twirled.add_register(ClassicalRegister(self.num_qubits))

            for idx, val in enumerate(string[::-1]):
                if val:
                    qc_twirled.x(self.bit_to_physical_mapping[idx])
                qc_twirled.measure(self.bit_to_physical_mapping[idx], idx)
            out_circuits.append(qc_twirled)
        return out_circuits

    def tw_data_from_backend(self, shots=8192):
        """Twirled data from the target backend using the generator circuits
        Parameters:
            shots(int): Total number of shots
        """
        self.tw_data = None
        self._job_error = None
        self.shots_per_circuit = int(shots/self.num_circuits)

        job = self.backend.run(self.tw_circuits(), shots=self.shots_per_circuit)
        self.job_id = job.job_id()
        res = job.result()
        self.tw_data = res.get_counts()
        return 

    def to_texmex_data(self):
        """
        Return untwirled data
        """
        if self.tw_data is None:
            raise M3Error('No data has been acquired')
        return calibration_to_texmex_counts(self.tw_data, self.generator)


def texmex_data(backend, circuit=None, shots=8192, generator=None,
                 num_random_circuits=None, seed=None):
    """Get TexMex Data from Circuit

    Parameters:
        backend (Backend): Target backend 
        shots (int): Number of shots
        circuit (circuit): Transpiled circuit to twirl 
        generator (Generator): Generator for twirling circuits
        num_random_circuits (int): Number of random circuits if
                                    using random generator, default=16
        seed (int): Seed for random circuit generation, default=None
    """
    tw_circuit = Tw_Circuit(backend, circuit, generator,num_random_circuits,seed)
    tw_circuit.tw_data_from_backend(shots)
    return tw_circuit.to_texmex_data()