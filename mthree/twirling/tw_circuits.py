"""Twirled Circuit object"""
import threading
import warnings
import datetime
from dateutil import tz
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister

from mthree.exceptions import M3Error
from mthree._helpers import system_info
from mthree.generators import HadamardGenerator
from mthree.calibrations.mapping import calibration_mapping
from mthree.utils import final_measurement_mapping
from mthree.twirling.tw_utils import undo_twirling_and_merge,vals_from_dict, util_expval, util_expval_and_stddev

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

        self._tw_data = None
        self.num_circuits = self.generator.length
        
        self.job_id = None
        self._timestamp = None
        self._thread = None
        self._job_error = None

    def __getattribute__(self, attr):
        """This allows for checking the status of the threaded cals call

        For certain attr this will join the thread and/or raise an error.
        """
        __dict__ = super().__getattribute__("__dict__")
        if attr in __dict__:
            if attr in ["_tw_data", "timestamp"]:
                self._thread_check()
        return super().__getattribute__(attr)

    def _thread_check(self):
        """Check if a thread is running and join it.

        Raise an error if one is given.
        """
        if self._thread and self._thread != threading.current_thread():
            self._thread.join()
            self._thread = None
        if self._job_error:
            raise self._job_error  # pylint: disable=raising-bad-type

    @property
    def tw_data(self):
        """Twirled Circuit data"""
        if self._tw_data is None and self._thread is None:
            raise M3Error("Circuits are not executed")
        return self._tw_data

    @property
    def timestamp(self):
        """Timestamp of job

        Time is stored as UTC but returned in local time

        Returns:
            datetime: Timestamp in local time
        """
        if self._timestamp is None:
            return self._timestamp
        return self._timestamp.astimezone(tz.tzlocal())

    @tw_data.setter
    def tw_data(self, counts):
        if self._tw_data is not None:
            raise M3Error("Circuits are already executed")
        self._tw_data = counts

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

    def tw_data_from_backend(self, shots=8192, async_job=True, overwrite=False):
        """Twirled data from the target backend using the generator circuits

        Parameters:
            shots(int): Total number of shots
            async_job (bool): Perform data acquistion asyncronously, default=True
            overwrite (bool): Overwrite previously acquired data, default=False

        Raises:
            M3Error: Twirled data is already acquired and overwrite=False
        """
        if self._tw_data is not None and (not overwrite):
            M3Error("Twirled data is already acquired and overwrite=False")
        self._tw_data = None
        self._job_error = None
        shots_per_circuit = int(shots/self.num_circuits)

        job = self.backend.run(self.tw_circuits(), shots=shots_per_circuit)
        self.job_id = job.job_id()
        if async_job:
            thread = threading.Thread(
                target=_job_thread,
                args=(job, self),
            )
            self._thread = thread
            self._thread.start()
        else:
            _job_thread(job, self)

    def to_untwirled_data(self):
        """
        Return untwirled data
        """
        if self.tw_data is None:
            raise M3Error('No data has been acquired')
        return undo_twirling_and_merge(self.tw_data, self.generator)


def _job_thread(job, cal):
    """Process job result async"""
    try:
        res = job.result()
    # pylint: disable=broad-except
    except Exception as error:
        cal._job_error = error
        return
    else:
        cal.tw_data = res.get_counts()
        timestamp = res.date
        # Needed since Aer result date is str but IBMQ job is datetime
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.isoformat()
        # Go to UTC times because we are going to use this for
        # resultsDB storage as well
        dt = datetime.datetime.fromisoformat(timestamp)
        dt_utc = dt.astimezone(datetime.timezone.utc)
        cal._timestamp = dt_utc

# consider cythonizing undo_twirling_and_merge function for faster processing