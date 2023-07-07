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
from mthree.utils import final_measurement_mapping
from mthree.twirling.tw_utils import chop_undo_twirling_and_merge

class Tw_Circuits:
    """Twirled Circuits object"""

    def __init__(self, backend, circuits=None, generator=None,
                 num_random_circuits=None, seed=None):
        """Twirled Circuit object

        Parameters:
            backend (Backend): Target backend 
            circuits (list): List of transpiled circuits to twirl 
            generator (Generator): Generator for twirling circuits
            num_random_circuits (int): Number of random circuits if
                                       using random generator, default=16
            seed (int): Seed for random circuit generation, default=None
        """
        self.backend = backend

        if circuits is None:
            raise ValueError('There must be at least one circuit to twirl.')
        else:
            if not isinstance(circuits, list):
                self.circuits = [circuits]
            else:
                self.circuits = circuits

        self.generator = generator
        self.backend_info = system_info(backend)
        self.tw_generators_list = None
        self.tw_measurement_maps = None

        self.num_tw_circuits = None

        self.num_random_circuits = num_random_circuits
        if seed is None:
            self.seed = np.random.seed()
        
        self._tw_circuits_data = None
        self.shots_per_circuit = None

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
            if attr in ["_tw_circuits_data", "timestamp"]:
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
    def tw_circuits_data(self):
        """Twirled Circuit data"""
        if self._tw_circuits_data is None and self._thread is None:
            raise M3Error("Twirled Circuits are not executed")
        return self._tw_circuits_data

    @property
    def timestamp(self):
        """Timestamp of circuit job

        Time is stored as UTC but returned in local time

        Returns:
            datetime: Timestamp in local time
        """
        if self._timestamp is None:
            return self._timestamp
        return self._timestamp.astimezone(tz.tzlocal())

    @tw_circuits_data.setter
    def tw_circuits_data(self, cals):
        if self._tw_circuits_data is not None:
            raise M3Error("Twirled Circuits are already executed")
        self._tw_circuits_data = cals

    def tw_circuits_and_generators(self):
        """Twirled Circuits and list of generators from list of input circuits and underlying generator

        Returns:
            list: Twirled circuits 
            list: List of Generators used to twirl circuits
        """
        out_circuits = []
        list_of_generators = []
        measurement_maps = []

        for qc in self.circuits:
            # find final measurement layout and number of active measured qubits
            layout = final_measurement_mapping(qc)
            num_measured_qubits = len(layout)
            measurement_maps.append(layout)

            # find generator from the given generator method 
            if self.generator is None:
                gen = HadamardGenerator(num_measured_qubits)
            elif 'Random' in self.generator.__name__:
                # For random and random-compliment generators
                if self.num_random_circuits is None:
                    self.num_random_circuits = 16
                gen = self.generator(num_measured_qubits, num_arrays=self.num_random_circuits, seed=self.seed)
            else:
                gen = self.generator(num_measured_qubits)
                if self.num_random_circuits is not None or self.seed is not None:
                    warnings.warn(f'random generator settings not applicable for {gen.name} generator')
            # append to list of generators for all circuits
            list_of_generators.append(list(gen))

            # obtain circuits after twirling by X gates based on generator 
            for string in gen:
                qc_twirled = qc.remove_final_measurements(inplace=False)
                qc_twirled.add_register(ClassicalRegister(num_measured_qubits))

                for idx, val in enumerate(string[::-1]):
                    if val:
                        qc_twirled.x(layout[idx])
                    qc_twirled.measure(layout[idx], idx)
                out_circuits.append(qc_twirled)
        return out_circuits, list_of_generators, measurement_maps

    def tw_data_from_backend(self, shots=int(1e4), async_cal=True, overwrite=False):
        """Twirled Data from the target backend using the generator circuits

        Parameters:
            shots(int): Number of shots per twirled circuit
            async_cal (bool): Perform data acquistion asyncronously, default=True
            overwrite (bool): Overwrite a previous data collection, default=False

        Raises:
            M3Error: Twirled Circuits are already executed and overwrite=False
        """
        if self._tw_circuits_data is not None and (not overwrite):
            M3Error("Twirled Circuits are already executed and overwrite=False")
        self._tw_circuits_data = None
        tw_circuits, tw_generators, tw_measurement_maps = self.tw_circuits_and_generators()
        
        self.num_tw_circuits = len(tw_circuits)
        self.tw_generators_list = tw_generators
        self.tw_measurement_maps = tw_measurement_maps

        self._job_error = None
        self.shots_per_circuit = shots
        cal_job = self.backend.run(tw_circuits, shots=self.shots_per_circuit)
        self.job_id = cal_job.job_id()
        if async_cal:
            thread = threading.Thread(
                target=_job_thread,
                args=(cal_job, self),
            )
            self._thread = thread
            self._thread.start()
        else:
            _job_thread(cal_job, self)

    def to_tw_circuits_data(self):
        if self.tw_circuits_data is None:
            raise M3Error('Twirled Circuit data is unavailable')
        return chop_undo_twirling_and_merge(self.tw_circuits_data, self.tw_generators_list)


def _job_thread(job, cal):
    """Process job result async"""
    try:
        res = job.result()
    # pylint: disable=broad-except
    except Exception as error:
        cal._job_error = error
        return
    else:
        cal.tw_circuits_data = res.get_counts()
        timestamp = res.date
        # Needed since Aer result date is str but IBMQ job is datetime
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.isoformat()
        # Go to UTC times because we are going to use this for
        # resultsDB storage as well
        dt = datetime.datetime.fromisoformat(timestamp)
        dt_utc = dt.astimezone(datetime.timezone.utc)
        cal._timestamp = dt_utc
