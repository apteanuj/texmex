import numpy as np
from qiskit import *
from qiskit.providers.fake_provider import FakeManila, FakeKolkata, FakeMelbourne
from mthree.twirling.twirl_circuit import texmex_data
from mthree.utils import final_measurement_mapping
from mthree import M3Mitigation
from mthree.calibrations import Calibration
from mthree.generators import CompleteGenerator, HadamardGenerator, RandomGenerator
import matplotlib.pyplot as plt
from mthree.utils import expval

backend = FakeKolkata()

def GHZ(N):
    qc = QuantumCircuit(N, N)
    qc.h(0)
    for i in range(1,N):
        qc.cx(0,i)
    for i in range(N):
        qc.measure(i,i)
    trans_qc = transpile(qc, backend, optimization_level=2, seed_transpiler=12345)
    return trans_qc

def m3circuit(N):
    qc = QuantumCircuit(N, N)
    qc.x(range(N))
    qc.h(range(N))

    for kk in range(N // 2, 0, -1):
        qc.ch(kk, kk - 1)
    for kk in range(N // 2, N - 1):
        qc.ch(kk, kk + 1)
    for i in range(N):
        qc.measure(i,i)
    trans_qc = transpile(qc, backend, optimization_level=2, seed_transpiler=12345)
    return trans_qc

def donothing(N):
    qc = QuantumCircuit(N, N)
    for i in range(N):
        qc.measure(i,i)
    trans_qc = transpile(qc, backend, optimization_level=2, seed_transpiler=12345)
    return trans_qc

transpiled_circuit = GHZ(6)
measurement_map = final_measurement_mapping(transpiled_circuit)
qubits = [value for key,value in measurement_map.items()]
#print(qubits)
op = 'ZZZZZZ'
print(f"Circuit depth: {transpiled_circuit.depth()}")

cal = Calibration(backend=backend,qubits=qubits)
cal.calibrate_from_backend(shots=2**13)

texmex_counts = texmex_data(backend=backend, circuit=transpiled_circuit, shots=2**13)

print(f"Unmitigated Expectation value: {expval(texmex_counts,op):.4f}")
print(f"Mitigated Expectation value: {cal.mitigated_expval_std(texmex_counts,qubits,op)[0]:.4f}")
print(f"Mitigation Standard Deviation: {cal.mitigated_expval_std(texmex_counts,qubits,op)[1]:.4f}")