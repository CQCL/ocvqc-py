from pytket_mbqc_py.qubit_manager import QubitManager
from pytket_mbqc_py.graph_circuit import GraphCircuit
from pytket.unit_id import BitRegister
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from typing import List, Tuple, Dict
from pytket import Qubit
from pytket_mbqc_py.wasm_file_handler import get_wasm_file_handler
from pytket import BitRegister
import numpy as np
import networkx as nx  # type:ignore

def run_path_graph_n_qubits(num_qubits: int, number_shots: int):
    circuit = GraphCircuit(n_qubits_total=num_qubits)
    index_dict = {}

    for i in range(num_qubits):
        if i == 0:
            qubit_one, index_dict[f"index_{i}"] = circuit.add_input_vertex()
        elif i == num_qubits-1:
            index_dict[f"index_{i}"] = circuit.add_output_vertex()
        else:
            index_dict[f"index_{i}"] = circuit.add_graph_vertex()

    keys = list(index_dict.keys())

    for i in range(1, len(keys)):
        key_i = keys[i]
        key_i_minus_1 = keys[i - 1]
        vertex_i = index_dict[key_i]
        vertex_i_minus_1 = index_dict[key_i_minus_1]
        circuit.add_edge(vertex_i_minus_1, vertex_i)

    for i in range(len(keys)-1):   
        vertex_i = index_dict[keys[i]]
        print(vertex_i)
        circuit.corrected_measure(vertex=vertex_i)

    circuit.correct_outputs()
    output_reg = BitRegister(name="output", size=1)
    circuit.add_c_register(register=output_reg)
    circuit.Measure(qubit=circuit.output_qubits[num_qubits-1], bit=output_reg[0])

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_circuit = backend.get_compiled_circuit(circuit)

    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )
    compiled_circuit = backend.get_compiled_circuit(circuit)
    n_shots = number_shots

    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
        wasm_file_handler=get_wasm_file_handler(),
    )

    values = result.get_counts(output_reg).values()
    values_array = np.array(list(values))
    total = np.sum(values_array)
    normalized_values = values_array / total
    normalized_values



