from pytket_mbqc_py import get_wasm_file_handler
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket_mbqc_py import GraphCircuit
from pytket.unit_id import BitRegister
import pytest


def test_plus_state():
    circuit = GraphCircuit(n_qubits_total=2)

    input_qubit, index_one = circuit.add_input_vertex()
    circuit.H(input_qubit)

    circuit.add_output_vertex()

    circuit.add_edge(vertex_one=0, vertex_two=1)
    circuit.corrected_measure(vertex=0)

    circuit._apply_correction(vertex=1)

    output_reg = BitRegister(name="output", size=1)
    circuit.add_c_register(register=output_reg)
    circuit.Measure(qubit=circuit.output_qubits[1], bit=output_reg[0])

    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )
    compiled_circuit = backend.get_compiled_circuit(circuit)
    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
        wasm_file_handler=get_wasm_file_handler(),
    )

    assert result.get_counts(output_reg)[(0,)] == 100


def test_x_gate():
    circuit = GraphCircuit(n_qubits_total=3)

    input_qubit, index_one = circuit.add_input_vertex()

    circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=0, vertex_two=1)
    circuit.add_output_vertex()
    circuit.add_edge(vertex_one=1, vertex_two=2)
    circuit.corrected_measure(vertex=0, t_multiple=0)
    circuit.corrected_measure(vertex=1, t_multiple=4)

    circuit._apply_correction(vertex=2)
    output_reg = BitRegister(name="output", size=1)
    circuit.add_c_register(register=output_reg)
    circuit.Measure(qubit=circuit.output_qubits[2], bit=output_reg[0])

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_circuit = backend.get_compiled_circuit(circuit)
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=100,
        wasm_file_handler=get_wasm_file_handler(),
    )

    assert result.get_counts(output_reg)[(1,)] == 100


@pytest.mark.parametrize(
    "input_state, output_state",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_cnot(input_state, output_state):
    circuit = GraphCircuit(n_qubits_total=5)

    target_qubit, _ = circuit.add_input_vertex()
    if input_state[1]:
        circuit.X(target_qubit)

    circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=0, vertex_two=1)

    control_qubit, _ = circuit.add_input_vertex()
    if input_state[0]:
        circuit.X(control_qubit)

    circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=1, vertex_two=3)
    circuit.corrected_measure(vertex=0, t_multiple=0)

    circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=2, vertex_two=4)

    circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=3, vertex_two=5)
    circuit.corrected_measure(vertex=1, t_multiple=0)

    circuit.add_output_vertex()
    circuit.add_edge(vertex_one=4, vertex_two=6)
    circuit.corrected_measure(vertex=2, t_multiple=0)

    circuit.add_output_vertex()
    circuit.add_edge(vertex_one=5, vertex_two=7)

    circuit.add_edge(vertex_one=5, vertex_two=6)

    circuit.corrected_measure(vertex=3, t_multiple=0)
    circuit.corrected_measure(vertex=4, t_multiple=0)
    circuit.corrected_measure(vertex=5, t_multiple=0)

    circuit.correct_outputs()
    output_reg = BitRegister(name="output", size=2)
    circuit.add_c_register(register=output_reg)
    circuit.Measure(qubit=circuit.output_qubits[6], bit=output_reg[0])
    circuit.Measure(qubit=circuit.output_qubits[7], bit=output_reg[1])

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_circuit = backend.get_compiled_circuit(circuit)

    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
        wasm_file_handler=get_wasm_file_handler(),
    )

    assert result.get_counts(output_reg)[output_state] == n_shots
