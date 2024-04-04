from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket_mbqc_py import GraphCircuit, CNOTBlocksGraphCircuit
from pytket.unit_id import BitRegister
import pytest


def test_plus_state():
    circuit = GraphCircuit(n_physical_qubits=2)

    input_qubit, vertex_one = circuit.add_input_vertex()
    circuit.H(input_qubit)

    vertex_two = circuit.add_graph_vertex()

    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
    circuit.corrected_measure(vertex=vertex_one)

    output_qubits = circuit.get_outputs()

    output_reg = BitRegister(name="output", size=1)
    circuit.add_c_register(register=output_reg)
    circuit.Measure(qubit=output_qubits[vertex_two], bit=output_reg[0])

    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )
    compiled_circuit = backend.get_compiled_circuit(circuit)
    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )

    assert result.get_counts(output_reg)[(0,)] == 100


def test_x_gate():
    circuit = GraphCircuit(n_physical_qubits=3)

    _, vertex_one = circuit.add_input_vertex()

    vertex_two = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
    vertex_three = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_three)
    circuit.corrected_measure(vertex=vertex_one, t_multiple=0)
    circuit.corrected_measure(vertex=vertex_two, t_multiple=4)

    output_qubits = circuit.get_outputs()
    output_reg = BitRegister(name="output", size=1)
    circuit.add_c_register(register=output_reg)
    circuit.Measure(
        qubit=output_qubits[vertex_three],
        bit=output_reg[0],
    )

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_circuit = backend.get_compiled_circuit(circuit)
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=100,
    )

    assert result.get_counts(output_reg)[(1,)] == 100


@pytest.mark.parametrize(
    "input_state, output_state",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_cnot(input_state, output_state):
    circuit = GraphCircuit(n_physical_qubits=5)

    target_qubit, vertex_one = circuit.add_input_vertex()
    if input_state[1]:
        circuit.X(target_qubit)

    vertex_two = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)

    control_qubit, vertex_three = circuit.add_input_vertex()
    if input_state[0]:
        circuit.X(control_qubit)

    vertex_four = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_four)
    circuit.corrected_measure(vertex=vertex_one, t_multiple=0)

    vertex_five = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_three, vertex_two=vertex_five)

    vertex_six = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_four, vertex_two=vertex_six)
    circuit.corrected_measure(vertex=vertex_two, t_multiple=0)

    vertex_seven = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_five, vertex_two=vertex_seven)
    circuit.corrected_measure(vertex=vertex_three, t_multiple=0)

    vertex_eight = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_eight)

    circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_seven)

    circuit.corrected_measure(vertex=vertex_four, t_multiple=0)
    circuit.corrected_measure(vertex=vertex_five, t_multiple=0)
    circuit.corrected_measure(vertex=vertex_six, t_multiple=0)

    output_qubits = circuit.get_outputs()
    output_reg = BitRegister(name="output", size=2)
    circuit.add_c_register(register=output_reg)
    circuit.Measure(qubit=output_qubits[vertex_seven], bit=output_reg[0])
    circuit.Measure(qubit=output_qubits[vertex_eight], bit=output_reg[1])

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_circuit = backend.get_compiled_circuit(circuit)

    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )

    assert result.get_counts(output_reg)[output_state] == n_shots


@pytest.mark.parametrize(
    "input_state, output_state, n_columns",
    [
        ((1, 0), (1, 1), 1),
        ((1, 1), (1, 1), 2),
        ((1, 1, 0), (1, 1, 1), 2),
        ((0, 1, 0), (0, 1, 1), 3),
        ((1, 1, 1), (1, 1, 1), 4),
    ],
)
def test_cnot_block(input_state, output_state, n_columns):

    n_physical_qubits=20
    n_rows = len(input_state) - 1

    circuit = CNOTBlocksGraphCircuit(
        n_physical_qubits=n_physical_qubits,
        input_state=input_state,
        n_rows=n_rows,
        n_columns=n_columns,
    )
        
    output_vertex_quibts = circuit.get_outputs()
    output_reg = BitRegister(name="output", size=len(output_vertex_quibts))
    circuit.add_c_register(register=output_reg)
    for i, qubit in enumerate(output_vertex_quibts.values()):
        circuit.Measure(qubit=qubit, bit=output_reg[i])

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_circuit = backend.get_compiled_circuit(circuit)

    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )
    assert result.get_counts(output_reg)[output_state] == n_shots
