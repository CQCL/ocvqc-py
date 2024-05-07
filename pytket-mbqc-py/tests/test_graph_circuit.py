import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
from pytket.unit_id import BitRegister

from pytket_mbqc_py import CNOTBlocksGraphCircuit, GraphCircuit


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
    "input_state, output_state, n_layers",
    [
        ((1, 0), (1, 1), 1),
        ((1, 1), (1, 1), 2),
        ((1, 1, 0), (1, 1, 1), 2),
        ((0, 1, 0), (0, 1, 1), 3),
        ((1, 1, 1), (1, 1, 1), 4),
        ((0, 0, 1), (0, 0, 1), 1),
        ((0, 1, 1), (0, 1, 0), 1),
    ],
)
def test_cnot_block(input_state, output_state, n_layers):
    n_physical_qubits = 20

    circuit = CNOTBlocksGraphCircuit(
        n_physical_qubits=n_physical_qubits,
        input_state=input_state,
        n_layers=n_layers,
    )

    output_vertex_quibts = circuit.get_outputs()
    output_reg = BitRegister(name="output", size=len(output_vertex_quibts))
    circuit.add_c_register(register=output_reg)
    for i, qubit in enumerate(output_vertex_quibts.values()):
        circuit.Measure(qubit=qubit, bit=output_reg[i])
    assert circuit.output_state == output_state

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_circuit = backend.get_compiled_circuit(circuit)

    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )
    assert result.get_counts(output_reg)[output_state] == n_shots


@pytest.mark.high_compute
def test_large_cnot_block():
    input_state = (0, 1, 1, 0)
    output_state = (0, 1, 0, 1)
    n_layers = 3
    n_physical_qubits = 20

    circuit = CNOTBlocksGraphCircuit(
        n_physical_qubits=n_physical_qubits,
        input_state=input_state,
        n_layers=n_layers,
    )

    output_vertex_quibts = circuit.get_outputs()
    output_reg = BitRegister(name="output", size=len(output_vertex_quibts))
    circuit.add_c_register(register=output_reg)
    for i, qubit in enumerate(output_vertex_quibts.values()):
        circuit.Measure(qubit=qubit, bit=output_reg[i])
    assert circuit.output_state == output_state

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_circuit = backend.get_compiled_circuit(circuit)

    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )
    assert result.get_counts(output_reg)[output_state] == n_shots


def test_3_q_ghz():
    graph_circuit = GraphCircuit(n_physical_qubits=5)

    input_quibt, input_vertex = graph_circuit.add_input_vertex()

    graph_circuit.H(input_quibt)

    vertex_layer_1_1 = graph_circuit.add_graph_vertex()
    vertex_layer_1_2 = graph_circuit.add_graph_vertex()

    graph_circuit.add_edge(input_vertex, vertex_layer_1_1)
    graph_circuit.add_edge(input_vertex, vertex_layer_1_2)

    graph_circuit.corrected_measure(vertex=input_vertex)

    vertex_layer_2_1 = graph_circuit.add_graph_vertex()
    vertex_layer_2_2 = graph_circuit.add_graph_vertex()

    graph_circuit.add_edge(vertex_layer_1_1, vertex_layer_2_1)
    graph_circuit.add_edge(vertex_layer_1_1, vertex_layer_2_2)

    graph_circuit.corrected_measure(vertex=vertex_layer_1_1)

    output_qubits = graph_circuit.get_outputs()

    graph_circuit.H(output_qubits[3])
    graph_circuit.H(output_qubits[4])

    graph_circuit.CX(output_qubits[2], output_qubits[3])
    graph_circuit.CX(output_qubits[2], output_qubits[4])
    graph_circuit.H(output_qubits[2])

    output_c_reg = graph_circuit.add_c_register(
        name="output measure reg", size=len(output_qubits)
    )
    for qubit, bit in zip(output_qubits.values(), output_c_reg):
        graph_circuit.Measure(qubit=qubit, bit=bit)

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_graph_circuit = backend.get_compiled_circuit(
        circuit=graph_circuit, optimisation_level=0
    )
    n_shots = 100
    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=output_c_reg)[(0, 0, 0)] == n_shots


@pytest.mark.parametrize(
    "input_state, output_state",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_cnot_early_measure(input_state, output_state):
    circuit = GraphCircuit(n_physical_qubits=3)

    target_qubit, vertex_one = circuit.add_input_vertex()
    if input_state[1]:
        circuit.X(target_qubit)

    vertex_two = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)

    circuit.corrected_measure(vertex=vertex_one, t_multiple=0)

    control_qubit, vertex_three = circuit.add_input_vertex()
    if input_state[0]:
        circuit.X(control_qubit)

    vertex_four = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_four)

    circuit.corrected_measure(vertex=vertex_two, t_multiple=0)

    vertex_five = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_three, vertex_two=vertex_five)

    circuit.corrected_measure(vertex=vertex_three, t_multiple=0)

    vertex_six = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_four, vertex_two=vertex_six)

    circuit.corrected_measure(vertex=vertex_four, t_multiple=0)

    vertex_seven = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_five, vertex_two=vertex_seven)

    circuit.corrected_measure(vertex=vertex_five, t_multiple=0)

    vertex_eight = circuit.add_graph_vertex()
    circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_eight)

    circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_seven)

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

def test_2q_t_gate_example():

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler = api_offline)

    graph_circuit = GraphCircuit(n_physical_qubits=6)

    _, input_vertex_0 = graph_circuit.add_input_vertex()

    # H[0]S[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(input_vertex_0, graph_vertex_0_0)
    graph_circuit.corrected_measure(input_vertex_0, t_multiple=2)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    _, input_vertex_1 = graph_circuit.add_input_vertex()

    # CZ[0,1]H[1]H[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=0)

    graph_vertex_1_0 = graph_circuit.add_graph_vertex()

    graph_vertex_0_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)

    graph_circuit.add_edge(input_vertex_1, graph_vertex_1_0)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_1_0)

    graph_circuit.corrected_measure(input_vertex_1, t_multiple=0)

    graph_vertex_1_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_1_0, graph_vertex_1_1)

    # H[0]
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    # H[1]
    graph_circuit.corrected_measure(graph_vertex_1_0, t_multiple=0)

    # H[0]Z[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=4)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)

    # CZ[0,1]H[0]H[1]
    graph_vertex_0_2 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_2)

    graph_vertex_1_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_1_1, graph_vertex_1_0)
    graph_circuit.corrected_measure(graph_vertex_1_1, t_multiple=0)

    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=0)

    graph_vertex_0_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_2, graph_vertex_0_1)

    graph_vertex_1_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_1_0, graph_vertex_1_1)

    graph_circuit.add_edge(graph_vertex_0_2, graph_vertex_1_0)

    # H[0]
    graph_circuit.corrected_measure(graph_vertex_0_2, t_multiple=0)

    # H[1]
    graph_circuit.corrected_measure(graph_vertex_1_0, t_multiple=0)

    # H[0]S[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=2)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    outputs = graph_circuit.get_outputs()
    out_meas_reg = graph_circuit.add_c_register(name='output measure', size=len(outputs))
    for qubit, bit in zip(outputs.values(), out_meas_reg):
        graph_circuit.Measure(qubit=qubit, bit=bit)

    copmiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)
    n_shots = 1000
    result = backend.run_circuit(circuit=copmiled_graph_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=out_meas_reg)[(1,0)] == n_shots

def test_1q_t_gate_example():

    ################################
    # The following compiles to I

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler = api_offline)

    graph_circuit = GraphCircuit(n_physical_qubits=2)

    _, input_vertex_0 = graph_circuit.add_input_vertex()

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(input_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(input_vertex_0, t_multiple=0)

    # H[0]T[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=1)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    # H[0]T[0]S[0]Z[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=7)

    outputs = graph_circuit.get_outputs()
    out_meas_reg = graph_circuit.add_c_register(name='output measure', size=len(outputs))
    for qubit, bit in zip(outputs.values(), out_meas_reg):
        graph_circuit.Measure(qubit=qubit, bit=bit)

    copmiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots=1000
    result = backend.run_circuit(circuit=copmiled_graph_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=out_meas_reg)[(0,)] == n_shots

    ################################
    # The following compiles to X

    graph_circuit = GraphCircuit(n_physical_qubits=2)

    _, input_vertex_0 = graph_circuit.add_input_vertex()

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(input_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(input_vertex_0, t_multiple=0)

    # H[0]T[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=1)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    # H[0]T[0]S[0]Z[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex()
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=3)

    outputs = graph_circuit.get_outputs()
    out_meas_reg = graph_circuit.add_c_register(name='output measure', size=len(outputs))
    for qubit, bit in zip(outputs.values(), out_meas_reg):
        graph_circuit.Measure(qubit=qubit, bit=bit)

    copmiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    result = backend.run_circuit(circuit=copmiled_graph_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=out_meas_reg)[(1,)] == n_shots 
