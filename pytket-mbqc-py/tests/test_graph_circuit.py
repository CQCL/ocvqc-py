import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
from pytket.unit_id import BitRegister, Qubit

from pytket_mbqc_py import CNOTBlocksGraphCircuit, GraphCircuit


def test_zero_state():

    circuit = GraphCircuit(n_physical_qubits=2, n_logical_qubits=3)

    vertex_one = circuit.add_graph_vertex(measurement_order=0)
    vertex_two = circuit.add_graph_vertex(measurement_order=1)

    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
    circuit.corrected_measure(vertex=vertex_one)

    vertex_three = circuit.add_graph_vertex(measurement_order=2)
    circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_three)
    circuit.corrected_measure(vertex=vertex_two)
    circuit.corrected_measure(vertex=vertex_three)

    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )
    compiled_circuit = backend.get_compiled_circuit(circuit)
    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )

    output_bit = circuit.vertex_reg[vertex_three][0]
    assert result.get_counts(cbits=[output_bit])[(0,)] == n_shots


# def test_plus_state():
#     circuit = GraphCircuit(n_physical_qubits=2, n_logical_qubits=2)

#     input_qubit, vertex_one = circuit.add_input_vertex(measurement_order=0)
#     circuit.H(input_qubit)

#     vertex_two = circuit.add_graph_vertex(measurement_order=None)

#     circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
#     circuit.corrected_measure(vertex=vertex_one)

#     output_qubits = circuit.get_outputs()

#     output_reg = BitRegister(name="output", size=1)
#     circuit.add_c_register(register=output_reg)
#     circuit.Measure(qubit=output_qubits[vertex_two], bit=output_reg[0])

#     backend = QuantinuumBackend(
#         device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
#     )
#     compiled_circuit = backend.get_compiled_circuit(circuit)
#     n_shots = 100
#     result = backend.run_circuit(
#         circuit=compiled_circuit,
#         n_shots=n_shots,
#     )

#     assert result.get_counts(output_reg)[(0,)] == 100

def test_one_state():

    circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=3,
    )

    vertex_one = circuit.add_graph_vertex(measurement_order=0)
    vertex_two = circuit.add_graph_vertex(measurement_order=1)
    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
    circuit.corrected_measure(vertex=vertex_one, t_multiple=4)

    vertex_three = circuit.add_graph_vertex(measurement_order=2)
    circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_three)
    circuit.corrected_measure(vertex=vertex_two, t_multiple=0)
    circuit.corrected_measure(vertex=vertex_three, t_multiple=0)

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_circuit = backend.get_compiled_circuit(circuit)
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=100,
    )

    output_bit = circuit.vertex_reg[vertex_three][0]
    assert result.get_counts([output_bit])[(1,)] == 100


# def test_x_gate():
#     circuit = GraphCircuit(
#         n_physical_qubits=3,
#         n_logical_qubits=3,
#     )

#     _, vertex_one = circuit.add_input_vertex(measurement_order=0)

#     vertex_two = circuit.add_graph_vertex(measurement_order=1)
#     circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
#     vertex_three = circuit.add_graph_vertex(measurement_order=None)
#     circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_three)
#     circuit.corrected_measure(vertex=vertex_one, t_multiple=0)
#     circuit.corrected_measure(vertex=vertex_two, t_multiple=4)

#     output_qubits = circuit.get_outputs()
#     output_reg = BitRegister(name="output", size=1)
#     circuit.add_c_register(register=output_reg)
#     circuit.Measure(
#         qubit=output_qubits[vertex_three],
#         bit=output_reg[0],
#     )

#     backend = QuantinuumBackend(
#         device_name="H1-1LE",
#         api_handler=QuantinuumAPIOffline(),
#     )

#     compiled_circuit = backend.get_compiled_circuit(circuit)
#     result = backend.run_circuit(
#         circuit=compiled_circuit,
#         n_shots=100,
#     )

#     assert result.get_counts(output_reg)[(1,)] == 100


@pytest.mark.parametrize(
    "input_state, output_state",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_cnot(input_state, output_state):

    control_value = input_state[0]
    target_value = input_state[1]

    circuit = GraphCircuit(
        n_physical_qubits=4,
        n_logical_qubits=20,
    )

    plus_state = circuit.add_graph_vertex(measurement_order=0)

    if control_value == 0 and target_value == 1:
        measurement_order = 4
    else:
        measurement_order = 2
    control_state = circuit.add_graph_vertex(measurement_order)
    circuit.add_edge(vertex_one=plus_state, vertex_two=control_state)
    circuit.corrected_measure(vertex=plus_state, t_multiple=0)

    plus_state = circuit.add_graph_vertex(measurement_order=1)

    if control_value == 0 and target_value == 0:
        measurement_order = 3
    elif control_value == 1 and target_value == 0:
        measurement_order = 5
    elif control_value == 0 and target_value == 1:
        measurement_order = 2
    else:
        measurement_order = 4
    target_state = circuit.add_graph_vertex(measurement_order)
    circuit.add_edge(vertex_one=plus_state, vertex_two=target_state)
    circuit.corrected_measure(vertex=plus_state, t_multiple=0)

    if control_value:
        measurement_order=3
        control_state_h = circuit.add_graph_vertex(measurement_order)
        circuit.add_edge(vertex_one=control_state, vertex_two=control_state_h)
        circuit.corrected_measure(vertex=control_state, t_multiple=0)

        if target_value == 0:
            measurement_order=4
        else:
            measurement_order=6
        control_state = circuit.add_graph_vertex(measurement_order)
        circuit.add_edge(vertex_one=control_state_h, vertex_two=control_state)
        circuit.corrected_measure(vertex=control_state_h, t_multiple=4)
        
    if target_value:
        if control_value == 0:
            measurement_order = 3
        else:
            measurement_order = 5
        target_state_h = circuit.add_graph_vertex(measurement_order)
        circuit.add_edge(vertex_one=target_state, vertex_two=target_state_h)
        circuit.corrected_measure(vertex=target_state, t_multiple=0)
        
        if control_value == 0:
            measurement_order = 5
        else:
            measurement_order = 7
        target_state = circuit.add_graph_vertex(measurement_order)
        circuit.add_edge(vertex_one=target_state_h, vertex_two=target_state)
        circuit.corrected_measure(vertex=target_state_h, t_multiple=4)
        
    bump = control_value*2 + target_value*2

    control_state_h = circuit.add_graph_vertex(measurement_order=4 + bump)
    circuit.add_edge(vertex_one=control_state, vertex_two=control_state_h)
    circuit.corrected_measure(vertex=control_state, t_multiple=0)

    target_state_h = circuit.add_graph_vertex(measurement_order=5 + bump)
    circuit.add_edge(vertex_one=target_state, vertex_two=target_state_h)
    circuit.corrected_measure(vertex=target_state, t_multiple=0)

    control_state = circuit.add_graph_vertex(measurement_order=6 + bump)
    circuit.add_edge(vertex_one=control_state_h, vertex_two=control_state)
    circuit.corrected_measure(vertex=control_state_h, t_multiple=0)

    target_state = circuit.add_graph_vertex(measurement_order=7 + bump)
    circuit.add_edge(vertex_one=target_state_h, vertex_two=target_state)

    control_state_h = circuit.add_graph_vertex(measurement_order=8 + bump)
    circuit.add_edge(vertex_one=control_state, vertex_two=control_state_h)

    circuit.add_edge(vertex_one=target_state_h, vertex_two=control_state)
    circuit.corrected_measure(vertex=target_state_h, t_multiple=0)
    circuit.corrected_measure(vertex=control_state, t_multiple=0)

    target_state_h = circuit.add_graph_vertex(measurement_order=9 + bump)
    circuit.add_edge(vertex_one=target_state, vertex_two=target_state_h)
    circuit.corrected_measure(vertex=target_state, t_multiple=0)

    circuit.corrected_measure(vertex=control_state_h, t_multiple=0)
    circuit.corrected_measure(vertex=target_state_h, t_multiple=0)
        
    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_circuit = backend.get_compiled_circuit(circuit)
    n_shots=100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )

    output_reg = [circuit.vertex_reg[control_state_h][0], circuit.vertex_reg[target_state_h][0]]
    assert result.get_counts(output_reg)[output_state] == n_shots


# @pytest.mark.parametrize(
#     "input_state, output_state",
#     [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
# )
# def test_cnot(input_state, output_state):
#     circuit = GraphCircuit(
#         n_physical_qubits=5,
#         n_logical_qubits=8,
#     )

#     target_qubit, vertex_one = circuit.add_input_vertex(measurement_order=0)
#     if input_state[1]:
#         circuit.X(target_qubit)

#     vertex_two = circuit.add_graph_vertex(measurement_order=1)
#     circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)

#     control_qubit, vertex_three = circuit.add_input_vertex(measurement_order=2)
#     if input_state[0]:
#         circuit.X(control_qubit)

#     vertex_four = circuit.add_graph_vertex(measurement_order=3)
#     circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_four)
#     circuit.corrected_measure(vertex=vertex_one, t_multiple=0)

#     vertex_five = circuit.add_graph_vertex(measurement_order=4)
#     circuit.add_edge(vertex_one=vertex_three, vertex_two=vertex_five)

#     vertex_six = circuit.add_graph_vertex(measurement_order=5)
#     circuit.add_edge(vertex_one=vertex_four, vertex_two=vertex_six)
#     circuit.corrected_measure(vertex=vertex_two, t_multiple=0)

#     vertex_seven = circuit.add_graph_vertex(measurement_order=None)
#     circuit.add_edge(vertex_one=vertex_five, vertex_two=vertex_seven)
#     circuit.corrected_measure(vertex=vertex_three, t_multiple=0)

#     vertex_eight = circuit.add_graph_vertex(measurement_order=None)
#     circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_eight)

#     circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_seven)

#     circuit.corrected_measure(vertex=vertex_four, t_multiple=0)
#     circuit.corrected_measure(vertex=vertex_five, t_multiple=0)
#     circuit.corrected_measure(vertex=vertex_six, t_multiple=0)

#     output_qubits = circuit.get_outputs()
#     output_reg = BitRegister(name="output", size=2)
#     circuit.add_c_register(register=output_reg)
#     circuit.Measure(qubit=output_qubits[vertex_seven], bit=output_reg[0])
#     circuit.Measure(qubit=output_qubits[vertex_eight], bit=output_reg[1])

#     api_offline = QuantinuumAPIOffline()
#     backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
#     compiled_circuit = backend.get_compiled_circuit(circuit)

#     n_shots = 100
#     result = backend.run_circuit(
#         circuit=compiled_circuit,
#         n_shots=n_shots,
#     )

#     assert result.get_counts(output_reg)[output_state] == n_shots


# @pytest.mark.parametrize(
#     "input_state, output_state, n_layers, n_logical_qubits",
#     [
#         ((1, 0), (1, 1), 1, 6),
#         ((1, 1), (1, 1), 2, 10),
#         ((1, 1, 0), (1, 1, 1), 2, 19),
#         ((0, 1, 0), (0, 1, 1), 3, 27),
#         ((1, 1, 1), (1, 1, 1), 4, 35),
#         ((0, 0, 1), (0, 0, 1), 1, 11),
#         ((0, 1, 1), (0, 1, 0), 1, 11),
#     ],
# )
# def test_cnot_block(input_state, output_state, n_layers, n_logical_qubits):
#     n_physical_qubits = 15

#     circuit = CNOTBlocksGraphCircuit(
#         n_physical_qubits=n_physical_qubits,
#         input_state=input_state,
#         n_layers=n_layers,
#         n_logical_qubits=n_logical_qubits,
#     )

#     output_vertex_quibts = circuit.get_outputs()
#     output_reg = BitRegister(name="output", size=len(output_vertex_quibts))
#     circuit.add_c_register(register=output_reg)
#     for i, qubit in enumerate(output_vertex_quibts.values()):
#         circuit.Measure(qubit=qubit, bit=output_reg[i])
#     assert circuit.output_state == output_state

#     api_offline = QuantinuumAPIOffline()
#     backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
#     compiled_circuit = backend.get_compiled_circuit(circuit)

#     n_shots = 100
#     result = backend.run_circuit(
#         circuit=compiled_circuit,
#         n_shots=n_shots,
#     )
#     assert result.get_counts(output_reg)[output_state] == n_shots


# @pytest.mark.high_compute
# def test_large_cnot_block():
#     input_state = (0, 1, 1, 0)
#     output_state = (0, 1, 0, 1)
#     n_layers = 3
#     n_physical_qubits = 20

#     circuit = CNOTBlocksGraphCircuit(
#         n_physical_qubits=n_physical_qubits,
#         input_state=input_state,
#         n_layers=n_layers,
#         n_logical_qubits=40,
#     )

#     output_vertex_quibts = circuit.get_outputs()
#     output_reg = BitRegister(name="output", size=len(output_vertex_quibts))
#     circuit.add_c_register(register=output_reg)
#     for i, qubit in enumerate(output_vertex_quibts.values()):
#         circuit.Measure(qubit=qubit, bit=output_reg[i])
#     assert circuit.output_state == output_state

#     api_offline = QuantinuumAPIOffline()
#     backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
#     compiled_circuit = backend.get_compiled_circuit(circuit)

#     n_shots = 100
#     result = backend.run_circuit(
#         circuit=compiled_circuit,
#         n_shots=n_shots,
#     )
#     assert result.get_counts(output_reg)[output_state] == n_shots


def test_3_q_ghz():
    graph_circuit = GraphCircuit(n_physical_qubits=5, n_logical_qubits=5)

    input_quibt, input_vertex = graph_circuit.add_input_vertex(measurement_order=0)

    graph_circuit.H(input_quibt)

    vertex_layer_1_1 = graph_circuit.add_graph_vertex(measurement_order=1)
    vertex_layer_1_2 = graph_circuit.add_graph_vertex(measurement_order=None)

    graph_circuit.add_edge(input_vertex, vertex_layer_1_1)
    graph_circuit.add_edge(input_vertex, vertex_layer_1_2)

    graph_circuit.corrected_measure(vertex=input_vertex)

    vertex_layer_2_1 = graph_circuit.add_graph_vertex(measurement_order=None)
    vertex_layer_2_2 = graph_circuit.add_graph_vertex(measurement_order=None)

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


# @pytest.mark.parametrize(
#     "input_state, output_state",
#     [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
# )
# def test_cnot_early_measure(input_state, output_state):
#     circuit = GraphCircuit(
#         n_physical_qubits=3,
#         n_logical_qubits=8,
#     )

#     target_qubit, vertex_one = circuit.add_input_vertex(measurement_order=0)
#     if input_state[1]:
#         circuit.X(target_qubit)

#     vertex_two = circuit.add_graph_vertex(measurement_order=1)
#     circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)

#     circuit.corrected_measure(vertex=vertex_one, t_multiple=0)

#     control_qubit, vertex_three = circuit.add_input_vertex(measurement_order=2)
#     if input_state[0]:
#         circuit.X(control_qubit)

#     vertex_four = circuit.add_graph_vertex(measurement_order=3)
#     circuit.add_edge(vertex_one=vertex_two, vertex_two=vertex_four)

#     circuit.corrected_measure(vertex=vertex_two, t_multiple=0)

#     vertex_five = circuit.add_graph_vertex(measurement_order=4)
#     circuit.add_edge(vertex_one=vertex_three, vertex_two=vertex_five)

#     circuit.corrected_measure(vertex=vertex_three, t_multiple=0)

#     vertex_six = circuit.add_graph_vertex(measurement_order=5)
#     circuit.add_edge(vertex_one=vertex_four, vertex_two=vertex_six)

#     circuit.corrected_measure(vertex=vertex_four, t_multiple=0)

#     vertex_seven = circuit.add_graph_vertex(measurement_order=None)
#     circuit.add_edge(vertex_one=vertex_five, vertex_two=vertex_seven)

#     circuit.corrected_measure(vertex=vertex_five, t_multiple=0)

#     vertex_eight = circuit.add_graph_vertex(measurement_order=None)
#     circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_eight)

#     circuit.add_edge(vertex_one=vertex_six, vertex_two=vertex_seven)

#     circuit.corrected_measure(vertex=vertex_six, t_multiple=0)

#     output_qubits = circuit.get_outputs()
#     output_reg = BitRegister(name="output", size=2)
#     circuit.add_c_register(register=output_reg)
#     circuit.Measure(qubit=output_qubits[vertex_seven], bit=output_reg[0])
#     circuit.Measure(qubit=output_qubits[vertex_eight], bit=output_reg[1])

#     api_offline = QuantinuumAPIOffline()
#     backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
#     compiled_circuit = backend.get_compiled_circuit(circuit)

#     n_shots = 100
#     result = backend.run_circuit(
#         circuit=compiled_circuit,
#         n_shots=n_shots,
#     )

#     assert result.get_counts(output_reg)[output_state] == n_shots


@pytest.mark.high_compute
def test_2q_t_gate_example():
    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)

    graph_circuit = GraphCircuit(
        n_physical_qubits=6,
        n_logical_qubits=16,
    )

    _, input_vertex_0 = graph_circuit.add_input_vertex(measurement_order=0)

    # H[0]S[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=1)
    graph_circuit.add_edge(input_vertex_0, graph_vertex_0_0)
    graph_circuit.corrected_measure(input_vertex_0, t_multiple=2)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    _, input_vertex_1 = graph_circuit.add_input_vertex(measurement_order=3)

    # CZ[0,1]H[1]H[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=4)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=0)

    graph_vertex_1_0 = graph_circuit.add_graph_vertex(measurement_order=5)

    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=6)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)

    graph_circuit.add_edge(input_vertex_1, graph_vertex_1_0)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_1_0)

    graph_circuit.corrected_measure(input_vertex_1, t_multiple=0)

    graph_vertex_1_1 = graph_circuit.add_graph_vertex(measurement_order=7)
    graph_circuit.add_edge(graph_vertex_1_0, graph_vertex_1_1)

    # H[0]
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    # H[1]
    graph_circuit.corrected_measure(graph_vertex_1_0, t_multiple=0)

    # H[0]Z[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=8)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=4)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=9)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)

    # CZ[0,1]H[0]H[1]
    graph_vertex_0_2 = graph_circuit.add_graph_vertex(measurement_order=10)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_2)

    graph_vertex_1_0 = graph_circuit.add_graph_vertex(measurement_order=11)
    graph_circuit.add_edge(graph_vertex_1_1, graph_vertex_1_0)
    graph_circuit.corrected_measure(graph_vertex_1_1, t_multiple=0)

    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=0)

    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=12)
    graph_circuit.add_edge(graph_vertex_0_2, graph_vertex_0_1)

    graph_vertex_1_1 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(graph_vertex_1_0, graph_vertex_1_1)

    graph_circuit.add_edge(graph_vertex_0_2, graph_vertex_1_0)

    # H[0]
    graph_circuit.corrected_measure(graph_vertex_0_2, t_multiple=0)

    # H[1]
    graph_circuit.corrected_measure(graph_vertex_1_0, t_multiple=0)

    # H[0]S[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=13)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=2)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    outputs = graph_circuit.get_outputs()
    out_meas_reg = graph_circuit.add_c_register(
        name="output measure", size=len(outputs)
    )
    for qubit, bit in zip(outputs.values(), out_meas_reg):
        graph_circuit.Measure(qubit=qubit, bit=bit)

    copmiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)
    n_shots = 1000
    result = backend.run_circuit(circuit=copmiled_graph_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=out_meas_reg)[(1, 0)] == n_shots


def test_1q_t_gate_example():
    ################################
    # The following compiles to I

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)

    graph_circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=5,
    )

    _, input_vertex_0 = graph_circuit.add_input_vertex(measurement_order=0)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=1)
    graph_circuit.add_edge(input_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(input_vertex_0, t_multiple=0)

    # H[0]T[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=1)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=3)
    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    # H[0]T[0]S[0]Z[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=7)

    outputs = graph_circuit.get_outputs()
    out_meas_reg = graph_circuit.add_c_register(
        name="output measure", size=len(outputs)
    )
    for qubit, bit in zip(outputs.values(), out_meas_reg):
        graph_circuit.Measure(qubit=qubit, bit=bit)

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 1000
    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=out_meas_reg)[(0,)] == n_shots

    ################################
    # The following compiles to X

    graph_circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=5,
    )

    _, input_vertex_0 = graph_circuit.add_input_vertex(measurement_order=0)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=1)
    graph_circuit.add_edge(input_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(input_vertex_0, t_multiple=0)

    # H[0]T[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=1)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=3)
    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    # H[0]T[0]S[0]Z[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=3)

    outputs = graph_circuit.get_outputs()
    out_meas_reg = graph_circuit.add_c_register(
        name="output measure", size=len(outputs)
    )
    for qubit, bit in zip(outputs.values(), out_meas_reg):
        graph_circuit.Measure(qubit=qubit, bit=bit)

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=out_meas_reg)[(1,)] == n_shots


def test_mismatched_ordered_measure():
    # A test where the measurement order
    # does not match the initialisation order

    graph_circuit = GraphCircuit(
        n_physical_qubits=4,
        n_logical_qubits=10,
    )

    _, input_vertex_zero = graph_circuit.add_input_vertex(measurement_order=0)
    _, input_vertex_one = graph_circuit.add_input_vertex(measurement_order=1)

    graph_vertex_two = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_vertex_three = graph_circuit.add_graph_vertex(measurement_order=3)

    graph_circuit.add_edge(input_vertex_zero, graph_vertex_two)
    graph_circuit.add_edge(input_vertex_one, graph_vertex_three)

    graph_circuit.corrected_measure(input_vertex_zero)
    graph_circuit.corrected_measure(input_vertex_one)

    graph_vertex_four = graph_circuit.add_graph_vertex(measurement_order=4)
    graph_vertex_five = graph_circuit.add_graph_vertex(measurement_order=6)

    graph_circuit.add_edge(graph_vertex_two, graph_vertex_four)
    graph_circuit.add_edge(graph_vertex_three, graph_vertex_five)
    graph_circuit.add_edge(graph_vertex_two, graph_vertex_three)

    graph_circuit.corrected_measure(graph_vertex_two)
    graph_circuit.corrected_measure(graph_vertex_three)

    graph_vertex_six = graph_circuit.add_graph_vertex(measurement_order=5)
    graph_circuit.add_edge(graph_vertex_four, graph_vertex_six)
    graph_circuit.corrected_measure(graph_vertex_four)

    graph_vertex_seven = graph_circuit.add_graph_vertex(measurement_order=7)
    graph_circuit.add_edge(graph_vertex_six, graph_vertex_seven)
    graph_circuit.corrected_measure(graph_vertex_six)

    graph_vertex_eight = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_vertex_nine = graph_circuit.add_graph_vertex(measurement_order=None)

    graph_circuit.add_edge(graph_vertex_five, graph_vertex_nine)
    graph_circuit.add_edge(graph_vertex_seven, graph_vertex_eight)
    graph_circuit.add_edge(graph_vertex_five, graph_vertex_seven)

    graph_circuit.corrected_measure(graph_vertex_five)
    graph_circuit.corrected_measure(graph_vertex_seven)

    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )

    output_qubit_dict = graph_circuit.get_outputs()
    output_reg = graph_circuit.add_c_register(name="output meas", size=2)
    graph_circuit.Measure(output_qubit_dict[8], output_reg[0])
    graph_circuit.Measure(output_qubit_dict[9], output_reg[1])

    compiled_circuit = backend.get_compiled_circuit(graph_circuit)
    n_shots = 100
    result = backend.run_circuit(circuit=compiled_circuit, n_shots=n_shots)
    # This circuit does not implemented the identity, but in the measurement
    # and initialisation basis used the ideal outcome is (0, 0)
    assert result.get_counts(cbits=output_reg)[(0, 0)] == n_shots


@pytest.mark.parametrize(
    "input_state, output_state",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_cnot_entangled_output(input_state, output_state):
    graph_circuit = GraphCircuit(n_physical_qubits=4, n_logical_qubits=4)

    qubit_zero, vertex_zero = graph_circuit.add_input_vertex(measurement_order=0)

    if input_state[0]:
        graph_circuit.X(qubit_zero)

    graph_circuit.H(qubit_zero)

    vertex_one = graph_circuit.add_graph_vertex(measurement_order=None)

    with pytest.raises(
        Exception,
        match="Vertex 1 does not have a measurement order and cannot be measured.",
    ):
        graph_circuit.corrected_measure(vertex_one)

    graph_circuit.add_edge(
        vertex_one=vertex_zero,
        vertex_two=vertex_one,
    )

    graph_circuit.corrected_measure(vertex_zero)

    with pytest.raises(
        Exception,
        match="Vertex 0 has already been measured and cannot be measured again.",
    ):
        graph_circuit.corrected_measure(vertex_zero)

    with pytest.raises(
        Exception,
        match="Cannot add edge after measure. In particular \[0\] have been measured.",
    ):
        graph_circuit.add_edge(
            vertex_one=vertex_zero,
            vertex_two=vertex_one,
        )

    with pytest.raises(
        Exception,
        match="Measurement order must be unique. A vertex is already measured at order 0.",
    ):
        graph_circuit.add_input_vertex(measurement_order=0)

    qubit_two, vertex_two = graph_circuit.add_input_vertex(measurement_order=1)

    if input_state[1]:
        graph_circuit.X(qubit_two)

    with pytest.raises(
        Exception,
        match="Too many vertex registers, 4, were created. Consider setting n_logical_qubits=3 upon initialising this class.",
    ):
        graph_circuit.get_outputs()

    vertex_three = graph_circuit.add_graph_vertex(measurement_order=None)

    with pytest.raises(
        Exception,
        match="An insufficient number of initialisation registers were created. A new vertex cannot be added.",
    ):
        graph_circuit.add_graph_vertex(measurement_order=None)

    with pytest.raises(
        Exception,
        match="Please ensure that edges point towards unmeasured qubits. In this case 3 is an output but 2 is not.",
    ):
        graph_circuit.add_edge(
            vertex_one=vertex_three,
            vertex_two=vertex_two,
        )

    with pytest.raises(
        Exception,
        match="There is no vertex with the index 4. Existing vertices are \[0, 1, 2, 3\].",
    ):
        graph_circuit.add_edge(
            vertex_one=4,
            vertex_two=vertex_two,
        )

    with pytest.raises(
        Exception,
        match="There is no vertex with the index 5. Existing vertices are \[0, 1, 2, 3\].",
    ):
        graph_circuit.add_edge(
            vertex_one=vertex_three,
            vertex_two=5,
        )

    graph_circuit.add_edge(
        vertex_one=vertex_two,
        vertex_two=vertex_three,
    )

    graph_circuit.add_edge(
        vertex_one=vertex_one,
        vertex_two=vertex_three,
    )

    with pytest.raises(
        Exception,
        match="Vertices \[2\] have a measurement order but have not been measured. Please measure them, or set their order to None.",
    ):
        output_dict = graph_circuit.get_outputs()

    graph_circuit.corrected_measure(vertex_two)
    output_dict = graph_circuit.get_outputs()

    graph_circuit.H(output_dict[3])

    output_meas_reg = graph_circuit.add_c_register(name="output measure", size=2)
    graph_circuit.Measure(output_dict[1], output_meas_reg[0])
    graph_circuit.Measure(output_dict[3], output_meas_reg[1])

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_circuit = backend.get_compiled_circuit(circuit=graph_circuit)
    n_shots = 100
    result = backend.run_circuit(circuit=compiled_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=output_meas_reg)[output_state] == n_shots


def test_error_messages():
    graph_circuit = GraphCircuit(n_physical_qubits=6, n_logical_qubits=6)

    vertex_zero = graph_circuit.add_graph_vertex(measurement_order=0)
    vertex_one = graph_circuit.add_graph_vertex(measurement_order=3)
    vertex_two = graph_circuit.add_graph_vertex(measurement_order=2)
    vertex_three = graph_circuit.add_graph_vertex(measurement_order=4)
    vertex_four = graph_circuit.add_graph_vertex(measurement_order=5)
    vertex_five = graph_circuit.add_graph_vertex(measurement_order=7)

    with pytest.raises(
        Exception,
        match="1 is measured after 0. The respective measurements orders are 3 and 0. Cannot add edge into the past.",
    ):
        graph_circuit.add_edge(
            vertex_one=vertex_one,
            vertex_two=vertex_zero,
        )

    graph_circuit.add_edge(
        vertex_one=vertex_zero,
        vertex_two=vertex_three,
    )
    graph_circuit.add_edge(
        vertex_one=vertex_zero,
        vertex_two=vertex_one,
    )

    with pytest.raises(
        Exception,
        match="Adding the edge \(2, 1\) does not define a valid flow. In particular \[0\] are neighbours of 1 but are in the past of 2. As 1 would become the flow of 2 all of the neighbours of 1 must be in the past of 2.",
    ):
        graph_circuit.add_edge(
            vertex_one=vertex_two,
            vertex_two=vertex_one,
        )

    graph_circuit.add_edge(
        vertex_one=vertex_two,
        vertex_two=vertex_four,
    )

    with pytest.raises(
        Exception,
        match="Adding the edge \(0, 4\) does not define a valid flow. In particular 4 is the flow of \[2\], some of which are measured after 0.",
    ):
        graph_circuit.add_edge(
            vertex_one=vertex_zero,
            vertex_two=vertex_four,
        )

    # with pytest.raises(
    #     Exception,
    #     match="Vertex 4 has no flow and cannot be measured.",
    # ):
    #     graph_circuit.corrected_measure(vertex=vertex_four)

    graph_circuit.add_edge(
        vertex_one=vertex_four,
        vertex_two=vertex_five,
    )

    graph_circuit.corrected_measure(vertex=vertex_zero)

    with pytest.raises(
        Exception,
        match="The vertices \[1, 2, 3\] are ordered to be measured before vertex 4, but are unmeasured.",
    ):
        graph_circuit.corrected_measure(vertex=vertex_four)

    with pytest.raises(
        Exception,
        match="Vertex 2 has order 2 but there is no vertex with order 1.",
    ):
        graph_circuit.corrected_measure(vertex=vertex_two)


def test_single_unmeasured_vertex():
    graph_circuit = GraphCircuit(n_physical_qubits=6, n_logical_qubits=1)

    graph_circuit.add_graph_vertex(measurement_order=None)
    assert graph_circuit.get_outputs() == {0: Qubit(1)}


def test_too_few_qubits():
    circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=2,
    )
    reg = circuit.add_c_register(
        name="my_reg",
        size=3,
    )
    circuit.get_qubit(reg[0])
    circuit.get_qubit(reg[1])

    with pytest.raises(
        Exception,
        match="You have run out of qubits.",
    ):
        circuit.get_qubit(reg[2])

    with pytest.raises(
        Exception,
        match="There are no unused qubits which can be used to generate randomness.",
    ):
        circuit.populate_random_bits(
            bit_list=reg[2],
        )


def test_randomness_generation():
    circuit = GraphCircuit(
        n_physical_qubits=3,
        n_logical_qubits=2,
    )

    reg_one = circuit.add_c_register(
        name="my_first_random_reg",
        size=16,
    )
    circuit.populate_random_bits(bit_list=reg_one.to_list())

    reg_two = circuit.add_c_register(
        name="my_second_random_reg",
        size=32,
    )
    circuit.populate_random_bits(bit_list=reg_two.to_list())

    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )
    n_shots = 1000

    compiled_circuit = backend.get_compiled_circuit(circuit)
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
        seed=0,
    )

    for cbits in reg_one:
        assert abs(result.get_counts(cbits=[cbits])[(0,)] - n_shots / 2) < 35

    for cbits in reg_two:
        assert abs(result.get_counts(cbits=[cbits])[(0,)] - n_shots / 2) < 35
