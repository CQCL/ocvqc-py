import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend

from pytket_mbqc_py import GraphCircuit


def test_zero_state():
    circuit = GraphCircuit(n_physical_qubits=2, n_logical_qubits=3)

    vertex_one = circuit.add_graph_vertex(measurement_order=0)
    vertex_two = circuit.add_graph_vertex(measurement_order=1)

    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
    circuit.corrected_measure(vertex=vertex_one)

    vertex_three = circuit.add_graph_vertex(measurement_order=None)
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


def test_one_state():
    circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=3,
    )

    vertex_one = circuit.add_graph_vertex(measurement_order=0)
    vertex_two = circuit.add_graph_vertex(measurement_order=1)
    circuit.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
    circuit.corrected_measure(vertex=vertex_one, t_multiple=4)

    vertex_three = circuit.add_graph_vertex(measurement_order=None)
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
        measurement_order = 3
        control_state_h = circuit.add_graph_vertex(measurement_order)
        circuit.add_edge(vertex_one=control_state, vertex_two=control_state_h)
        circuit.corrected_measure(vertex=control_state, t_multiple=0)

        if target_value == 0:
            measurement_order = 4
        else:
            measurement_order = 6
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

    bump = control_value * 2 + target_value * 2

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

    control_state_h = circuit.add_graph_vertex(measurement_order=None)
    circuit.add_edge(vertex_one=control_state, vertex_two=control_state_h)

    circuit.add_edge(vertex_one=target_state_h, vertex_two=control_state)
    circuit.corrected_measure(vertex=target_state_h, t_multiple=0)
    circuit.corrected_measure(vertex=control_state, t_multiple=0)

    target_state_h = circuit.add_graph_vertex(measurement_order=None)
    circuit.add_edge(vertex_one=target_state, vertex_two=target_state_h)
    circuit.corrected_measure(vertex=target_state, t_multiple=0)

    circuit.corrected_measure(vertex=control_state_h, t_multiple=0)
    circuit.corrected_measure(vertex=target_state_h, t_multiple=0)

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_circuit = backend.get_compiled_circuit(circuit)
    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
    )

    output_reg = [
        circuit.vertex_reg[control_state_h][0],
        circuit.vertex_reg[target_state_h][0],
    ]
    assert result.get_counts(output_reg)[output_state] == n_shots


def test_3_q_ghz():
    graph_circuit = GraphCircuit(n_physical_qubits=5, n_logical_qubits=10)

    input_vertex = graph_circuit.add_graph_vertex(measurement_order=0)

    vertex_layer_1_1 = graph_circuit.add_graph_vertex(measurement_order=1)
    vertex_layer_1_2 = graph_circuit.add_graph_vertex(measurement_order=2)

    graph_circuit.add_edge(input_vertex, vertex_layer_1_1)
    graph_circuit.add_edge(input_vertex, vertex_layer_1_2)

    graph_circuit.corrected_measure(vertex=input_vertex)

    vertex_layer_2_1 = graph_circuit.add_graph_vertex(measurement_order=None)
    vertex_layer_2_2 = graph_circuit.add_graph_vertex(measurement_order=None)

    graph_circuit.add_edge(vertex_layer_1_1, vertex_layer_2_1)
    graph_circuit.add_edge(vertex_layer_1_1, vertex_layer_2_2)

    graph_circuit.corrected_measure(vertex=vertex_layer_1_1)

    vertex_layer_1_2_h = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(vertex_layer_1_2, vertex_layer_1_2_h)
    graph_circuit.corrected_measure(vertex=vertex_layer_1_2)

    graph_circuit.corrected_measure(vertex=vertex_layer_2_1)
    graph_circuit.corrected_measure(vertex=vertex_layer_2_2)
    graph_circuit.corrected_measure(vertex=vertex_layer_1_2_h)

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    compiled_graph_circuit = backend.get_compiled_circuit(
        circuit=graph_circuit, optimisation_level=0
    )
    n_shots = 100
    result = backend.run_circuit(
        circuit=compiled_graph_circuit,
        n_shots=n_shots,
        seed=0,
    )
    output_reg = [
        graph_circuit.vertex_reg[vertex_layer_1_2_h][0],
        graph_circuit.vertex_reg[vertex_layer_2_1][0],
        graph_circuit.vertex_reg[vertex_layer_2_2][0],
    ]
    assert abs(result.get_counts(cbits=output_reg)[(0, 0, 0)] - (n_shots / 2)) < 1.5 * (
        n_shots**0.5
    )
    assert abs(result.get_counts(cbits=output_reg)[(1, 1, 1)] - (n_shots / 2)) < 1.5 * (
        n_shots**0.5
    )


def test_2q_t_gate_example():
    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)

    graph_circuit = GraphCircuit(
        n_physical_qubits=6,
        n_logical_qubits=20,
    )

    input_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=0)

    # H[0]
    input_vertex_0_h = graph_circuit.add_graph_vertex(measurement_order=1)
    graph_circuit.add_edge(input_vertex_0, input_vertex_0_h)
    graph_circuit.corrected_measure(input_vertex_0, t_multiple=0)

    # H[0]S[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_circuit.add_edge(input_vertex_0_h, graph_vertex_0_0)
    graph_circuit.corrected_measure(input_vertex_0_h, t_multiple=2)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=4)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    input_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=3)

    # H[1]
    input_vertex_1_h = graph_circuit.add_graph_vertex(measurement_order=5)
    graph_circuit.add_edge(input_vertex_1, input_vertex_1_h)
    graph_circuit.corrected_measure(input_vertex_1, t_multiple=0)

    # CZ[0,1]H[1]H[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=6)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=0)

    graph_vertex_1_0 = graph_circuit.add_graph_vertex(measurement_order=7)

    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=8)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)

    graph_circuit.add_edge(input_vertex_1_h, graph_vertex_1_0)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_1_0)

    graph_circuit.corrected_measure(input_vertex_1_h, t_multiple=0)

    graph_vertex_1_1 = graph_circuit.add_graph_vertex(measurement_order=9)
    graph_circuit.add_edge(graph_vertex_1_0, graph_vertex_1_1)

    # H[0]
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    # H[1]
    graph_circuit.corrected_measure(graph_vertex_1_0, t_multiple=0)

    # H[0]Z[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=10)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=4)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=11)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)

    # CZ[0,1]H[0]H[1]
    graph_vertex_0_2 = graph_circuit.add_graph_vertex(measurement_order=12)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_2)

    graph_vertex_1_0 = graph_circuit.add_graph_vertex(measurement_order=13)
    graph_circuit.add_edge(graph_vertex_1_1, graph_vertex_1_0)
    graph_circuit.corrected_measure(graph_vertex_1_1, t_multiple=0)

    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=0)

    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=14)
    graph_circuit.add_edge(graph_vertex_0_2, graph_vertex_0_1)

    graph_vertex_1_1 = graph_circuit.add_graph_vertex(measurement_order=16)
    graph_circuit.add_edge(graph_vertex_1_0, graph_vertex_1_1)

    graph_circuit.add_edge(graph_vertex_0_2, graph_vertex_1_0)

    # H[0]
    graph_circuit.corrected_measure(graph_vertex_0_2, t_multiple=0)

    # H[1]
    graph_circuit.corrected_measure(graph_vertex_1_0, t_multiple=0)

    # H[0]S[0]
    graph_vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=15)
    graph_circuit.add_edge(graph_vertex_0_1, graph_vertex_0_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=2)

    # H[0]
    graph_vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=17)
    graph_circuit.add_edge(graph_vertex_0_0, graph_vertex_0_1)
    graph_circuit.corrected_measure(graph_vertex_0_0, t_multiple=0)

    # H[1]
    output_1 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(graph_vertex_1_1, output_1)
    graph_circuit.corrected_measure(graph_vertex_1_1, t_multiple=0)

    # H[0]
    output_0 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(graph_vertex_0_1, output_0)
    graph_circuit.corrected_measure(graph_vertex_0_1, t_multiple=0)

    graph_circuit.corrected_measure(output_1, t_multiple=0)
    graph_circuit.corrected_measure(output_0, t_multiple=0)

    copmiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)
    n_shots = 100
    result = backend.run_circuit(circuit=copmiled_graph_circuit, n_shots=n_shots)

    output_reg = [
        graph_circuit.vertex_reg[output_1][0],
        graph_circuit.vertex_reg[output_0][0],
    ]
    assert result.get_counts(cbits=output_reg)[(1, 0)] == n_shots


def test_1q_t_gate_example():
    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)

    graph_circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=5,
    )

    with pytest.raises(
        Exception,
        match="There is no vertex with the index 0. Existing vertices are \[\].",
    ):
        graph_circuit.add_edge(0, 1)

    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=0)

    with pytest.raises(
        Exception,
        match="There is no vertex with the index 1. Existing vertices are \[0\].",
    ):
        graph_circuit.add_edge(graph_vertex_1, 1)

    # H[0]T[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=1)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=1)

    with pytest.raises(
        Exception,
        match="Cannot add edge after measure. In particular \[0\] have been measured.",
    ):
        graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    with pytest.raises(
        Exception,
        match="Vertex 1 has already been measured and cannot be measured again.",
    ):
        graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    # H[0]T[0]S[0]Z[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=3)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=7)

    with pytest.raises(
        Exception,
        match="Measurement order must be unique. A vertex is already measured at order 3.",
    ):
        graph_circuit.add_graph_vertex(measurement_order=3)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=None)

    with pytest.raises(
        Exception,
        match="Edges must point towards output qubits.",
    ):
        graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)

    with pytest.raises(
        Exception,
        match="Vertex 3 is not an output and has no flow.",
    ):
        graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    with pytest.raises(
        Exception,
        match="An insufficient number of initialisation registers were created.",
    ):
        graph_circuit.add_graph_vertex(measurement_order=5)

    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=0)

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 100
    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    out_meas_reg = [graph_circuit.vertex_reg[graph_vertex_1][0]]
    assert result.get_counts(cbits=out_meas_reg)[(0,)] == n_shots

    ################################
    # The following compiles to X

    graph_circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=5,
    )

    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=0)

    # H[0]T[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=1)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=1)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    # H[0]T[0]S[0]Z[0]
    graph_vertex_0 = graph_circuit.add_graph_vertex(measurement_order=3)
    graph_circuit.add_edge(graph_vertex_1, graph_vertex_0)
    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=3)

    # H[0]
    graph_vertex_1 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(graph_vertex_0, graph_vertex_1)
    graph_circuit.corrected_measure(graph_vertex_0, t_multiple=0)

    graph_circuit.corrected_measure(graph_vertex_1, t_multiple=0)

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    out_meas_reg = [graph_circuit.vertex_reg[graph_vertex_1][0]]
    assert result.get_counts(cbits=out_meas_reg)[(1,)] == n_shots


def test_mismatched_ordered_measure():
    # A test where the measurement order
    # does not match the initialisation order

    graph_circuit = GraphCircuit(
        n_physical_qubits=4,
        n_logical_qubits=14,
    )

    plus_state = graph_circuit.add_graph_vertex(measurement_order=0)
    input_vertex_zero = graph_circuit.add_graph_vertex(measurement_order=2)
    graph_circuit.add_edge(plus_state, input_vertex_zero)
    graph_circuit.corrected_measure(plus_state)

    plus_state = graph_circuit.add_graph_vertex(measurement_order=1)
    input_vertex_one = graph_circuit.add_graph_vertex(measurement_order=3)
    graph_circuit.add_edge(plus_state, input_vertex_one)
    graph_circuit.corrected_measure(plus_state)

    graph_vertex_two = graph_circuit.add_graph_vertex(measurement_order=4)
    graph_vertex_three = graph_circuit.add_graph_vertex(measurement_order=5)

    graph_circuit.add_edge(input_vertex_zero, graph_vertex_two)
    graph_circuit.add_edge(input_vertex_one, graph_vertex_three)

    graph_circuit.corrected_measure(input_vertex_zero)
    graph_circuit.corrected_measure(input_vertex_one)

    graph_vertex_four = graph_circuit.add_graph_vertex(measurement_order=6)
    graph_vertex_five = graph_circuit.add_graph_vertex(measurement_order=8)

    graph_circuit.add_edge(graph_vertex_two, graph_vertex_four)
    graph_circuit.add_edge(graph_vertex_three, graph_vertex_five)
    graph_circuit.add_edge(graph_vertex_two, graph_vertex_three)

    graph_circuit.corrected_measure(graph_vertex_two)
    graph_circuit.corrected_measure(graph_vertex_three)

    graph_vertex_six = graph_circuit.add_graph_vertex(measurement_order=7)
    graph_circuit.add_edge(graph_vertex_four, graph_vertex_six)
    graph_circuit.corrected_measure(graph_vertex_four)

    graph_vertex_seven = graph_circuit.add_graph_vertex(measurement_order=9)
    graph_circuit.add_edge(graph_vertex_six, graph_vertex_seven)
    graph_circuit.corrected_measure(graph_vertex_six)

    graph_vertex_eight = graph_circuit.add_graph_vertex(measurement_order=10)
    graph_vertex_nine = graph_circuit.add_graph_vertex(measurement_order=11)

    graph_circuit.add_edge(graph_vertex_five, graph_vertex_nine)
    graph_circuit.add_edge(graph_vertex_seven, graph_vertex_eight)
    graph_circuit.add_edge(graph_vertex_five, graph_vertex_seven)

    graph_circuit.corrected_measure(graph_vertex_five)
    graph_circuit.corrected_measure(graph_vertex_seven)

    output_zero = graph_circuit.add_graph_vertex(measurement_order=12)
    graph_circuit.add_edge(graph_vertex_eight, output_zero)
    graph_circuit.corrected_measure(graph_vertex_eight)

    output_one = graph_circuit.add_graph_vertex(measurement_order=13)
    graph_circuit.add_edge(graph_vertex_nine, output_one)
    graph_circuit.corrected_measure(graph_vertex_nine)

    backend = QuantinuumBackend(
        device_name="H1-1LE", api_handler=QuantinuumAPIOffline()
    )

    output_reg = [
        graph_circuit.vertex_reg[output_zero][0],
        graph_circuit.vertex_reg[output_one][0],
    ]

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
    graph_circuit = GraphCircuit(
        n_physical_qubits=3,
        n_logical_qubits=10,
    )

    # Rotation followed by H on control to create input
    vertex_0_0 = graph_circuit.add_graph_vertex(measurement_order=0)
    vertex_0_1 = graph_circuit.add_graph_vertex(measurement_order=1)
    graph_circuit.add_edge(vertex_0_0, vertex_0_1)
    graph_circuit.corrected_measure(vertex_0_0, t_multiple=4 * input_state[0])

    # H on control
    vertex_0_2 = graph_circuit.add_graph_vertex(measurement_order=5)
    graph_circuit.add_edge(vertex_0_1, vertex_0_2)
    graph_circuit.corrected_measure(vertex_0_1, t_multiple=0)

    # Rotation followed by H on target to create input
    vertex_1_0 = graph_circuit.add_graph_vertex(measurement_order=2)
    vertex_1_1 = graph_circuit.add_graph_vertex(measurement_order=3)
    graph_circuit.add_edge(vertex_1_0, vertex_1_1)
    graph_circuit.corrected_measure(vertex_1_0, t_multiple=4 * input_state[1])

    # H on target
    vertex_1_2 = graph_circuit.add_graph_vertex(measurement_order=4)
    graph_circuit.add_edge(vertex_1_1, vertex_1_2)
    graph_circuit.corrected_measure(vertex_1_1, t_multiple=0)

    # H on target
    vertex_1_3 = graph_circuit.add_graph_vertex(measurement_order=6)
    graph_circuit.add_edge(vertex_1_2, vertex_1_3)
    graph_circuit.corrected_measure(vertex_1_2, t_multiple=0)

    # HHCX
    vertex_0_3 = graph_circuit.add_graph_vertex(measurement_order=7)
    graph_circuit.add_edge(vertex_0_2, vertex_0_3)
    graph_circuit.corrected_measure(vertex_0_2, t_multiple=0)

    vertex_1_4 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(vertex_1_3, vertex_1_4)
    graph_circuit.corrected_measure(vertex_1_3, t_multiple=0)

    # H on control
    vertex_0_4 = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(vertex_0_3, vertex_0_4)
    graph_circuit.add_edge(vertex_0_3, vertex_1_4)
    graph_circuit.corrected_measure(vertex_0_3, t_multiple=0)

    graph_circuit.corrected_measure(vertex_0_4, t_multiple=0)
    graph_circuit.corrected_measure(vertex_1_4, t_multiple=0)

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 100
    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    out_meas_reg = [
        graph_circuit.vertex_reg[vertex_0_4][0],
        graph_circuit.vertex_reg[vertex_1_4][0],
    ]
    assert result.get_counts(cbits=out_meas_reg)[output_state] == n_shots


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
    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)

    graph_circuit = GraphCircuit(n_physical_qubits=6, n_logical_qubits=1)

    vertex = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.corrected_measure(vertex=vertex)

    output_reg = [graph_circuit.vertex_reg[vertex][0]]

    compiled_circuit = backend.get_compiled_circuit(graph_circuit)
    n_shots = 100
    result = backend.run_circuit(circuit=compiled_circuit, n_shots=n_shots)
    assert result.get_counts(cbits=output_reg)[(0,)] == n_shots


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
        assert abs(result.get_counts(cbits=[cbits])[(0,)] - n_shots / 2) <= 1.5 * (
            n_shots**0.5
        )

    for cbits in reg_two:
        assert abs(result.get_counts(cbits=[cbits])[(0,)] - n_shots / 2) <= 1.5 * (
            n_shots**0.5
        )
