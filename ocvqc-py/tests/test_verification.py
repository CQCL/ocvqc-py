import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend

from ocvqc_py import GraphCircuit


class VerifiedCNOT(GraphCircuit):
    def __init__(self, control: bool, target: bool) -> None:
        self.vertex_is_dummy_list = [
            [
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
            ],
            [
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
            ],
        ]

        super().__init__(
            n_physical_qubits=3,
            n_logical_qubits=8,
            vertex_is_dummy_list=self.vertex_is_dummy_list,
        )

        # Rotation followed by H on control
        v_0_0 = self.add_graph_vertex(measurement_order=0)
        v_0_1 = self.add_graph_vertex(measurement_order=1)
        self.add_edge(vertex_one=v_0_0, vertex_two=v_0_1)
        self.corrected_measure(vertex=v_0_0, t_multiple=4 * control)

        # H on control
        v_0_2 = self.add_graph_vertex(measurement_order=3)
        self.add_edge(vertex_one=v_0_1, vertex_two=v_0_2)
        self.corrected_measure(vertex=v_0_1, t_multiple=0)

        # rotation followed by control on target
        v_1_0 = self.add_graph_vertex(measurement_order=2)
        v_1_1 = self.add_graph_vertex(measurement_order=4)
        self.add_edge(vertex_one=v_1_0, vertex_two=v_1_1)
        self.corrected_measure(vertex=v_1_0, t_multiple=4 * target)

        # HHCX
        v_0_3 = self.add_graph_vertex(measurement_order=5)
        self.add_edge(vertex_one=v_0_2, vertex_two=v_0_3)
        self.corrected_measure(vertex=v_0_2, t_multiple=0)

        v_1_2 = self.add_graph_vertex(measurement_order=None)
        self.add_edge(vertex_one=v_1_1, vertex_two=v_1_2)
        self.corrected_measure(vertex=v_1_1, t_multiple=0)

        v_0_4 = self.add_graph_vertex(measurement_order=None)
        self.add_edge(vertex_one=v_0_3, vertex_two=v_0_4)
        self.add_edge(vertex_one=v_0_3, vertex_two=v_1_2)

        # H on control
        self.corrected_measure(vertex=v_0_3, t_multiple=0)

        # Output measures
        self.corrected_measure(vertex=v_0_4, t_multiple=0)
        self.corrected_measure(vertex=v_1_2, t_multiple=0)

        self.out_reg = [
            self.vertex_reg[v_0_4][0],
            self.vertex_reg[v_1_2][0],
        ]


@pytest.mark.parametrize(
    "input_state, output_state",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_verified_cnot(input_state, output_state):
    control, target = input_state

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    graph_circuit = VerifiedCNOT(control=control, target=target)
    compiled_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 100
    result = backend.run_circuit(circuit=compiled_circuit, n_shots=n_shots)

    # Test that if this is a computation round then the output is as expected.
    # Get the results of the output register, and the is test flag.
    output_count = result.get_counts(
        cbits=[graph_circuit.is_test_bit] + graph_circuit.out_reg
    )
    # Check all outputs corresponding to computation rounds are as expected.
    for output in list(output_count.keys()):
        if output[0] == 0:
            assert (output[1], output[2]) == output_state

    # Test that the registers specifying the vertex type are as expected.
    # Get the bits specifying the vertex type, and the test round flag bit.
    vertex_is_dummy_count = result.get_counts(
        cbits=[register[5] for register in graph_circuit.vertex_reg]
        + [graph_circuit.is_test_bit]
    )
    # For each set of vertex type bits:
    #   - if it is a computation round check that they are all 0
    #   - if it is a test round check they are one of the allowed colours.
    for vertex_is_dummy_int in list(vertex_is_dummy_count.keys()):
        if vertex_is_dummy_int[-1] == 0:
            assert all(bit == 0 for bit in vertex_is_dummy_int[:-1])
        else:
            vertex_is_dummy = [bool(bit) for bit in vertex_is_dummy_int[:-1]]
            assert vertex_is_dummy in graph_circuit.vertex_is_dummy_list

    # Check that all of the vertex types list you expect to see are present.
    for vertex_is_dummy in graph_circuit.vertex_is_dummy_list:
        vertex_is_dummy_int = tuple(int(bit) for bit in vertex_is_dummy) + tuple([1])
        assert vertex_is_dummy_int in list(vertex_is_dummy_count.keys())
    assert tuple([0] * (len(graph_circuit.vertex_is_dummy_list[0]) + 1)) in list(
        vertex_is_dummy_count.keys()
    )

    # Check that all test bits are 0 when performing a test round
    for register in graph_circuit.vertex_reg:
        for shot in result.get_shots(
            cbits=register.to_list() + [graph_circuit.is_test_bit]
        ):
            # If this vertex is a compute vertex,
            # and this is a test round, then
            # the measurement outcome should be 0
            if (not shot[5]) and (shot[8]):
                assert not shot[0]

    # Check roughly half of the shots are tests
    assert abs(
        result.get_counts(cbits=[graph_circuit.is_test_bit])[(0,)] - (n_shots / 2)
    ) < 1.5 * (n_shots**0.5)


def test_utility_methods():
    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )
    n_shots = 1000

    with pytest.raises(
        Exception,
        match="There must be a colour for each of the logical qubits. In this case there are 3 logical qubits and 4 colours.",
    ):
        graph_circuit = GraphCircuit(
            n_physical_qubits=2,
            n_logical_qubits=3,
            vertex_is_dummy_list=[
                [True, False, True, False],
                [False, True, False],
            ],
        )

    with pytest.raises(
        Exception,
        match="The vertices \[1\] are never test qubits.",
    ):
        graph_circuit = GraphCircuit(
            n_physical_qubits=2,
            n_logical_qubits=3,
            vertex_is_dummy_list=[
                [False, True, False],
            ],
        )

    with pytest.raises(
        Exception,
        match="You can only use 0 or two colours.",
    ):
        graph_circuit = GraphCircuit(
            n_physical_qubits=2,
            n_logical_qubits=3,
            vertex_is_dummy_list=[
                [False, False, False],
            ],
        )

    with pytest.raises(
        Exception,
        match="Vertex 0 and vertex 1 have the same colour.",
    ):
        graph_circuit = GraphCircuit(
            n_physical_qubits=2,
            n_logical_qubits=3,
            vertex_is_dummy_list=[
                [False, False, True],
                [False, True, False],
            ],
        )

        vertex_one = graph_circuit.add_graph_vertex(measurement_order=0)
        vertex_two = graph_circuit.add_graph_vertex(measurement_order=1)

        graph_circuit.add_edge(vertex_one, vertex_two)

    graph_circuit = GraphCircuit(
        n_physical_qubits=2,
        n_logical_qubits=3,
        vertex_is_dummy_list=[
            [True, False, True],
            [False, True, False],
        ],
    )

    vertex_one = graph_circuit.add_graph_vertex(measurement_order=0)
    vertex_two = graph_circuit.add_graph_vertex(measurement_order=1)

    graph_circuit.add_edge(vertex_one, vertex_two)
    graph_circuit.corrected_measure(vertex=vertex_one, t_multiple=4)

    vertex_three = graph_circuit.add_graph_vertex(measurement_order=None)
    graph_circuit.add_edge(vertex_two, vertex_three)
    graph_circuit.corrected_measure(vertex=vertex_two, t_multiple=0)

    with pytest.raises(
        Exception,
        match="Vertices \[2\] are output vertices but have not been measured.",
    ):
        compiled_circuit = backend.get_compiled_circuit(circuit=graph_circuit)
        result = backend.run_circuit(circuit=compiled_circuit, n_shots=n_shots)
        graph_circuit.get_output_result(result=result)

    graph_circuit.corrected_measure(vertex=vertex_three, t_multiple=0)

    compiled_circuit = backend.get_compiled_circuit(circuit=graph_circuit)
    result = backend.run_circuit(circuit=compiled_circuit, n_shots=n_shots)

    assert list(graph_circuit.get_output_result(result=result).get_counts().keys()) == [
        (1,)
    ]

    assert graph_circuit.get_failure_rate(result=result) == 0.0


@pytest.mark.parametrize(
    "input_state, output_state",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_verified_cnot_utilities(input_state, output_state):
    control, target = input_state

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    graph_circuit = VerifiedCNOT(control=control, target=target)
    compiled_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 100
    result = backend.run_circuit(circuit=compiled_circuit, n_shots=n_shots)

    assert graph_circuit.get_failure_rate(result=result) == 0.0

    assert list(
        graph_circuit.get_output_result(result=result)
        .get_counts(cbits=graph_circuit.out_reg)
        .keys()
    ) == [output_state]

    assert abs(
        graph_circuit.get_output_result(result=result).get_counts(
            cbits=graph_circuit.out_reg
        )[output_state]
        - (n_shots / 2)
    ) < 1.5 * (n_shots**0.5)
