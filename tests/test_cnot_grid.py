import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
from pytket.passes import DecomposeClassicalExp
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

from ocvqc_py import CNOTBlock


@pytest.mark.parametrize(
    "input_string, n_layers, output_state",
    [
        ((0, 0), 1, (0, 0)),
        ((0, 1), 1, (0, 1)),
        ((1, 0), 1, (1, 1)),
        ((1, 1), 1, (1, 0)),
        ((1, 1, 1), 2, (1, 1, 0)),
        ((1, 1, 0), 3, (1, 0, 1)),
        ((0, 1, 0), 4, (0, 1, 0)),
        ((1, 0, 1, 0), 2, (1, 0, 0, 0)),
    ],
)
def test_cnot_grid(input_string, n_layers, output_state):
    graph_circuit = CNOTBlock(
        input_string=input_string,
        n_layers=n_layers,
    )

    assert graph_circuit.ideal_outcome == output_state

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 100
    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    counts = graph_circuit.get_output_result(result=result).get_counts()
    assert counts[output_state] == n_shots


@pytest.mark.parametrize(
    "input_string, n_layers, output_state",
    [
        ((0, 0), 1, (0, 0)),
        ((0, 1), 1, (0, 1)),
        ((1, 0), 1, (1, 1)),
        ((1, 1), 1, (1, 0)),
        ((1, 1, 1), 2, (1, 1, 0)),
        ((1, 1, 0), 3, (1, 0, 1)),
        ((0, 1, 0), 4, (0, 1, 0)),
        ((1, 0, 1, 0), 2, (1, 0, 0, 0)),
    ],
)
def test_cnot_grid_verified(input_string, n_layers, output_state):
    graph_circuit = CNOTBlock(
        input_string=input_string,
        n_layers=n_layers,
        verify=True,
    )

    n_shots = 100
    assert graph_circuit.ideal_outcome == output_state

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    counts = graph_circuit.get_output_result(result=result).get_counts()
    assert counts[output_state] == counts.total()
    assert graph_circuit.get_failure_rate(result) == 0

    qasm_graph_circuit_str = circuit_to_qasm_str(
        graph_circuit, header="hqslib1", maxwidth=256
    )
    qasm_graph_circuit = circuit_from_qasm_str(qasm_graph_circuit_str, maxwidth=256)

    DecomposeClassicalExp().apply(qasm_graph_circuit)

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=qasm_graph_circuit)

    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    counts = graph_circuit.get_output_result(result=result).get_counts()
    assert counts[output_state] == counts.total()
    assert graph_circuit.get_failure_rate(result) == 0
