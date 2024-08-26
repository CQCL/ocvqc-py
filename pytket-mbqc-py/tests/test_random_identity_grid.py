import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend

from pytket_mbqc_py import RandomIdentityGraph


@pytest.mark.parametrize(
    "input_string, n_layers",
    [
        ((0, 0), 2),
        ((0, 1), 2),
        ((1, 0), 2),
        ((1, 1), 2),
        ((1, 1, 1), 2),
        ((1, 1, 0), 4),
        ((0, 1, 0), 6),
        ((1, 0, 1, 0), 2),
    ],
)
def test_cnot_grid(input_string, n_layers):
    graph_circuit = graph_circuit = RandomIdentityGraph(
        n_layers=n_layers, input_string=input_string
    )

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 100
    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    counts = graph_circuit.get_output_result(result=result).get_counts()
    assert counts[input_string] == n_shots
