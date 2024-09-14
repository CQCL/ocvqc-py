import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
from pytket.passes import DecomposeClassicalExp

from pytket_mbqc_py import TwoQubitGrover


@pytest.mark.parametrize(
    "tau, ideal_output",
    [
        (0, (0, 0)),
        (1, (0, 1)),
        (2, (1, 0)),
        (3, (1, 1)),
    ],
)
def test_two_qubit_grove_grid(tau, ideal_output):
    graph_circuit = TwoQubitGrover(tau=tau)

    backend = QuantinuumBackend(
        device_name="H1-1LE",
        api_handler=QuantinuumAPIOffline(),
    )

    # DecomposeClassicalExp().apply(graph_circuit)
    compiled_graph_circuit = backend.get_compiled_circuit(circuit=graph_circuit)

    n_shots = 100
    result = backend.run_circuit(circuit=compiled_graph_circuit, n_shots=n_shots)
    counts = graph_circuit.get_output_result(result=result).get_counts()
    assert counts[ideal_output] == counts.total()


def test_cnot_grid_error():
    with pytest.raises(
        Exception,
        match="tau must be one of 0, 1, 2 or 3.",
    ):
        TwoQubitGrover(tau=4)
