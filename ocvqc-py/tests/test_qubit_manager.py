import pytest
from pytket import Qubit

from ocvqc_py import QubitManager


def test_simple_measurement():
    qubit_mgr = QubitManager(n_physical_qubits=2)

    reg = qubit_mgr.add_c_register(
        name="meas_reg",
        size=1,
    )

    qubit = qubit_mgr.get_qubit(measure_bit=reg[0])

    assert qubit_mgr.available_qubit_list == [Qubit(1)]
    assert qubit_mgr.all_qubit_list == [Qubit(0), Qubit(1)]

    # Test the case where the qubit to be measured is not in the circuit.
    with pytest.raises(
        Exception,
        match=f"The qubit q\[2\] is not in use and so cannot be measured. ",
    ):
        qubit_mgr.managed_measure(Qubit(2))

    # Test the case where the qubit to be measured has not been initialised.
    with pytest.raises(
        Exception,
        match="The qubit q\[1\] is not in use and so cannot be measured. ",
    ):
        qubit_mgr.managed_measure(Qubit(1))

    qubit_mgr.managed_measure(qubit)

    assert qubit_mgr.available_qubit_list == [Qubit(0), Qubit(1)]
    assert qubit_mgr.all_qubit_list == [Qubit(0), Qubit(1)]
