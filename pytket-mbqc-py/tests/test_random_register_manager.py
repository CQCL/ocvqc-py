from itertools import product

import pytest
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
from pytket.unit_id import BitRegister

from pytket_mbqc_py import RandomRegisterManager


def test_multiple_register_initialisation():
    rand_reg_mgr = RandomRegisterManager(n_physical_qubits=3)
    rand_reg_mgr.generate_random_registers(n_registers=2)

    rand_reg_mgr.get_qubit()
    rand_reg_mgr.generate_random_registers(n_registers=2)

    rand_reg_mgr.get_qubit()
    rand_reg_mgr.generate_random_registers(n_registers=2)

    assert all(
        BitRegister(name=f"rand_{index}", size=3) in rand_reg_mgr.c_registers
        for index in range(6)
    )

    rand_reg_mgr.get_qubit()
    with pytest.raises(
        Exception,
        match="There are no unused qubits which can be used to generate randomness.",
    ):
        rand_reg_mgr.generate_random_registers(n_registers=2)

    with pytest.raises(
        Exception,
        match="You have run out of qubits.",
    ):
        rand_reg_mgr.get_qubit()


def test_random_register_manager():
    # Here we test a total of 15 random bits generated on 2 qubits.
    # Note that this mean one of the qubits must be used more than the other.
    rand_reg_mngr = RandomRegisterManager(n_physical_qubits=2)
    n_bits_per_reg = 3
    n_registers = 5
    reg_list = rand_reg_mngr.generate_random_registers(
        n_registers=n_registers,
        n_bits_per_reg=n_bits_per_reg,
    )

    # Check that there are 3 registers, and that each has 3 bits.
    assert len(reg_list) == n_registers
    assert all(reg.size == n_bits_per_reg for reg in reg_list)

    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
    n_shots = 100

    compiled_circuit = backend.get_compiled_circuit(rand_reg_mngr)
    result = backend.run_circuit(
        circuit=compiled_circuit,
        n_shots=n_shots,
        seed=0,
    )
    result.get_counts(cbits=reg_list[0])

    # Each bit string in each register should occur with equal probability.
    # This may fail sometimes as we are performing a statistical check.
    # With the correct seed it should pass every time.
    for cbits in reg_list:
        counts = result.get_counts(cbits=cbits)
        assert all(
            abs(counts[bit_string] - (n_shots / (2**n_bits_per_reg))) < (n_shots**0.5)
            for bit_string in product([0, 1], repeat=n_bits_per_reg)
        )
