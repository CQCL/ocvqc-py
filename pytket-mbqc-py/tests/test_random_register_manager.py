from itertools import product

from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend

from pytket_mbqc_py import RandomRegisterManager


def test_random_register_manager():
    # Here we test a total of 15 random bits generated on 2 qubits.
    # Note that this mean one of the qubits must be used more than the other.
    rand_reg_mngr = RandomRegisterManager(n_physical_qubits=2)
    n_bits_per_reg = 3
    n_registers = 5
    reg_list = list(
        rand_reg_mngr.generate_random_registers(
            n_registers=n_registers,
            n_bits_per_reg=n_bits_per_reg,
        )
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
