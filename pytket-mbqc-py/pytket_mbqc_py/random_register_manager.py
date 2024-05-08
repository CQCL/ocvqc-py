from pytket_mbqc_py.qubit_manager import QubitManager

class RandomRegisterManager(QubitManager):

    def __init__(self, n_physical_qubits):

        super().__init__(
            n_physical_qubits=n_physical_qubits
        )

    def generate_random_registers(self, n_registers, n_bits_per_reg = 3):

        n_unused_qubits = len(self.qubit_list) - len(self.physical_qubits_used)
        assert n_unused_qubits >= 0

        if n_unused_qubits == 0:
            raise Exception(
                "There are no unused qubits "
                + "which can be used to generate randomness."
            )
        
        def get_random_bits(n_random_bits):

            for _ in range(n_random_bits // n_unused_qubits):
                 
                qubit_list = [self.get_qubit() for _ in range(n_unused_qubits)]

                for qubit in qubit_list:

                    self.H(qubit=qubit)
                    self.managed_measure(qubit=qubit)
                    yield self.qubit_meas_reg[qubit][0]

            qubit_list = [self.get_qubit() for _ in range(n_random_bits % n_unused_qubits)]

            for qubit in qubit_list:

                self.H(qubit=qubit)
                self.managed_measure(qubit=qubit)
                yield self.qubit_meas_reg[qubit][0]

        for bit_index, bit in enumerate(get_random_bits(n_random_bits = n_bits_per_reg * n_registers)):

            if bit_index % n_bits_per_reg == 0:
                
                reg = self.add_c_register(
                    name=f'rand_{bit_index // n_bits_per_reg}',
                    size=n_bits_per_reg,
                )
            
            self.add_classicalexpbox_bit(
                expression=reg[bit_index % n_bits_per_reg] | bit,
                target=[reg[bit_index % n_bits_per_reg]],
            )

            if bit_index % n_bits_per_reg == n_bits_per_reg - 1:
                yield reg
