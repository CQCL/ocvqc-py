"""
Tools for managing random bits. This include generating random bits
to dedicated registers.
"""

from collections.abc import Iterator

from pytket.unit_id import Bit, BitRegister

from pytket_mbqc_py.qubit_manager import QubitManager


class RandomRegisterManager(QubitManager):
    """Class for generating random bits, and managing dedicated registers
    where they are stored.
    """

    def generate_random_registers(
        self,
        n_registers: int,
        n_bits_per_reg: int = 3,
        max_n_randomness_qubits: int = 2,
    ) -> Iterator[BitRegister]:
        """Generate registers containing random bits. This is achieved
        by initialising hadamard basis plus states and measuring them.

        :param n_registers: The number of registers to generate.
        :param n_bits_per_reg: The number of bits in each register.
        :param max_n_randomness_qubits: The maximum number of qubits to use
            to generate randomness. If a number of qubits less than this
            number are actually available then the number available will
            be used.
        :raises Exception: Raised if there are no qubits left to use to
            generate randomness.
        :yield: Registers containing random bits.
        """

        if len(self.available_qubit_list) == 0:
            raise Exception(
                "There are no unused qubits "
                + "which can be used to generate randomness."
            )

        # The number of qubits used is the smaller of the maximum number set
        # by the user, or the number which is available.
        n_randomness_qubits = min(
            len(self.available_qubit_list), max_n_randomness_qubits
        )

        def get_random_bits(n_random_bits: int) -> Iterator[Bit]:
            """An iterator over bits populated with random values.
            These are created by initialising hadamard plus states and
            measuring the state in the computation basis. At most
            `n_randomness_qubits` qubit are created in the plus state, then
            measured and reset as appropriate until the requested number
            of bits have been created.

            Note that the physical bits may be reused so should be used
            before the next step of the iteration.

            :param n_random_bits: The number of random bits to generate.
            :yield: Bits containing random values.
            """
            # We repeatedly initialise and measure qubits in groups
            # of size n_randomness_qubits. This allows randomness generation
            # to be done in parallel where possible.
            for _ in range(n_random_bits // n_randomness_qubits):
                # Initialise all qubits.
                qubit_list = [self.get_qubit() for _ in range(n_randomness_qubits)]

                # For each qubit, initialise and measure.
                for qubit in qubit_list:
                    self.H(qubit=qubit)
                    self.managed_measure(qubit=qubit)
                    yield self.qubit_meas_reg[qubit][0]

            # This may leave a number of random bits less than
            # n_randomness_qubits to generate. We do this here.
            qubit_list = [
                self.get_qubit() for _ in range(n_random_bits % n_randomness_qubits)
            ]

            for qubit in qubit_list:
                self.H(qubit=qubit)
                self.managed_measure(qubit=qubit)
                yield self.qubit_meas_reg[qubit][0]

        # For each bit, copy it's value to a persistent
        # register. This is done in groups of n_bits_per_reg.
        for bit_index, bit in enumerate(
            get_random_bits(n_random_bits=n_bits_per_reg * n_registers)
        ):
            # If this is the first bit to copy to a new register, create
            # the new register.
            if bit_index % n_bits_per_reg == 0:
                reg = self.add_c_register(
                    name=f"rand_{bit_index // n_bits_per_reg}",
                    size=n_bits_per_reg,
                )

            # Copy the bit to the persistent register.
            self.add_classicalexpbox_bit(
                expression=reg[bit_index % n_bits_per_reg] | bit,
                target=[reg[bit_index % n_bits_per_reg]],
            )

            # If this is the last bit to fill the register, yield the register.
            if bit_index % n_bits_per_reg == n_bits_per_reg - 1:
                yield reg
