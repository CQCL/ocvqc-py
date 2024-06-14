"""
Utilities for managing qubits.
In particular for measuring and reusing qubits.
"""

from typing import Dict, List

from pytket import Circuit
from pytket.unit_id import BitRegister, Qubit, Bit


class QubitManager(Circuit):
    """
    Manages a collection of qubits. In particular maintains
    a list of qubits which are not in use. This can be added
    to by measuring qubits (making them available again)
    and drawn from by initialising qubits.

    :ivar available_qubit_list: List of available quits.
        These are those qubits which have either never been used, or have been
        used but have been measured and can be reset.
    :ivar all_qubit_list: All qubits which could in principle be used.
    :ivar physical_qubits_used: A set containing the qubits which
        have been made use of at some point.
    """

    available_qubit_list: List[Qubit]
    all_qubit_list: List[Qubit]
    # qubit_meas_reg: Dict[Qubit, BitRegister]
    physical_qubits_used: set[Qubit]

    def __init__(self, n_physical_qubits: int) -> None:
        """Initialisation method. Creates tools for
        tracking the given number of qubits.

        :param n_physical_qubits: The number of qubits
            to manage.
        """
        self.available_qubit_list = [Qubit(index=i) for i in range(n_physical_qubits)]
        self.all_qubit_list = [Qubit(index=i) for i in range(n_physical_qubits)]
        # self.qubit_meas_reg = {
        #     qubit: BitRegister(name=f"meas_{i}", size=1)
        #     for i, qubit in enumerate(self.available_qubit_list)
        # }
        self.qubit_meas_bit = dict()
        self.physical_qubits_used = set()

        # self.qubits_created = 0

        super().__init__()

        # for meas_reg in self.qubit_meas_reg.values():
        #     self.add_c_register(register=meas_reg)

        for qubit in self.all_qubit_list:
            self.add_qubit(id=qubit)

    def get_qubit(self, measure_bit: Bit) -> Qubit:
        """Return a qubit which is not in use, and which
        is included in the underlying circuit.

        :raises Exception: Raised if there are no more available qubits.
        :return: A qubit in the 0 computational basis state.
        """
        if len(self.available_qubit_list) == 0:
            raise Exception("You have run out of qubits.")

        qubit = self.available_qubit_list.pop(0)
        self.physical_qubits_used.add(qubit)
        # self.qubit_meas_reg[qubit] = self.add_c_register(
        #     name=f'meas_{self.qubits_created}', size=1
        # )
        self.qubit_meas_bit[qubit] = measure_bit
        # self.add_c_setreg(0, self.qubit_meas_reg[qubit])
        # print(self.qubit_meas_bit[qubit])
        self.add_c_setbits([0], [self.qubit_meas_bit[qubit]])
        self.Reset(qubit=qubit)

        # self.qubits_created += 1

        return qubit

    def managed_measure(self, qubit: Qubit) -> None:
        """Measure the given qubit, storing the result in the
        qubit's classical register. This will return the qubit to
        the list of available qubits.

        :param qubit: The qubit to be measured.

        :raises Exception: Raised if the qubit to be measured is not currently
            in use. A qubit is in use if it has been initialised with the
            get_qubit method, but has not yet been measured.
        """

        in_use_qubits = set(self.all_qubit_list) - set(self.available_qubit_list)
        if qubit not in in_use_qubits:
            raise Exception(
                f"The qubit {qubit} is not in use and so cannot be measured. "
                + f"Qubits in use are {in_use_qubits}. "
                + "For a qubit to be in use it must be initialised with the "
                + "get_qubit method."
            )
        self.available_qubit_list.insert(0, qubit)
        # self.Measure(qubit=qubit, bit=self.qubit_meas_reg[qubit][0])
        self.Measure(qubit=qubit, bit=self.qubit_meas_bit[qubit])
