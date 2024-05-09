"""
Utilities for managing qubits.
In particular for measuring and reusing qubits.
"""

from typing import Dict, List

from pytket import Circuit
from pytket.unit_id import BitRegister, Qubit


class QubitManager(Circuit):
    """
    Manages a collection of qubits. In particular maintains
    a list of qubits which are not in use. This can be added
    to by measuring qubits (making them available again)
    and drawn from by initialising qubits.

    :ivar qubit_list: List of available quits.
    :ivar qubit_initialised: The qubits which have been added
        to the circuit. Qubits which are never required
        are never added to the circuit.
    :ivar qubit_meas_reg: A dictionary mapping qubits to the
        classical registers where their measurement results
        will be stored.
    """

    qubit_list: List[Qubit]
    qubit_initialised: Dict[Qubit, bool]
    qubit_meas_reg: Dict[Qubit, BitRegister]

    def __init__(self, n_physical_qubits: int) -> None:
        """Initialisation method. Creates tools for
        tracking the given number of qubits.

        :param n_physical_qubits: The number of qubits
            to manage.
        """
        self.qubit_list = [Qubit(index=i) for i in range(n_physical_qubits)]
        self.qubit_initialised = {qubit: False for qubit in self.qubit_list}
        self.qubit_meas_reg = {
            qubit: BitRegister(name=f"meas_{i}", size=1)
            for i, qubit in enumerate(self.qubit_list)
        }

        super().__init__()

        for meas_reg in self.qubit_meas_reg.values():
            self.add_c_register(register=meas_reg)

    def get_qubit(self) -> Qubit:
        """Return a qubit which is not in use, and which
        is included in the underlying circuit.

        :raises Exception: Raised if there are no more available qubits.
        :return: A qubit in the 0 computational basis state.
        """
        if len(self.qubit_list) == 0:
            raise Exception("You have run out of qubits.")

        qubit = self.qubit_list.pop(0)
        if not self.qubit_initialised[qubit]:
            self.add_qubit(id=qubit)
            self.qubit_initialised[qubit] = True

        self.add_c_setreg(0, self.qubit_meas_reg[qubit])
        self.Reset(qubit=qubit)

        return qubit

    @property
    def physical_qubits_used(self) -> List[Qubit]:
        """The physical qubits which have been used."""
        return [
            qubit
            for qubit, initialised in self.qubit_initialised.items()
            if initialised
        ]

    @property
    def initialised_qubits(self) -> List[Qubit]:
        """Qubits which have been initialised."""
        return [
            qubit
            for qubit, initialised in self.qubit_initialised.items()
            if initialised
        ]

    def managed_measure(self, qubit: Qubit) -> None:
        """Measure the given qubit, storing the result in the
        qubit's classical register. This will return the qubit to
        the list of available qubits.

        :param qubit: The qubit to be measured.
        :type qubit: Qubit

        :raises Exception: Raised if the qubit to be measured does
            not belong to the circuit.
        """
        if not self.qubit_initialised[qubit]:
            raise Exception(f"The qubit {qubit} has not been initialised.")
        self.qubit_list.insert(0, qubit)
        self.Measure(qubit=qubit, bit=self.qubit_meas_reg[qubit][0])
