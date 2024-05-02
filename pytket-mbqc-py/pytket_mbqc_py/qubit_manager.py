from pytket import Circuit
from pytket.unit_id import BitRegister, Qubit


class QubitManager(Circuit):
    def __init__(self, n_physical_qubits: int) -> None:
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
    def physical_qubits_used(self):
        return [
            qubit
            for qubit, initialised in self.qubit_initialised.items()
            if initialised
        ]

    def managed_measure(self, qubit: Qubit) -> None:
        self.qubit_list.insert(0, qubit)
        self.Measure(qubit=qubit, bit=self.qubit_meas_reg[qubit][0])
