from pytket import Circuit
from pytket.unit_id import Qubit, BitRegister
import math
import random



class QubitManager(Circuit):
    def __init__(self, n_qubits: int) -> None:
        index_bits_required = max(1, math.ceil(math.log2(n_qubits)))
        if index_bits_required >= 32:
            raise Exception("You cannot index that many qubits.")

        self.qubit_list = [Qubit(index=i) for i in range(n_qubits)]
        self.qubit_initialised = {qubit: False for qubit in self.qubit_list}
        self.qubit_meas_reg = {
            qubit: BitRegister(name=f"meas_{i}", size=1)
            for i, qubit in enumerate(self.qubit_list)
        }
        self.qubit_initial_angle = {
            qubit: BitRegister(name=f"qubit_initial_angle_{i}", size=1)
            for i, qubit in enumerate(self.qubit_list)            
        }
        self.qubit_x_corr_reg = {
            qubit: BitRegister(name=f"x_corr_{i}", size=1)
            for i, qubit in enumerate(self.qubit_list)
        }
        self.qubit_z_corr_reg = {
            qubit: BitRegister(name=f"z_corr_{i}", size=1)
            for i, qubit in enumerate(self.qubit_list)
        }

        super().__init__()

        for meas_reg, qubit_initial_angle,x_corr_reg, z_corr_reg in zip(
            self.qubit_meas_reg.values(),
            self.qubit_initial_angle.values(),
            self.qubit_x_corr_reg.values(),
            self.qubit_z_corr_reg.values(),
        ):
            self.add_c_register(register=meas_reg)
            self.add_c_register(register=qubit_initial_angle)
            self.add_c_register(register=x_corr_reg)
            self.add_c_register(register=z_corr_reg)

        self.index_reg = BitRegister(name="index", size=index_bits_required)
        self.add_c_register(register=self.index_reg)

    def get_qubit(self) -> Qubit:
        if len(self.qubit_list) == 0:
            raise Exception("You have run out of qubits.")

        qubit = self.qubit_list.pop(0)
        if not self.qubit_initialised[qubit]:
            self.add_qubit(id=qubit)
            self.qubit_initialised[qubit] = True

        self.add_c_setreg(0, self.qubit_initial_angle[qubit])
        self.add_c_setreg(0, self.qubit_x_corr_reg[qubit])
        self.add_c_setreg(0, self.qubit_z_corr_reg[qubit])
        self.add_c_setreg(0, self.qubit_meas_reg[qubit])

        return qubit

    def get_plus_state(self, angle=None) -> Qubit:
            qubit = self.get_qubit()
        
            self.Reset(qubit=qubit)
            self.H(qubit=qubit)
            if angle is not None:
                self.Rz(angle=angle, qubit=qubit)
        
            return qubit
        
    def get_dummy_state(self,initial_state) -> Qubit:
        qubit = self.get_qubit()
    
        self.Reset(qubit=qubit)
        if initial_state == 1:
            self.X(qubit=qubit)
            return qubit
        elif initial_state == 0:
            return qubit
        else:
            raise Exception("Random bit is not 0 or 1")
    

    def return_qubit(self, qubit: Qubit) -> None:
        self.qubit_list.insert(0, qubit)

    def managed_measure(self, qubit: Qubit) -> None:
        super().Measure(qubit=qubit, bit=self.qubit_meas_reg[qubit][0])
        self.return_qubit(qubit)
