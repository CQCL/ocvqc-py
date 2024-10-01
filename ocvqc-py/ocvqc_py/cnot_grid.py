"""
Module for generating MBQC patterns which correspond to repeating
sequences of CNOT gates.
"""

from typing import Tuple

from .graph_circuit import GraphCircuit


class CNOTBlock(GraphCircuit):
    """
    A class generating MBQC patterns which correspond to repeating
    sequences of CNOT gates. In particular, CNOT gates are acted between
    neighbouring qubits in a line architecture. This pattern is repeated
    many times. This circuit is translated into an MBQC pattern.
    """

    def __init__(
        self,
        input_string: Tuple[int, ...],
        n_layers: int,
        verify: bool = False,
    ):
        """Initialisation method.

        :param input_string: The input into the CNOT circuit. Note that the
            circuit is a classical circuit and so the ideal output can
            be calculated efficiently.
        :param n_layers: The number of repetitions of a layer of CNOT gates
            acting between neighbours in a line architecture.
        :param verify: Flag indicating if the computation should be verified,
            defaults to False.
        """
        self.n_layers = n_layers
        self.input_string = input_string

        n_qubits = len(input_string)

        if verify:
            vertex_is_dummy = [True for _ in range(n_qubits)]
            vertex_is_dummy.extend([False for _ in range(n_qubits)])
            vertex_is_dummy.extend([True, False] * n_qubits * n_layers)
            vertex_is_dummy.extend([True] * n_qubits)

            vertex_is_dummy_list = [
                vertex_is_dummy,
                [not is_dummy for is_dummy in vertex_is_dummy],
            ]
        else:
            vertex_is_dummy_list = []

        super().__init__(
            n_physical_qubits=n_qubits + 1,
            n_logical_qubits=3 * n_qubits + 2 * n_layers * n_qubits,
            vertex_is_dummy_list=vertex_is_dummy_list,
        )

        # We will track the last alive vertex for each qubits.
        qubit_list = [
            self.add_graph_vertex(measurement_order=qubit) for qubit in range(n_qubits)
        ]

        # First the qubits should be initialised.
        for i, qubit in enumerate(qubit_list):
            measurement_order = n_qubits + 2 * i
            init_qubit = self.add_graph_vertex(measurement_order=measurement_order)
            self.add_edge(vertex_one=qubit, vertex_two=init_qubit)
            self.corrected_measure(vertex=qubit, t_multiple=4 * input_string[i])
            qubit_list[i] = init_qubit

        for layer in range(n_layers):
            measurement_order = n_qubits + 2 * n_qubits * layer + 1

            # A H is acted first on the first qubit.
            zero_h_qubit = self.add_graph_vertex(measurement_order=measurement_order)
            self.add_edge(vertex_one=qubit_list[0], vertex_two=zero_h_qubit)
            self.corrected_measure(vertex=qubit_list[0], t_multiple=0)
            qubit_list[0] = zero_h_qubit

            # Then a sequence of CZHH gates is acted.
            for control_index, target_index in zip(
                range(n_qubits - 1), range(1, n_qubits)
            ):
                measurement_order = (
                    n_qubits
                    + 2 * n_qubits * layer
                    + 2 * n_qubits
                    + control_index * (2 ** (layer != n_layers - 1))
                )
                control_output = self.add_graph_vertex(
                    measurement_order=measurement_order
                )
                self.add_edge(
                    vertex_one=qubit_list[control_index], vertex_two=control_output
                )
                # Note that an edge is added upwards so that the flow works out.
                if control_index > 0:
                    self.add_edge(
                        vertex_one=qubit_list[control_index],
                        vertex_two=qubit_list[control_index - 1],
                    )
                self.corrected_measure(vertex=qubit_list[control_index], t_multiple=0)
                qubit_list[control_index] = control_output

                measurement_order = (
                    n_qubits + 2 * n_qubits * layer + 1 + 2 * target_index
                )
                target_output = self.add_graph_vertex(
                    measurement_order=measurement_order
                )
                self.add_edge(
                    vertex_one=qubit_list[target_index], vertex_two=target_output
                )
                self.corrected_measure(vertex=qubit_list[target_index], t_multiple=0)
                qubit_list[target_index] = target_output

            measurement_order = (
                n_qubits
                + 2 * n_qubits * layer
                + 2 * n_qubits
                + (n_qubits - 1) * (2 ** (layer != n_layers - 1))
            )

            # A H is then added to the last qubits.
            final_h_qubit = self.add_graph_vertex(measurement_order=measurement_order)
            self.add_edge(vertex_one=qubit_list[-1], vertex_two=final_h_qubit)
            self.add_edge(vertex_one=qubit_list[-1], vertex_two=qubit_list[-2])
            self.corrected_measure(vertex=qubit_list[-1], t_multiple=0)
            qubit_list[-1] = final_h_qubit

        # H is then acted on all qubits.
        for i, qubit in enumerate(qubit_list):
            output_qubit = self.add_graph_vertex(measurement_order=None)
            self.add_edge(vertex_one=qubit, vertex_two=output_qubit)
            self.corrected_measure(vertex=qubit, t_multiple=0)
            qubit_list[i] = output_qubit

        # At which point they are all measured.
        for i, qubit in enumerate(qubit_list):
            self.corrected_measure(vertex=qubit, t_multiple=0)

    @property
    def ideal_outcome(self) -> Tuple[int, ...]:
        """The outcome of the circuit in the presence of no noise.

        :return: The ideal outcome.
        """
        ideal_outcome = list(self.input_string)
        for _ in range(self.n_layers):
            for index in range(len(ideal_outcome) - 1):
                ideal_outcome[index + 1] ^= ideal_outcome[index]
        return tuple(ideal_outcome)
