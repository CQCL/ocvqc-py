"""Class for generating graph states implementing layers of
CNOT gates.
"""

from typing import List, Tuple

from .graph_circuit import GraphCircuit


class CNOTBlocksGraphCircuit(GraphCircuit):
    """Class for generating graph state implementing
    layers of CNOT gates. Consider an n qubit input state
    initialised in a computational bases state. A 'layer' of CNOT gates
    consists of a CNOT gate acting between qubit 1 and 2, then
    another between 2 and 3, etc until a CNOT acts between qubit n-1 and n.
    Note that as the inputs are initialised in computational basis states
    the CNOT gates can be considered to be classical CNOT gates,
    and the ideal outcome is deterministic.
    """

    def __init__(
        self,
        n_physical_qubits: int,
        input_state: Tuple[int],
        n_layers: int,
        n_registers: int,
    ) -> None:
        """Initialisation method.

        :param n_physical_qubits: The maximum number of physical qubits
            available. These qubits will be reused and so the total
            number of 'logical' qubits may be larger.
        :type n_physical_qubits: int
        :param input_state: Integer tuple describing the input
            to the circuit. This will be a classical binary
            string and so the outcome is deterministic.
        :type input_state: Tuple[int]
        :param n_layers: The number of layers of CNOT gates.
        :type n_layers: int
        """

        self.input_state = input_state
        self.n_layers = n_layers

        # The number of rows of CNOT blocks
        # note that this is one less than the number of entries in the
        # input state as each CNOT has two inputs.
        n_rows = len(input_state) - 1

        # This structure contains information about which
        # vertices in the graph state corresponds to which
        # vertices in each of the CNOT blocks.
        # Note that each CNOT block contains 6 vertices implementing
        # a CNOT gate. However as some of these may be outputs from
        # other CNOT blocks and so there may be repeats.
        cnot_block_vertex_list: List[List[List[int]]] = [
            [[] for _ in range(n_rows)] for _ in range(n_layers)
        ]

        super().__init__(
            n_physical_qubits=n_physical_qubits,
            n_registers=n_registers,
        )

        for layer in range(n_layers):
            # If this is the first layer then the control qubit of the first row needs
            # to be initialised. If not then the control vertex is taken from
            # the layer before.
            if layer == 0:
                control_qubit, control_vertex = self.add_input_vertex()
                if input_state[0]:
                    self.X(control_qubit)
            else:
                control_vertex = cnot_block_vertex_list[layer - 1][0][4]

            for row in range(n_rows):
                # for each block the 0th qubit is the control.
                cnot_block_vertex_list[layer][row].append(control_vertex)

                # If this is the 0th layer then the target qubit needs to be
                # initialised.
                if layer == 0:
                    target_qubit, target_vertex = self.add_input_vertex()
                    if input_state[row + 1]:
                        self.X(target_qubit)
                # If this is the last row then the target vertex is the output
                # target vertex of the cnot on the same row but previous
                # layer.
                elif row == n_rows - 1:
                    target_vertex = cnot_block_vertex_list[layer - 1][row][5]
                # Otherwise the target vertex is the output control vertex
                # of the CNOT block in the previous layer and the next row.
                else:
                    target_vertex = cnot_block_vertex_list[layer - 1][row + 1][4]
                cnot_block_vertex_list[layer][row].append(target_vertex)

                # Now the rest of the CNOT block can be constructed.
                cnot_block_vertex_list[layer][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[layer][row][0],
                    vertex_two=cnot_block_vertex_list[layer][row][2],
                )

                cnot_block_vertex_list[layer][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[layer][row][1],
                    vertex_two=cnot_block_vertex_list[layer][row][3],
                )

                cnot_block_vertex_list[layer][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[layer][row][2],
                    vertex_two=cnot_block_vertex_list[layer][row][4],
                )

                cnot_block_vertex_list[layer][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[layer][row][3],
                    vertex_two=cnot_block_vertex_list[layer][row][5],
                )
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[layer][row][3],
                    vertex_two=cnot_block_vertex_list[layer][row][4],
                )

                # The control vertex of the next row will be the
                # output target vertex of this row.
                control_vertex = cnot_block_vertex_list[layer][row][5]

                # If this is not the 0th layer then the previous layer
                # can be measured.
                if layer > 0:
                    # If this is the 1th later then the inputs of the previous
                    # layer (the 0th layer) will not have been measured and should now be.
                    # Note that or other layers they will have been measured by this point
                    # as they are the 4th and 5th vertices of previous layers.
                    if layer == 1:
                        # If this is the 0th row then we need to measure the input
                        # control. It is not necessary in general as it would be the
                        # output target of previous blocks.
                        if row == 0:
                            self.corrected_measure(
                                vertex=cnot_block_vertex_list[layer - 1][row][0],
                                t_multiple=0,
                            )

                        self.corrected_measure(
                            vertex=cnot_block_vertex_list[layer - 1][row][1],
                            t_multiple=0,
                        )

                    # The rest of the block is measured.
                    self.corrected_measure(
                        vertex=cnot_block_vertex_list[layer - 1][row][2],
                        t_multiple=0,
                    )
                    self.corrected_measure(
                        vertex=cnot_block_vertex_list[layer - 1][row][3],
                        t_multiple=0,
                    )
                    self.corrected_measure(
                        vertex=cnot_block_vertex_list[layer - 1][row][4],
                        t_multiple=0,
                    )
                    self.corrected_measure(
                        vertex=cnot_block_vertex_list[layer - 1][row][5],
                        t_multiple=0,
                    )

        # When we reach the end we can measure everything that is not the output.
        for row in range(n_rows):
            # If there is only 1 layer then the inputs will need to be
            # measured. If there are more layers then they will already
            # have been measured as inputs to previous later layers.
            if n_layers == 1:
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[0][row][0],
                    t_multiple=0,
                )
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[0][row][1],
                    t_multiple=0,
                )

            # If there is more than one layer then the output
            # qubit in blocks on rows greater then 1 are yet to be measured.
            # they are measured on the 0th row as these are output controls
            # of previous rows.
            elif row > 0:
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[n_layers - 1][row][0],
                    t_multiple=0,
                )

            self.corrected_measure(
                vertex=cnot_block_vertex_list[n_layers - 1][row][2],
                t_multiple=0,
            )
            self.corrected_measure(
                vertex=cnot_block_vertex_list[n_layers - 1][row][3],
                t_multiple=0,
            )

    @property
    def output_state(self) -> Tuple[int, ...]:
        """The ideal output bit string.

        :return: The ideal output bit string.
        :rtype: Tuple[int]
        """
        output_state = list(self.input_state)
        for _ in range(self.n_layers):
            for i in range(len(self.input_state) - 1):
                output_state[i + 1] = output_state[i] ^ output_state[i + 1]
        return tuple(output_state)
