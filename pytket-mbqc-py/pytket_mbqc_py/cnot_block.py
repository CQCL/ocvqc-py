from .graph_circuit import GraphCircuit


class CNOTBlocksGraphCircuit(GraphCircuit):
    def __init__(self, n_physical_qubits, input_state, n_rows, n_columns):
        cnot_block_vertex_list = []

        super().__init__(n_physical_qubits=n_physical_qubits)

        for column in range(n_columns):
            cnot_block_vertex_list.append([])

            if column == 0:
                control_qubit, control_vertex = self.add_input_vertex()
                if input_state[0]:
                    self.X(control_qubit)
            else:
                control_vertex = cnot_block_vertex_list[column - 1][0][4]

            for row in range(n_rows):
                cnot_block_vertex_list[column].append([])
                cnot_block_vertex_list[column][row].append(control_vertex)

                if column == 0:
                    target_qubit, target_vertex = self.add_input_vertex()
                    if input_state[row + 1]:
                        self.X(target_qubit)
                elif row == n_rows - 1:
                    target_vertex = cnot_block_vertex_list[column - 1][row][5]
                else:
                    target_vertex = cnot_block_vertex_list[column - 1][row + 1][4]
                cnot_block_vertex_list[column][row].append(target_vertex)

                cnot_block_vertex_list[column][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[column][row][0],
                    vertex_two=cnot_block_vertex_list[column][row][2],
                )

                cnot_block_vertex_list[column][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[column][row][1],
                    vertex_two=cnot_block_vertex_list[column][row][3],
                )

                cnot_block_vertex_list[column][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[column][row][2],
                    vertex_two=cnot_block_vertex_list[column][row][4],
                )

                cnot_block_vertex_list[column][row].append(self.add_graph_vertex())
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[column][row][3],
                    vertex_two=cnot_block_vertex_list[column][row][5],
                )
                self.add_edge(
                    vertex_one=cnot_block_vertex_list[column][row][3],
                    vertex_two=cnot_block_vertex_list[column][row][4],
                )

                control_vertex = cnot_block_vertex_list[column][row][5]

                if column == 0:
                    continue

                if column == 1:
                    if row == 0:
                        self.corrected_measure(
                            vertex=cnot_block_vertex_list[column - 1][row][0],
                            t_multiple=0,
                        )

                    self.corrected_measure(
                        vertex=cnot_block_vertex_list[column - 1][row][1],
                        t_multiple=0,
                    )

                self.corrected_measure(
                    vertex=cnot_block_vertex_list[column - 1][row][2],
                    t_multiple=0,
                )
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[column - 1][row][3],
                    t_multiple=0,
                )
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[column - 1][row][4],
                    t_multiple=0,
                )
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[column - 1][row][5],
                    t_multiple=0,
                )

        for row in range(n_rows):
            if n_columns == 1:
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[0][row][0],
                    t_multiple=0,
                )
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[0][row][1],
                    t_multiple=0,
                )

            if row > 0:
                self.corrected_measure(
                    vertex=cnot_block_vertex_list[n_columns - 1][row][0],
                    t_multiple=0,
                )

            self.corrected_measure(
                vertex=cnot_block_vertex_list[n_columns - 1][row][2],
                t_multiple=0,
            )
            self.corrected_measure(
                vertex=cnot_block_vertex_list[n_columns - 1][row][3],
                t_multiple=0,
            )
