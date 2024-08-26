from typing import Tuple

from numpy.random import default_rng

from pytket_mbqc_py import GraphCircuit


class RandomIdentityGraph(GraphCircuit):
    def __init__(
        self, n_layers: int, input_string: Tuple[int], rng=default_rng()
    ) -> None:
        assert n_layers % 2 == 0
        n_qubits = len(input_string)

        super().__init__(
            n_physical_qubits=n_qubits + 1,
            n_logical_qubits=3 * n_qubits
            + n_layers * (2 * (n_qubits - 1) + 3 * n_qubits)
            + n_qubits,
        )

        qubit_list = [
            self.add_graph_vertex(measurement_order=qubit) for qubit in range(n_qubits)
        ]

        angles = rng.integers(low=0, high=8, size=((n_layers // 2), n_qubits))
        print(angles)

        print("Initialise")
        for i, qubit in enumerate(qubit_list):
            measurement_order = n_qubits + i
            init_qubit = self.add_graph_vertex(measurement_order=measurement_order)
            self.add_edge(vertex_one=qubit, vertex_two=init_qubit)
            self.corrected_measure(vertex=qubit, t_multiple=4 * input_string[i])
            qubit_list[i] = init_qubit

        print("Hadamard")
        for i, qubit in enumerate(qubit_list):
            measurement_order = 2 * n_qubits + max(2 * i - 1, 0)
            h_qubit = self.add_graph_vertex(measurement_order=measurement_order)
            self.add_edge(vertex_one=qubit, vertex_two=h_qubit)
            self.corrected_measure(vertex=qubit)
            qubit_list[i] = h_qubit

        for layer in range(n_layers // 2):
            measurement_order_to_layer = 2 * n_qubits + layer * (
                2 * (n_qubits - 1) + 3 * n_qubits
            )

            print("CNOT Layer")
            for control_index, target_index in zip(
                range(n_qubits - 1), range(1, n_qubits)
            ):
                print((control_index, target_index))

                trgt_measurement_order = measurement_order_to_layer + 2 * target_index
                ctrl_measurement_order = (
                    measurement_order_to_layer
                    + 2 * (n_qubits - 1)
                    + ((n_qubits - 1) - control_index)
                )

                ctrl_qubit = self.add_graph_vertex(
                    measurement_order=ctrl_measurement_order
                )
                self.add_edge(
                    vertex_one=qubit_list[control_index], vertex_two=ctrl_qubit
                )
                if control_index > 0:
                    self.add_edge(
                        vertex_one=qubit_list[control_index],
                        vertex_two=qubit_list[control_index - 1],
                    )
                self.corrected_measure(vertex=qubit_list[control_index])
                qubit_list[control_index] = ctrl_qubit

                trgt_qubit = self.add_graph_vertex(
                    measurement_order=trgt_measurement_order
                )
                self.add_edge(
                    vertex_one=qubit_list[target_index], vertex_two=trgt_qubit
                )
                self.corrected_measure(vertex=qubit_list[target_index])
                qubit_list[target_index] = trgt_qubit

            measurement_order_to_layer = (
                3 * n_qubits + (layer + 1) * (2 * (n_qubits - 1)) + layer * 3 * n_qubits
            )

            print("H layer")
            for index in range(n_qubits - 1, -1, -1):
                measurement_order = measurement_order_to_layer + index
                qubit = self.add_graph_vertex(measurement_order=measurement_order)
                self.add_edge(vertex_one=qubit_list[index], vertex_two=qubit)
                if index == n_qubits - 1:
                    self.add_edge(
                        vertex_one=qubit_list[index], vertex_two=qubit_list[index - 1]
                    )
                self.corrected_measure(vertex=qubit_list[index])
                qubit_list[index] = qubit

            measurement_order_to_layer += n_qubits

            print("Rotation Layer")
            for index in range(n_qubits):
                measurement_order = measurement_order_to_layer + index
                qubit = self.add_graph_vertex(measurement_order=measurement_order)
                self.add_edge(vertex_one=qubit_list[index], vertex_two=qubit)
                self.corrected_measure(
                    vertex=qubit_list[index], t_multiple=angles[layer, index]
                )
                qubit_list[index] = qubit

            measurement_order_to_layer += n_qubits

            print("Rotation H Layer")
            for index in range(n_qubits):
                if layer == n_layers // 2 - 1:
                    measurement_order = measurement_order_to_layer + index
                else:
                    measurement_order = measurement_order_to_layer + max(
                        2 * index - 1, 0
                    )
                qubit = self.add_graph_vertex(measurement_order=measurement_order)
                self.add_edge(vertex_one=qubit_list[index], vertex_two=qubit)
                self.corrected_measure(vertex=qubit_list[index])
                qubit_list[index] = qubit

        print("==== Inverse Layers ====")

        measurement_order_to_layer = 3 * n_qubits + (n_layers // 2) * (
            3 * n_qubits + 2 * (n_qubits - 1)
        )

        for layer in range(n_layers // 2):
            print("Inverse Rotation Layer")
            for index in range(n_qubits):
                measurement_order = measurement_order_to_layer + index
                qubit = self.add_graph_vertex(measurement_order=measurement_order)
                self.add_edge(vertex_one=qubit_list[index], vertex_two=qubit)
                self.corrected_measure(
                    vertex=qubit_list[index],
                    t_multiple=8 - angles[(n_layers // 2) - layer - 1, index],
                )
                qubit_list[index] = qubit

            measurement_order_to_layer += n_qubits

            print("Inverse Rotation H Layer")
            for index in range(n_qubits):
                measurement_order = measurement_order_to_layer + max(
                    2 * ((n_qubits - 1) - index) - 1, 0
                )
                qubit = self.add_graph_vertex(measurement_order=measurement_order)
                self.add_edge(vertex_one=qubit_list[index], vertex_two=qubit)
                self.corrected_measure(vertex=qubit_list[index])
                qubit_list[index] = qubit

            print("Inverse CNOT Layer")
            for control_index, target_index in zip(
                range(n_qubits - 2, -1, -1), range(n_qubits - 1, 0, -1)
            ):
                print((control_index, target_index))

                ctrl_measurement_order = measurement_order_to_layer + 2 * (
                    n_qubits - target_index
                )
                trgt_measurement_order = (
                    measurement_order_to_layer + 2 * (n_qubits) + control_index - 1
                )

                trgt_qubit = self.add_graph_vertex(
                    measurement_order=trgt_measurement_order
                )
                self.add_edge(
                    vertex_one=qubit_list[target_index], vertex_two=trgt_qubit
                )
                if target_index < n_qubits - 1:
                    self.add_edge(
                        vertex_one=qubit_list[target_index],
                        vertex_two=qubit_list[target_index + 1],
                    )
                self.corrected_measure(vertex=qubit_list[target_index])
                qubit_list[target_index] = trgt_qubit

                ctrl_qubit = self.add_graph_vertex(
                    measurement_order=ctrl_measurement_order
                )
                self.add_edge(
                    vertex_one=qubit_list[control_index], vertex_two=ctrl_qubit
                )
                self.corrected_measure(vertex=qubit_list[control_index])
                qubit_list[control_index] = ctrl_qubit

            measurement_order_to_layer += n_qubits + 2 * (n_qubits - 1)

            print("Inverse H Layer")
            for index in range(n_qubits):
                if layer == n_layers // 2 - 1:
                    measurement_order = None
                else:
                    measurement_order = measurement_order_to_layer + index
                qubit = self.add_graph_vertex(measurement_order=measurement_order)
                self.add_edge(vertex_one=qubit_list[index], vertex_two=qubit)
                if index == 0:
                    self.add_edge(
                        vertex_one=qubit_list[index], vertex_two=qubit_list[index + 1]
                    )
                self.corrected_measure(vertex=qubit_list[index])
                qubit_list[index] = qubit

            measurement_order_to_layer += n_qubits

        for i, qubit in enumerate(qubit_list):
            self.corrected_measure(vertex=qubit, t_multiple=0)
