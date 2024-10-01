"""
Module implementing the 2 qubit grover example from:
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.062422.
"""

from ocvqc_py import GraphCircuit


class TwoQubitGrover(GraphCircuit):
    """Class implementing the 2 qubit grover example from:
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.062422.
    The marked value is returned with probability 1.
    """

    def __init__(self, tau: int) -> None:
        """Initialisation method.

        :param tau: Marked value.
        :type tau: int
        :raises Exception: Raised if the marked value is not one of
            0, 1, 2, or 3.
        """
        match tau:
            case 0:
                theta_three = 4
                theta_four = 4
            case 1:
                theta_three = 4
                theta_four = 0
            case 2:
                theta_three = 0
                theta_four = 4
            case 3:
                theta_three = 0
                theta_four = 0
            case _:
                raise Exception("tau must be one of 0, 1, 2 or 3.")

        super().__init__(
            n_physical_qubits=3,
            n_logical_qubits=8,
            vertex_is_dummy_list=[
                [True, False, False, True, False, True, False, True],
                [False, True, True, False, True, False, True, False],
            ],
        )

        vertex_one = self.add_graph_vertex(measurement_order=0)

        vertex_two = self.add_graph_vertex(measurement_order=1)

        vertex_three = self.add_graph_vertex(measurement_order=3)
        self.add_edge(vertex_one=vertex_one, vertex_two=vertex_three)
        self.add_edge(vertex_one=vertex_one, vertex_two=vertex_two)
        self.corrected_measure(vertex=vertex_one)

        vertex_four = self.add_graph_vertex(measurement_order=2)
        self.add_edge(vertex_one=vertex_two, vertex_two=vertex_four)
        self.corrected_measure(vertex=vertex_two)

        vertex_five = self.add_graph_vertex(measurement_order=5)
        self.add_edge(vertex_one=vertex_four, vertex_two=vertex_five)
        self.corrected_measure(vertex=vertex_four, t_multiple=theta_four)

        vertex_six = self.add_graph_vertex(measurement_order=4)
        self.add_edge(vertex_one=vertex_three, vertex_two=vertex_six)
        self.corrected_measure(vertex=vertex_three, t_multiple=theta_three)

        vertex_seven = self.add_graph_vertex(measurement_order=None)
        self.add_edge(vertex_one=vertex_six, vertex_two=vertex_seven)
        self.corrected_measure(vertex=vertex_six)

        vertex_eight = self.add_graph_vertex(measurement_order=None)
        self.add_edge(vertex_one=vertex_five, vertex_two=vertex_eight)
        self.corrected_measure(vertex=vertex_five)

        self.add_edge(vertex_one=vertex_eight, vertex_two=vertex_seven)

        self.corrected_measure(vertex=vertex_seven, t_multiple=4)
        self.corrected_measure(vertex=vertex_eight, t_multiple=4)
