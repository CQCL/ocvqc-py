from pytket_mbqc_py.qubit_manager import QubitManager
from typing import List, Tuple, Dict
from pytket import Qubit
import networkx as nx  # type:ignore


class GraphCircuit(QubitManager):
    def __init__(
        self,
        n_physical_qubits: int,
    ) -> None:
        super().__init__(n_physical_qubits=n_physical_qubits)

        self.entanglement_graph = nx.Graph()
        self.flow_graph = nx.DiGraph()

        self.vertex_qubit: List[Qubit] = []
        self.vertex_measured: List[bool] = []

    def get_outputs(self) -> Dict[int, Qubit]:
        unmeasured_flow_vertices = [
            vertex
            for vertex in self._vertices_with_flow()
            if not self.vertex_measured[vertex]
        ]
        if len(unmeasured_flow_vertices) > 0:
            raise Exception(
                "Only output vertices can be unmeasured. "
                + f"In particular {unmeasured_flow_vertices} have flow but are not measured."
            )

        # At this point we know that all unmeasured qubits
        # have no flow. As such they are safely outputs.
        output_qubits = {
            vertex: qubit
            for vertex, qubit in enumerate(self.vertex_qubit)
            if not self.vertex_measured[vertex]
        }

        # We need to correct all unmeasured qubits.
        for vertex in output_qubits.keys():
            self._apply_x_correction(vertex=vertex)
            self._apply_z_correction(vertex=vertex)

        return output_qubits

    def _add_vertex(self, qubit: Qubit) -> int:
        self.vertex_qubit.append(qubit)
        self.vertex_measured.append(False)

        index = len(self.vertex_qubit) - 1
        self.entanglement_graph.add_node(node_for_adding=index)
        self.flow_graph.add_node(node_for_adding=index)

        return index

    def add_input_vertex(self) -> Tuple[Qubit, int]:
        qubit = self.get_qubit()
        index = self._add_vertex(qubit=qubit)

        return (qubit, index)

    def add_graph_vertex(self) -> int:
        qubit = self.get_qubit()
        self.H(qubit)
        index = self._add_vertex(qubit=qubit)

        return index

    def _vertices_with_flow(self) -> List[int]:
        return list(
            set(
                predecessor
                for vertex in self.flow_graph.nodes
                for predecessor in self.flow_graph.predecessors(vertex)
            )
        )

    def add_edge(self, vertex_one: int, vertex_two: int) -> None:
        if vertex_one > vertex_two:
            raise Exception(
                f"{vertex_one} is greater than {vertex_two}. "
                + "Cannot add edge into the past."
            )

        if vertex_one not in self.entanglement_graph.nodes:
            raise Exception(
                f"There is no vertex with the index {vertex_one}. "
                + "Use the entanglement_graph attribute to see existing vertices."
            )

        if vertex_two not in self.entanglement_graph.nodes:
            raise Exception(
                f"There is no vertex with the index {vertex_two}. "
                + "Use the entanglement_graph attribute to see existing vertices."
            )

        if self.vertex_measured[vertex_one] or self.vertex_measured[vertex_two]:
            raise Exception("Cannot add edge after measure.")

        # vertex_two is a new neighbour of vertex_one. As such none of the vertices of which
        # vertex_one is the flow can have been measured.
        measured_inverse_flow = [
            flow_inverse
            for flow_inverse in self.flow_graph.predecessors(vertex_one)
            if self.vertex_measured[flow_inverse]
        ]
        if len(measured_inverse_flow) > 0:
            raise Exception(
                f"Adding the edge ({vertex_one}, {vertex_two}) does not define a valid flow. "
                + f"In particular {measured_inverse_flow} are the the inverse flow of {vertex_one} "
                + "and have already been measured. "
                + "The inverse flow of qubits to which you wish to attach edges must not be measured. "
            )

        # If this is the first future of vertex_one then it will be taken to be its flow.
        # If vertex_two is to be the flow of vertex_one than we must check that neighbours of
        # vertex_two are measured after vertex_one.
        if vertex_one not in self._vertices_with_flow():
            past_neighbours = [
                vertex
                for vertex in self.entanglement_graph.neighbors(n=vertex_two)
                if vertex < vertex_one
            ]
            if len(past_neighbours) > 0:
                raise Exception(
                    f"Adding the edge ({vertex_one}, {vertex_two}) does not define a valid flow. "
                    + f"In particular {past_neighbours} "
                    + f"are neighbours of {vertex_two} but are in the past of {vertex_one}. "
                    + f"As {vertex_two} would become the flow of {vertex_one} all of the "
                    + f"neighbours of {vertex_two} must be in the past of {vertex_one}."
                )

        # vertex_one is a neighbour of vertex_two. As such vertex_one must be measured after
        # any vertices of which vertex_two is its flow.
        if any(
            flow_inverse > vertex_one
            for flow_inverse in self.flow_graph.predecessors(vertex_two)
        ):
            raise Exception(
                "This does not define a valid flow. "
                f"In particular {vertex_two} is the flow of {self.flow_graph.predecessors(vertex_two)}, "
                f"some of which are measured before {vertex_one}."
            )

        # If this is the first future of vertex_one then it is taken to be its flow.
        if vertex_one not in self._vertices_with_flow():
            self.flow_graph.add_edge(
                u_of_edge=vertex_one,
                v_of_edge=vertex_two,
            )

        self.CZ(self.vertex_qubit[vertex_one], self.vertex_qubit[vertex_two])

        assert vertex_one in self.entanglement_graph.nodes
        assert vertex_two in self.entanglement_graph.nodes
        self.entanglement_graph.add_edge(
            u_of_edge=vertex_one,
            v_of_edge=vertex_two,
        )

    def _apply_x_correction(self, vertex: int) -> None:
        self.X(
            self.vertex_qubit[vertex],
            condition=self.qubit_x_corr_reg[self.vertex_qubit[vertex]][0],
        )

    def _apply_z_correction(self, vertex: int) -> None:
        self.Z(
            self.vertex_qubit[vertex],
            condition=self.qubit_z_corr_reg[self.vertex_qubit[vertex]][0],
        )

    def _apply_classical_z_correction(self, vertex: int) -> None:
        self.add_classicalexpbox_bit(
            self.qubit_meas_reg[self.vertex_qubit[vertex]][0]
            ^ self.qubit_z_corr_reg[self.vertex_qubit[vertex]][0],
            [self.qubit_meas_reg[self.vertex_qubit[vertex]][0]],
        )

    def corrected_measure(self, vertex: int, t_multiple: int = 0) -> None:
        # Check that the vertex being measured has not already been measured.
        if self.vertex_measured[vertex]:
            raise Exception(
                f"Vertex {vertex} has already been measured and cannot be measured again."
            )

        # Check that all vertices before the one given have been measured.
        if any(self.vertex_measured[vertex:]):
            raise Exception(
                f"Measuring {vertex} does not respect the measurement order. "
                + f"Vertices {[vertex + i for i, measured in enumerate(self.vertex_measured[vertex:]) if measured]} "
                + f"are in the future of {vertex} and have already been measured."
            )

        # This is actually optional as the correction commutes with the measurement.
        self._apply_x_correction(vertex=vertex)

        inverse_t_multiple = 8 - t_multiple
        inverse_t_multiple = inverse_t_multiple % 8
        if inverse_t_multiple // 4:
            self.Z(self.vertex_qubit[vertex])
        if (inverse_t_multiple % 4) // 2:
            self.S(self.vertex_qubit[vertex])
        if inverse_t_multiple % 2:
            self.T(self.vertex_qubit[vertex])
        self.H(self.vertex_qubit[vertex])

        self.managed_measure(qubit=self.vertex_qubit[vertex])
        self._apply_classical_z_correction(vertex=vertex)
        self.vertex_measured[vertex] = True

        # Check that the flow of the vertex being measured has
        # not been measured.
        assert len(list(self.flow_graph.successors(vertex))) <= 1

        if len(list(self.flow_graph.successors(vertex))) == 0:
            raise Exception(
                f"Vertex {vertex} has no flow. "
                "It is not possible to perform a corrected measure of a qubit without flow. "
                "Please give this vertex a flow, or use the get_output to perform the necessary corrections."
            )
        vertex_flow = list(self.flow_graph.successors(vertex))[0]
        assert not self.vertex_measured[vertex_flow]

        self.add_classicalexpbox_bit(
            self.qubit_meas_reg[self.vertex_qubit[vertex]][0]
            ^ self.qubit_x_corr_reg[self.vertex_qubit[vertex_flow]][0],
            [self.qubit_x_corr_reg[self.vertex_qubit[vertex_flow]][0]],
        )

        for neighbour in self.entanglement_graph.neighbors(n=vertex_flow):
            if neighbour == vertex:
                continue

            # Check that the vertex being updated has not been measured
            assert not self.vertex_measured[neighbour]

            self.add_classicalexpbox_bit(
                self.qubit_meas_reg[self.vertex_qubit[vertex]][0]
                ^ self.qubit_z_corr_reg[self.vertex_qubit[neighbour]][0],
                [self.qubit_z_corr_reg[self.vertex_qubit[neighbour]][0]],
            )
