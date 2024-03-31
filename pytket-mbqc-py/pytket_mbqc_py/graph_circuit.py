from pytket_mbqc_py.qubit_manager import QubitManager
from typing import List, Tuple, Dict
from pytket import Qubit
from pytket_mbqc_py.wasm_file_handler import get_wasm_file_handler
import networkx as nx  # type:ignore


class GraphCircuit(QubitManager):
    def __init__(
        self,
        n_physical_qubits: int,
    ) -> None:
        super().__init__(n_physical_qubits=n_physical_qubits)

        self.wfh = get_wasm_file_handler()
        self.add_wasm_to_reg("init_corrections", self.wfh, [], [])

        self.entanglement_graph = nx.Graph()
        self.flow_graph = nx.DiGraph()

        self.output_vertices: List[int] = []

        self.vertex_qubit: List[Qubit] = []

        self.vertex_measured: List[bool] = []

    def correct_outputs(self) -> None:
        unmeasured_graph_vertices = [
            vertex
            for vertex in self._vertices_with_flow()
            if not self.vertex_measured[vertex]
        ]
        if len(unmeasured_graph_vertices) > 0:
            raise Exception(
                "Only output vertices can be unmeasured. "
                + f"In particular {unmeasured_graph_vertices} must be measured."
            )

        for vertex, measured in enumerate(self.vertex_measured):
            if not measured:
                self._apply_correction(vertex=vertex)

    def _add_vertex(self, qubit: Qubit) -> int:
        self.vertex_qubit.append(qubit)
        self.vertex_measured.append(False)

        index = len(self.vertex_qubit) - 1
        self.entanglement_graph.add_node(node_for_adding=index)
        self.flow_graph.add_node(node_for_adding=index)

        return index

    def add_input_vertex(self) -> Tuple[Qubit, int]:
        if len(self.vertex_qubit) == 100:
            raise Exception("The current maximum number of vertices is 100.")

        qubit = super().get_qubit()
        index = self._add_vertex(qubit=qubit)

        return (qubit, index)

    def add_output_vertex(self) -> int:
        index = self.add_graph_vertex()
        self.output_vertices.append(index)
        return index

    def add_graph_vertex(self) -> int:
        if len(self.vertex_qubit) == 100:
            raise Exception("The current maximum number of vertices is 100.")

        qubit = self.get_qubit()
        self.H(qubit)
        index = self._add_vertex(qubit=qubit)

        return index
    
    def _vertices_with_flow(self) -> List[int]:
        return list(set(predecessor for vertex in self.flow_graph.nodes for predecessor in self.flow_graph.predecessors(vertex)))

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
                "This does not define a valid flow. "
                + f"In particular {measured_inverse_flow} are the the inverse flow of {vertex_one} "
                + "but have already been measured."
            )

        # If this is the first future of vertex_one then it will be taken to be its flow.
        # This is only not the case if vertex_one is an output vertex, in which case it has no flow.
        # If vertex_two is to be the flow of vertex_one than we must check that neighbours of
        if (vertex_one not in self._vertices_with_flow()) and (
            vertex_one not in self.output_vertices
        ):
            vertex_neighbours = self.entanglement_graph.neighbors(n=vertex_two)
            if any(vertex < vertex_one for vertex in vertex_neighbours):
                raise Exception(
                    "This circuit does not have a valid flow. "
                    + f"In particular {[vertex for vertex in vertex_neighbours if vertex < vertex_one]} "
                    + f"are neighbours of {vertex_two} but are in the past of {vertex_one}."
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
        if vertex_one not in self._vertices_with_flow() and (
            vertex_one not in self.output_vertices
        ):
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

    def _apply_correction(self, vertex: int) -> None:
        self.add_c_setreg(vertex, self.index_reg)

        self.add_wasm_to_reg(
            "get_x_correction",
            self.wfh,
            [self.index_reg],
            [self.qubit_x_corr_reg[self.vertex_qubit[vertex]]],
        )
        self.X(
            self.vertex_qubit[vertex],
            condition=self.qubit_x_corr_reg[self.vertex_qubit[vertex]][0],
        )

        self.add_wasm_to_reg(
            "get_z_correction",
            self.wfh,
            [self.index_reg],
            [self.qubit_z_corr_reg[self.vertex_qubit[vertex]]],
        )
        self.Z(
            self.vertex_qubit[vertex],
            condition=self.qubit_z_corr_reg[self.vertex_qubit[vertex]][0],
        )

    def corrected_measure(self, vertex: int, t_multiple: int = 0) -> None:
        if vertex in self.output_vertices:
            raise Exception("This is an output qubit and cannot be measured.")

        if not all(self.vertex_measured[:vertex]):
            print(self.vertex_measured[:vertex])
            raise Exception(
                "Measurement order has not been respected. "
                + f"Vertices {[i for i, measured in enumerate(self.vertex_measured[:vertex]) if not measured]} "
                + f"are in the past of {vertex} but have not been measured."
            )

        self._apply_correction(vertex=vertex)

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
        self.vertex_measured[vertex] = True

        # Check that the flow of the vertex being measured has
        # not been measured.
        assert len(list(self.flow_graph.successors(vertex))) == 1
        vertex_flow = list(self.flow_graph.successors(vertex))[0]
        assert not self.vertex_measured[vertex_flow]

        self.add_c_setreg(value=vertex_flow, arg=self.index_reg)
        self.add_wasm_to_reg(
            "update_x_correction",
            self.wfh,
            [self.qubit_meas_reg[self.vertex_qubit[vertex]], self.index_reg],
            [],
        )

        for neighbour in self.entanglement_graph.neighbors(n=vertex_flow):
            if neighbour == vertex:
                continue

            # Check that the vertex being updated has not been measured
            assert not self.vertex_measured[neighbour]

            self.add_c_setreg(value=neighbour, arg=self.index_reg)
            self.add_wasm_to_reg(
                "update_z_correction",
                self.wfh,
                [self.qubit_meas_reg[self.vertex_qubit[vertex]], self.index_reg],
                [],
            )
