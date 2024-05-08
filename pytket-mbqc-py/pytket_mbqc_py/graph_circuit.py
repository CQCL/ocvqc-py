"""
Tools for managing Measurement Based Quantum Computation.
In particular, for managing graph state construction
and automatically adding measurement corrections.
"""

from functools import reduce
from typing import Dict, List, Tuple, Union

import networkx as nx  # type:ignore
from pytket import Qubit
from pytket.circuit.logic_exp import BitLogicExp
from pytket.unit_id import BitRegister

from pytket_mbqc_py.random_register_manager import RandomRegisterManager


class GraphCircuit(RandomRegisterManager):
    """Class for the automated construction of MBQC computations.
    In particular only graphs with valid flow can be constructed.
    Graph state construction and measurement corrections are added
    automatically. A child of
    :py:class:`~pytket_mbqc_py.qubit_manager.RandomRegisterManager`.

    :ivar entanglement_graph: Graph detailing the graph state entanglement.
    :ivar flow_graph: Graph describing the flow dependencies of the graph state.
    :ivar vertex_qubit: List mapping graph vertex to corresponding qubits.
    :ivar vertex_measured: List indicating if vertex has been measured.
    :ivar vertex_x_corr_reg: List mapping vertex index to
        the classical register where the required X correction is stored.
    :ivar vertex_random_reg: List mapping vertex to the corresponding
        register of random bits.
    """

    entanglement_graph: nx.Graph
    flow_graph: nx.DiGraph
    vertex_x_corr_reg: List[BitRegister]

    def __init__(
        self,
        n_physical_qubits: int,
        n_registers: int,
    ) -> None:
        """Initialisation method. Creates tools to track
        the graph state structure and the measurement corrections.

        :param n_physical_qubits: The number of physical qubits available.
        :n_registers: The number of random registers to generate. Defaults
            to 100 random registers.
        """
        super().__init__(n_physical_qubits=n_physical_qubits)

        self.entanglement_graph = nx.Graph()
        self.flow_graph = nx.DiGraph()

        self.vertex_qubit: List[Qubit] = []
        self.vertex_measured: List[bool] = []

        # We need to save the x correction information long term.
        # This is why there is one register per vertex, as these
        # values cannot be overwritten. They are saved long term
        # as they are needed to calculate z corrections on
        # neighbouring qubits, which could be needed after the
        # vertex has been measured.
        self.vertex_x_corr_reg: List[BitRegister] = []

        self.vertex_random_reg = list(
            self.generate_random_registers(n_registers=n_registers)
        )
        self.add_barrier(units=self.qubits)

    def get_outputs(self) -> Dict[int, Qubit]:
        """Return the output qubits. Output qubits are those that
        are unmeasured, and which do not have a flow. This should
        be treated as the final step in an MBQC computation. Indeed
        check are made that the state has been appropriately
        measured. This will also apply the appropriate corrections to
        the output qubits.

        :raises Exception: Raised if there there are qubits
            with flow which have not been measured. All
            qubits with flow must be measured before the MBQC
            computation can be finalised and the outputs recovered.
        :return: Dictionary mapping output vertices to physical
            qubits. These qubits can now be treated as normal circuit
            qubits.
        """

        # All qubits with flow must be measured. This is to
        # ensure that all correction have been made.
        unmeasured_flow_vertices = [
            vertex
            for vertex in self._vertices_with_flow
            if not self.vertex_measured[vertex]
        ]
        if len(unmeasured_flow_vertices) > 0:
            raise Exception(
                "Only output vertices can be unmeasured. "
                + f"In particular {unmeasured_flow_vertices} "
                + "have flow but are not measured."
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
            self.T(
                self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][0]
            )
            self.S(
                self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][0]
            )
            self.Z(
                self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][0]
            )

            self.S(
                self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][1]
            )
            self.Z(
                self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][1]
            )

            self.Z(
                self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][2]
            )

            self._apply_x_correction(vertex=vertex)
            self._apply_z_correction(vertex=vertex)

        return output_qubits

    def _add_vertex(self, qubit: Qubit) -> int:
        """Add a new vertex to the graph.
        This requires that the vertex is added to the
        entanglement and flow graphs. A register to save
        the vertex X correction to is also created.

        :param qubit: Qubit to be added.
        :return: The vertex in the graphs corresponding to this qubit
        """
        self.vertex_qubit.append(qubit)
        self.vertex_measured.append(False)

        index = len(self.vertex_qubit) - 1
        self.entanglement_graph.add_node(node_for_adding=index)
        self.flow_graph.add_node(node_for_adding=index)

        x_corr_reg = BitRegister(name=f"x_corr_{index}", size=1)
        self.vertex_x_corr_reg.append(x_corr_reg)
        self.add_c_register(register=x_corr_reg)

        return index

    def add_input_vertex(self) -> Tuple[Qubit, int]:
        """Add a new input vertex to the graph.
        This returns the input qubit created.
        You may perform transformations on this input qubit
        but all transformations must be completed before
        any edges are added to this vertex.

        :return: The qubit added, and the corresponding index in the graph.
        """
        qubit = self.get_qubit()
        index = self._add_vertex(qubit=qubit)

        # In the case of input qubits, the initialisation is not random.
        # As such the measurement angle does not need to be corrected.
        self.add_c_setreg(0, self.vertex_random_reg[index])

        return (qubit, index)

    def add_graph_vertex(self) -> int:
        """Add a new graph vertex.

        :return: The index of the vertex added.
        """
        qubit = self.get_qubit()
        self.H(qubit)
        index = self._add_vertex(qubit=qubit)

        self.T(qubit, condition=self.vertex_random_reg[index][0])
        self.S(qubit, condition=self.vertex_random_reg[index][1])
        self.Z(qubit, condition=self.vertex_random_reg[index][2])

        return index

    @property
    def _vertices_with_flow(self) -> List[int]:
        """List of qubits which have flow."""
        return list(
            set(
                predecessor
                for vertex in self.flow_graph.nodes
                for predecessor in self.flow_graph.predecessors(vertex)
            )
        )

    def add_edge(self, vertex_one: int, vertex_two: int) -> None:
        """Add an edge in the graph between the given vertices.
        This will make a number of checks to ensure that the
        resulting graph state is valid.

        Note that if vertex_two is the first neighbour of
        vertex_one then vertex_two will be taken to be the flow
        of vertex_one.

        :param vertex_one: Source vertex.
        :param vertex_two: Target vertex.
        :raises Exception: Raised the edge acts into the past.
            I.e. if vertex_two is measured before vertex_one.
            Equivalently if vertex_two < vertex_one
        :raises Exception: Raised if vertex_one does not exist in the graph.
        :raises Exception: Raised if vertex_two does not exist in the graph.
        :raises Exception: Raised if vertex_one or vertex_two
            has been measured.
        :raises Exception: Raised if vertex_two is the flow of vertex_one
            but their are neighbours of vertex_two which are measured
            before vertex_one. This does not allow correction from
            vertex_one to propagate to those neighbours.
        :raises Exception: Raised if an inverse flow of vertex_two
            is measured after vertex_one. This would not allow corrections
            from that inverse flow to propagate to vertex_one.
        """
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
            raise Exception(
                "Cannot add edge after measure."
                + f"In particular {[vertex for vertex in [vertex_one, vertex_two] if self.vertex_measured[vertex]]} have been measured."
            )

        # If this is the first future of vertex_one then it will be taken to be its flow.
        # If vertex_two is to be the flow of vertex_one than we must check that neighbours of
        # vertex_two are measured after vertex_one.
        if vertex_one not in self._vertices_with_flow:
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
                f"In particular {vertex_two} is the flow of {list(self.flow_graph.predecessors(vertex_two))}, "
                f"some of which are measured after {vertex_one}."
            )

        # If this is the first future of vertex_one then it is taken to be its flow.
        if vertex_one not in self._vertices_with_flow:
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
        """Apply X correction. This correction is drawn from
        the x correction register which is altered
        by the corrected measure method as appropriate.

        :param vertex: The vertex to be corrected.
        """
        self.X(
            self.vertex_qubit[vertex],
            condition=self.vertex_x_corr_reg[vertex][0],
        )

    def _get_z_correction_expression(self, vertex: int) -> Union[None, BitLogicExp]:
        """Create logical expression by taking the parity of
        the X corrections that have to be applied to the neighbouring
        qubits.

        :param vertex: Vertex to be corrected.
        :return: Logical expression calculating the parity
            of the neighbouring x correction registers.
        """

        neighbour_reg_list = [
            self.vertex_x_corr_reg[neighbour][0]
            for neighbour in self.entanglement_graph.neighbors(n=vertex)
        ]

        # This happens of this vertex has no neighbours.
        if len(neighbour_reg_list) == 0:
            return None

        return reduce(lambda a, b: a ^ b, neighbour_reg_list)

    def _apply_z_correction(self, vertex: int) -> None:
        """Apply Z correction on qubit. This correction is calculated
        using the X corrections that have to be applied to the neighbouring
        qubits.

        :param vertex: Vertex to be corrected.
        """
        condition = self._get_z_correction_expression(vertex=vertex)
        if condition is not None:
            self.Z(
                self.vertex_qubit[vertex],
                condition=condition,
            )

    def _apply_classical_z_correction(self, vertex: int) -> None:
        """Apply Z correction on measurement result. This correction is calculated
        using the X corrections that have to be applied to the neighbouring
        qubits.

        :param vertex: Vertex to be corrected.
        """
        condition = self._get_z_correction_expression(vertex=vertex)
        if condition is not None:
            self.add_classicalexpbox_bit(
                expression=self.qubit_meas_reg[self.vertex_qubit[vertex]][0]
                ^ condition,
                target=[self.qubit_meas_reg[self.vertex_qubit[vertex]][0]],
            )

    def corrected_measure(self, vertex: int, t_multiple: int = 0) -> None:
        """Perform a measurement, applying the appropriate corrections.
        Corrections required on the relevant flow qubit are also updated.

        :param vertex: Vertex to be measured.
        :param t_multiple: The angle in which to measure, defaults to 0.
            This defines the rotated hadamard basis to measure in.
        :type t_multiple: int, optional
        :raises Exception: Raised if this vertex has already been measured.
        :raises Exception: Raised if there are vertex in the past of this
            one which have not been measured.
        :raises Exception: Raised if this vertex does not have flow.
            Vertices without flow are output qubits.
        """
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

        self.T(self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][0])
        self.S(self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][0])
        self.Z(self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][0])

        self.S(self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][1])
        self.Z(self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][1])

        self.Z(self.vertex_qubit[vertex], condition=self.vertex_random_reg[vertex][2])

        # This is actually optional for graph vertices
        # as the X correction commutes with the hadamard
        # basis measurement.
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

        # measure and apply the necessary z corrections
        # classically.
        self.managed_measure(qubit=self.vertex_qubit[vertex])
        self._apply_classical_z_correction(vertex=vertex)
        self.vertex_measured[vertex] = True

        # Check that the vertex has at most one flow vertex
        assert len(list(self.flow_graph.successors(vertex))) <= 1

        # Check that his vertex has flow.
        # If it does not then this is an output vertex
        # or this is not a valid graph.
        if len(list(self.flow_graph.successors(vertex))) == 0:
            raise Exception(
                f"Vertex {vertex} has no flow. "
                "It is not possible to perform a corrected measure of a qubit without flow. "
                "Please give this vertex a flow, or use the get_output to perform the necessary corrections."
            )

        vertex_flow = list(self.flow_graph.successors(vertex))[0]

        # Check that the flow of the vertex being measured has
        # not been measured.
        assert not self.vertex_measured[vertex_flow]

        # Add an x correction to the flow of the
        # measured vertex.
        self.add_classicalexpbox_bit(
            self.qubit_meas_reg[self.vertex_qubit[vertex]][0]
            ^ self.vertex_x_corr_reg[vertex_flow][0],
            [self.vertex_x_corr_reg[vertex_flow][0]],
        )
