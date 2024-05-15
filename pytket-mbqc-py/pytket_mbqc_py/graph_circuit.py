"""
Tools for managing Measurement Based Quantum Computation.
In particular, for managing graph state construction
and automatically adding measurement corrections.
"""

from functools import reduce
from typing import Dict, List, Tuple, Union, cast

import networkx as nx  # type:ignore
from pytket import Qubit
from pytket.circuit.logic_exp import BitLogicExp
from pytket.unit_id import BitRegister, UnitID

from pytket_mbqc_py.random_register_manager import RandomRegisterManager


class GraphCircuit(RandomRegisterManager):
    """Class for the automated construction of MBQC computations.
    In particular only graphs with valid flow can be constructed.
    Graph state construction and measurement corrections are added
    automatically.

    :ivar entanglement_graph: Graph detailing the graph state entanglement.
    :ivar flow_graph: Graph describing the flow dependencies of the graph state.
    :ivar vertex_qubit: List mapping graph vertex to corresponding qubits.
    :ivar vertex_measured: List indicating if vertex has been measured.
    :ivar vertex_x_corr_reg: List mapping vertex index to
        the classical register where the required X correction is stored.
    :ivar vertex_init_reg: List mapping vertex to the register describing
        the state it was initialised in. In particular this is a 3 bit register
        with the 0th entry giving the T rotation, the 1st giving the
        S rotation, and the 2nd giving the Z rotation.
    :ivar measurement_order_list: List of vertex measurement order.
        Entry i corresponds to the position in the order at which
        vertex i is measured. If None then vertex is taken not to be
        measured, which is to say it is an output.
    """

    entanglement_graph: nx.Graph
    flow_graph: nx.DiGraph
    vertex_x_corr_reg: List[BitRegister]
    measurement_order_list: List[Union[None, int]]
    vertex_qubit: List[Qubit]
    vertex_measured: List[bool]
    vertex_init_reg: List[BitRegister]

    def __init__(
        self,
        n_physical_qubits: int,
        n_logical_qubits: int,
    ) -> None:
        """Initialisation method. Creates tools to track
        the graph state structure and the measurement corrections.

        :param n_physical_qubits: The number of physical qubits available.
        :param n_logical_qubits: The number of vertices in the graph state.
            This is used to initialise the appropriate number of registers.
        """
        super().__init__(n_physical_qubits=n_physical_qubits)

        self.entanglement_graph = nx.Graph()
        self.flow_graph = nx.DiGraph()

        self.vertex_qubit = []
        self.vertex_measured = []

        self.measurement_order_list = []

        # We need to save the x correction information long term.
        # This is why there is one register per vertex, as these
        # values cannot be overwritten. They are saved long term
        # as they are needed to calculate z corrections on
        # neighbouring qubits, which could be needed after the
        # vertex has been measured.
        self.vertex_x_corr_reg = []

        # Generate one random register per vertex.
        # When qubits are added they will be initialised in this
        # random register. This is except for the case of input qubits
        # which are initialised in the 0 state, and in which case this
        # register is overwritten.
        self.vertex_init_reg = self.generate_random_registers(n_registers=n_logical_qubits)
        self.add_barrier(
            units=cast(List[UnitID], self.qubits) + cast(List[UnitID], self.bits)
        )

    def get_outputs(self) -> Dict[int, Qubit]:
        """Return the output qubits. Output qubits are those that
        are unmeasured, and which do not have a flow. This should
        be treated as the final step in an MBQC computation. Indeed
        checks are made that the state has been appropriately
        measured. This will also apply the appropriate corrections to
        the output qubits.

        :raises Exception: Raised if there are vertices
            with a measurement order which have not been measured.
        :raises Exception: Raised if too many logical qubits
            were requested upon initialisation.

        :return: Dictionary mapping output vertices to physical
            qubits. These qubits can now be treated as normal circuit
            qubits.
        """

        # Checks if too many registers were created. This would not necessarily
        # be a problem, but it does save on registers, and reduces the
        # resources allocated to randomness generation.
        if len(self.vertex_init_reg) > len(self.vertex_qubit):
            raise Exception(
                f"Too many initialisation registers, {len(self.vertex_init_reg)}, were created. "
                + f"Consider setting n_logical_qubits={len(self.vertex_qubit)} "
                + "upon initialising this class."
            )

        # Outputs should be all of the qubits which have not been measured.
        output_qubits = {
            vertex: qubit
            for vertex, qubit in enumerate(self.vertex_qubit)
            if not self.vertex_measured[vertex]
        }

        # Unmeasured vertices should not have an order.
        # If they do they need to be measured.
        ordered_unmeasured_qubits = [
            vertex
            for vertex in output_qubits.keys()
            if self.measurement_order_list[vertex] is not None
        ]
        if len(ordered_unmeasured_qubits) > 0:
            raise Exception(
                f"Vertices {ordered_unmeasured_qubits} have a measurement order "
                + "but have not been measured. "
                + "Please measure them, or set their order to None."
            )

        # All qubits with flow must be measured. This is to
        # ensure that all corrections have been made.
        # It would be a bug if this is not the case as all vertices with
        # flow should have an order, and all vertices with an order
        # should have been measured.
        unmeasured_flow_vertices = [
            vertex
            for vertex in self._vertices_with_flow
            if not self.vertex_measured[vertex]
        ]
        assert unmeasured_flow_vertices == [], "A vertex with flow is unmeasured."

        # We need to correct all unmeasured qubits.
        for vertex in output_qubits.keys():
            # Corrections are first applied to invert the initialisation
            self.T(self.vertex_qubit[vertex], condition=self.vertex_init_reg[vertex][0])
            self.S(self.vertex_qubit[vertex], condition=self.vertex_init_reg[vertex][0])

            self.S(self.vertex_qubit[vertex], condition=self.vertex_init_reg[vertex][1])

            self.Z(
                self.vertex_qubit[vertex],
                condition=(
                    self.vertex_init_reg[vertex][2]
                    ^ self.vertex_init_reg[vertex][1]
                    ^ self.vertex_init_reg[vertex][0]
                ),
            )

            # Apply X correction according to correction register.
            # These are corrections resulting from the measurements
            self.X(
                self.vertex_qubit[vertex],
                condition=self.vertex_x_corr_reg[vertex][0],
            )

            # Apply Z corrections resulting from measurements.
            # This cannot be done classically as the vertex itself has not
            # been measured.
            self._apply_z_correction(vertex=vertex)

        return output_qubits

    def _add_vertex(self, qubit: Qubit, measurement_order: Union[int, None]) -> int:
        """Add a new vertex to the graph.
        This requires that the vertex is added to the
        entanglement and flow graphs. A register to save
        the vertex X correction to is also created.

        :param qubit: Qubit to be added.
        :param measurement_order: The order at which this vertex will
            be measured. None if the qubit is an output.
        :return: The vertex in the graphs corresponding to this qubit

        :raises Exception: Raised if an insufficient number of initialisation
            registers were created.
        :raises Exception: Raised if the measurement order given
            is already in use. That is to say a vertex measurement order
            must be unique.
        """

        if len(self.vertex_qubit) >= len(self.vertex_init_reg):
            raise Exception(
                "An insufficient number of initialisation registers "
                + "were created. A new vertex cannot be added."
            )

        if (measurement_order is not None) and (
            measurement_order in self.measurement_order_list
        ):
            raise Exception(
                "Measurement order must be unique. "
                + f"A vertex is already measured at order {measurement_order}."
            )

        index = len(self.vertex_qubit)
        self.entanglement_graph.add_node(node_for_adding=index)
        self.flow_graph.add_node(node_for_adding=index)

        x_corr_reg = BitRegister(name=f"x_corr_{index}", size=1)
        self.add_c_register(register=x_corr_reg)

        self.vertex_x_corr_reg.append(x_corr_reg)
        self.vertex_qubit.append(qubit)
        self.vertex_measured.append(False)
        self.measurement_order_list.append(measurement_order)

        return index

    def add_input_vertex(
        self, measurement_order: Union[int, None]
    ) -> Tuple[Qubit, int]:
        """Add a new input vertex to the graph.
        This returns the input qubit created.
        You may perform transformations on this input qubit
        but all transformations must be completed before
        any edges are added to this vertex.

        :param measurement_order: The order at which this vertex will
            be measured.

        :return: The qubit added, and the corresponding index in the graph.
        """
        qubit = self.get_qubit()
        index = self._add_vertex(qubit=qubit, measurement_order=measurement_order)

        # In the case of input qubits, the initialisation is not random.
        # As such the initialisation register should be set to 0.
        self.add_c_setreg(0, self.vertex_init_reg[index])

        return (qubit, index)

    def add_graph_vertex(self, measurement_order: Union[int, None]) -> int:
        """Add a new graph vertex.

        :param measurement_order: The order at which this vertex will
            be measured.

        :return: The index of the vertex added.
        """
        qubit = self.get_qubit()
        self.H(qubit)
        index = self._add_vertex(qubit=qubit, measurement_order=measurement_order)

        # The graph state is randomly initialised based on the
        # initialisation register.
        self.T(qubit, condition=self.vertex_init_reg[index][0])
        self.S(qubit, condition=self.vertex_init_reg[index][1])
        self.Z(qubit, condition=self.vertex_init_reg[index][2])

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
        of vertex_one. This is only not the case if vertex_two is an
        output vertex.

        :param vertex_one: Source vertex.
        :param vertex_two: Target vertex.

        :raises Exception: Raised if the edge points from a non-output
            vertex to an output vertex.
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

        # Check that edges only point towards unmeasured vertices.
        # This ensures that unmeasured vertices do not have flow.
        if (self.measurement_order_list[vertex_one] is None) and (
            self.measurement_order_list[vertex_two] is not None
        ):
            raise Exception(
                "Please ensure that edge point towards unmeasured qubits. "
                + f"In this case {vertex_one} is an output but {vertex_two} "
                + "is not."
            )

        if (
            (self.measurement_order_list[vertex_one] is not None)
            and (self.measurement_order_list[vertex_two] is not None)
            and cast(int, self.measurement_order_list[vertex_one])
            > cast(int, self.measurement_order_list[vertex_two])
        ):
            raise Exception(
                f"{vertex_one} is measured after {vertex_two}. "
                + "The respective measurements orders are "
                + f"{self.measurement_order_list[vertex_one]} and "
                + f"{self.measurement_order_list[vertex_two]}."
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

        # If edges are in the entanglement_graph they should be in
        # the flow_graph. It would be a bug if not.
        assert (
            vertex_one in self.flow_graph.nodes
        ), f"Vertex {vertex_one} missing from flow graph"
        assert (
            vertex_two in self.flow_graph.nodes
        ), f"Vertex {vertex_two} missing from flow graph"

        if self.vertex_measured[vertex_one] or self.vertex_measured[vertex_two]:
            raise Exception(
                "Cannot add edge after measure. "
                + f"In particular {[vertex for vertex in [vertex_one, vertex_two] if self.vertex_measured[vertex]]} have been measured."
            )

        # If this is the first future of vertex_one then it will be taken to be its flow.
        # This is only not the case if vertex_one is not measured, in which
        # case it has no flow.
        # If vertex_two is to be the flow of vertex_one than we must check that neighbours of
        # vertex_two are measured after vertex_one.
        if (self.measurement_order_list[vertex_one] is not None) and (
            vertex_one not in self._vertices_with_flow
        ):
            # Get a list of neighbours of vertex_two which are measured
            # before vertex_one.
            past_neighbours = [
                vertex
                for vertex in self.entanglement_graph.neighbors(n=vertex_two)
                if self.measurement_order_list[vertex]
                < self.measurement_order_list[vertex_one]
            ]
            # If there are any such vertices then an error should be raised.
            if len(past_neighbours) > 0:
                raise Exception(
                    f"Adding the edge ({vertex_one}, {vertex_two}) does not define a valid flow. "
                    + f"In particular {past_neighbours} "
                    + f"are neighbours of {vertex_two} but are in the past of {vertex_one}. "
                    + f"As {vertex_two} would become the flow of {vertex_one} all of the "
                    + f"neighbours of {vertex_two} must be in the past of {vertex_one}."
                )

        # No vertex with flow should be unmeasured.
        # In particular this assert ensures that the check following it
        # always compares integers.
        target_inverse_flow = list(self.flow_graph.predecessors(vertex_two))
        assert all(
            self.measurement_order_list[inverse_flow] is not None
            for inverse_flow in target_inverse_flow
        ), f"An inverse flow of {vertex_two} is not measured"

        # vertex_one is a neighbour of vertex_two. As such vertex_one must be measured after
        # any vertices of which vertex_two is its flow. If it is not measured
        # then it is certainly measured after vertex_two.
        if (self.measurement_order_list[vertex_one] is not None) and any(
            self.measurement_order_list[flow_inverse]
            > self.measurement_order_list[vertex_one]
            for flow_inverse in self.flow_graph.predecessors(vertex_two)
        ):
            raise Exception(
                f"Adding the edge {(vertex_one, vertex_two)} does not define a valid flow. "
                f"In particular {vertex_two} is the flow of {list(self.flow_graph.predecessors(vertex_two))}, "
                f"some of which are measured after {vertex_one}."
            )

        # If this is the first future of vertex_one then it is taken to be its flow.
        if (self.measurement_order_list[vertex_one] is not None) and (
            vertex_one not in self._vertices_with_flow
        ):
            self.flow_graph.add_edge(
                u_of_edge=vertex_one,
                v_of_edge=vertex_two,
            )

        self.CZ(self.vertex_qubit[vertex_one], self.vertex_qubit[vertex_two])

        self.entanglement_graph.add_edge(
            u_of_edge=vertex_one,
            v_of_edge=vertex_two,
        )

    def _get_z_correction_expression(self, vertex: int) -> Union[None, BitLogicExp]:
        """Create logical expression by taking the parity of
        the X corrections that have to be applied to the neighbouring
        qubits. If there are no neighbours then None will be returned.

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
        :raises Exception: Raised if this vertex has already been measured.
        :raises Exception: Raised if this vertex has no measurement order
        :raises Exception: Raised if this vertex does not have flow.
        :raises Exception: Raised if this vertex is not the first measured,
            but there is no vertex with order one less than the one considered.
            This would be raised if the order the vertices are measured
            in is not sequential.
        :raises Exception: Raised if there are vertex in the past of this
            one which have not been measured.
        """
        # Check that the vertex being measured has not already been measured.
        if self.vertex_measured[vertex]:
            raise Exception(
                f"Vertex {vertex} has already been measured and cannot be measured again."
            )

        if self.measurement_order_list[vertex] is None:
            raise Exception(
                "This vertex does not have a measurement order and "
                + "cannot be measured."
            )

        if vertex not in self._vertices_with_flow:
            raise Exception(f"Vertex {vertex} has no flow and cannot be measured.")

        # Assert that all vertices with order greater than that of the qubit
        # being measured have not yet been measured.
        assert all(
            not later_vertex_measured
            for later_vertex_measured, later_vertex_order in zip(
                self.measurement_order_list, self.vertex_measured
            )
            if (
                (later_vertex_order is not None)
                # This cast is safe as we have established above that the
                # vertex has a measurement order.
                and (
                    later_vertex_order > cast(int, self.measurement_order_list[vertex])
                )
            )
        ), "There are vertices with higher order which have already been measured"

        if (
            # safe to cast as we have check that the order is not None.
            (cast(int, self.measurement_order_list[vertex]) > 0)
            and (self.measurement_order_list[vertex] not in self.measurement_order_list)
        ):
            raise Exception(
                f"Vertex {vertex} has order "
                + f"{self.measurement_order_list[vertex]} "
                + f"but there is no vertex with order {self.measurement_order_list[vertex]}."
            )

        # List the vertices which have order less than the vertex considered,
        # but which have not yet been measured.
        unmeasured_earlier_vertex_list = [
            earlier_vertex
            for earlier_vertex, earlier_vertex_order in enumerate(
                self.measurement_order_list
            )
            if (
                (earlier_vertex_order is not None)
                and (
                    earlier_vertex_order
                    < cast(int, self.measurement_order_list[vertex])
                )
                and (not self.vertex_measured[earlier_vertex])
            )
        ]
        if len(unmeasured_earlier_vertex_list) > 0:
            raise Exception(
                f"The vertices {unmeasured_earlier_vertex_list} are ordered "
                + f"to be measured before vertex {vertex}, "
                + "but are unmeasured."
            )

        # Apply X correction according to correction register.
        # This is to correct for measurement outcomes.
        self.X(
            self.vertex_qubit[vertex],
            condition=self.vertex_x_corr_reg[vertex][0],
        )

        self.T(
            self.vertex_qubit[vertex],
            # Required to invert random T from initialisation.
            condition=self.vertex_init_reg[vertex][0],
        )
        self.S(
            self.vertex_qubit[vertex],
            # Required to invert random T from initialisation.
            # This additional term is required to account for the case where
            # the correcting T is commuted through an X correction.
            condition=(
                self.vertex_init_reg[vertex][0] & self.vertex_x_corr_reg[vertex][0]
            ),
        )

        self.S(
            self.vertex_qubit[vertex],
            # Required to invert random T from initialisation.
            condition=self.vertex_init_reg[vertex][0],
        )

        self.S(
            self.vertex_qubit[vertex],
            # Required to invert random S from initialisation.
            condition=self.vertex_init_reg[vertex][1],
        )

        self.Z(
            self.vertex_qubit[vertex],
            condition=(
                # Required to invert random T from initialisation.
                self.vertex_init_reg[vertex][0]
                # Required to invert random S from initialisation.
                ^ self.vertex_init_reg[vertex][1]
                # Required to invert random Z from initialisation.
                ^ self.vertex_init_reg[vertex][2]
                # Required to invert random S from initialisation.
                # This additional term is required to account for the case where
                # the correcting S is commuted through an X correction.
                ^ (self.vertex_init_reg[vertex][1] & self.vertex_x_corr_reg[vertex][0])
                # Required to invert random T from initialisation.
                # This additional term is required to account for the case where
                # the correcting S is commuted through an X correction.
                ^ (self.vertex_init_reg[vertex][0] & self.vertex_x_corr_reg[vertex][0])
                # Required to invert random T from initialisation.
                # This additional term is required to account for the case where
                # the correcting T is commuted through an X correction.
                ^ (self.vertex_init_reg[vertex][0] & self.vertex_x_corr_reg[vertex][0])
            ),
        )

        # Rotate measurement basis.
        # TODO: These measurements should be combined with the above
        # so that the measurement angles are hidden by the initialisation
        # angles.
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
