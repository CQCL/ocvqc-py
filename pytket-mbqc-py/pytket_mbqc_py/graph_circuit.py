"""
Tools for managing Measurement Based Quantum Computation.
In particular, for managing graph state construction
and automatically adding measurement corrections.
"""

from collections import Counter
from functools import reduce
from typing import List, Optional, Union, cast

import networkx as nx  # type:ignore
from pytket import Qubit
from pytket.backends.backendresult import BackendResult
from pytket.circuit.logic_exp import BitLogicExp, BitNot, BitZero
from pytket.unit_id import Bit, BitRegister, UnitID
from pytket.utils.outcomearray import OutcomeArray

from pytket_mbqc_py.qubit_manager import QubitManager


class GraphCircuit(QubitManager):
    """Class for the automated construction of MBQC computations.
    In particular only graphs with valid flow can be constructed.
    Graph state construction and measurement corrections are added
    automatically.

    :ivar entanglement_graph: Graph detailing the graph state entanglement.
    :ivar flow_graph: Graph describing the flow dependencies of the graph state.
    :ivar vertex_qubit: List mapping graph vertex to corresponding qubits.
    :ivar vertex_measured: List indicating if vertex has been measured.
    :ivar vertex_reg: List mapping vertex to the its register.
        In particular this is a 5 bit register with the 0th entry storing
        the measurement result, the 1st giving the random T rotation, the 2nd
        giving the S rotation, the 3rd giving the Z rotation, and
        the 4th storing the X correction.
    :ivar measurement_order_list: List of vertex measurement order.
        Entry i corresponds to the position in the order at which
        vertex i is measured. If None then vertex is taken not to be
        measured, which is to say it is an output.
    :ivar vertex_is_dummy_list: List of lists of boolean values.
        Each sublist can be taken to indicate a colour, while the lists
        together define a colouring.
    :ivar is_test_bit: Bit register indicating if this is a test round.
    """

    entanglement_graph: nx.Graph
    flow_graph: nx.DiGraph
    measurement_order_list: List[Union[None, int]]
    vertex_qubit: List[Qubit]
    vertex_measured: List[bool]
    vertex_reg: List[BitRegister]
    vertex_is_dummy_list: List[List[bool]]
    is_test_bit: Bit

    def __init__(
        self,
        n_physical_qubits: int,
        n_logical_qubits: int,
        vertex_is_dummy_list: List[List[bool]] = [],
    ) -> None:
        """Initialisation method. Creates tools to track
        the graph state structure and the measurement corrections.

        :param n_physical_qubits: The number of physical qubits available.
        :param n_logical_qubits: The number of vertices in the graph state.
            This is used to initialise the appropriate number of registers.
        :param vertex_is_dummy_list: List of lists of boolean values.
            Each sublist can be taken to indicate a colour, while the lists
            together define a colouring.
        """
        super().__init__(n_physical_qubits=n_physical_qubits)

        self.entanglement_graph = nx.Graph()
        self.flow_graph = nx.DiGraph()

        self.vertex_qubit = []
        self.vertex_measured = []

        self.measurement_order_list = []

        if vertex_is_dummy_list != []:
            vertex_not_test = [
                vertex
                for vertex in range(n_physical_qubits)
                if all(
                    vertex_is_dummy[vertex] for vertex_is_dummy in vertex_is_dummy_list
                )
            ]
            if len(vertex_not_test) > 0:
                raise Exception(
                    f"The vertices {vertex_not_test} are never test qubits. "
                )

        self.vertex_is_dummy_list = vertex_is_dummy_list

        # There is one register per vertex.
        # The bits in the register are as follows:
        #   - 0 : Storage for measurement results.
        #   - 1 : First random initialisation bit.
        #   - 2 : Second random initialisation bit.
        #   - 3 : Third random initialisation bit.
        #   - 4 : X correction register.
        #   - 5 : Is dummy register
        #   - 6 : Dummy randomness
        # When qubits are added they will be initialised in this
        # random register. This is except for the case of input qubits
        # which are initialised in the 0 state, and in which case this
        # register is overwritten.
        # We need to save the x correction information long term.
        # This is why there is one register per vertex, as these
        # values cannot be overwritten. They are saved long term
        # as they are needed to calculate z corrections on
        # neighbouring qubits, which could be needed after the
        # vertex has been measured.
        self.vertex_reg = [
            self.add_c_register(
                name=f"vertex_{vertex_index}",
                size=7,
            )
            for vertex_index in range(n_logical_qubits)
        ]

        self.populate_random_bits(
            bit_list=[
                bit
                for register in self.vertex_reg
                for bit in register.to_list()[1:4] + [register.to_list()[6]]
            ]
        )

        self.is_test_bit = Bit(name="is test bit", index=0)
        self.add_bit(id=self.is_test_bit)

        if len(self.vertex_is_dummy_list) == 0:
            self.add_c_setbits(
                values=[False],
                args=[self.is_test_bit],
            )

        elif len(self.vertex_is_dummy_list) == 2:
            self.populate_random_bits(bit_list=[self.is_test_bit])

            for vertex_is_dummy in self.vertex_is_dummy_list:
                if not len(vertex_is_dummy) == n_logical_qubits:
                    raise Exception(
                        "There must be a colour for each of the logical qubits. "
                        f"In this case there are {n_logical_qubits} "
                        f"logical qubits and {len(vertex_is_dummy)} colours."
                    )

            colour_choice_bit = Bit(name="colour_choice_bit", index=0)
            self.add_bit(colour_choice_bit)
            self.populate_random_bits(bit_list=[colour_choice_bit])

            self.add_c_setbits(
                values=self.vertex_is_dummy_list[0],
                args=[register[5] for register in self.vertex_reg],
                condition=colour_choice_bit & self.is_test_bit,
            )

            self.add_c_setbits(
                values=self.vertex_is_dummy_list[1],
                args=[register[5] for register in self.vertex_reg],
                condition=BitNot(colour_choice_bit) & self.is_test_bit,
            )

        else:
            raise Exception("You can only use 0 or two colours.")

        # Isolate the initialisation randomness generation from the
        # rest of the circuit.
        self.add_barrier(
            units=cast(List[UnitID], self.qubits) + cast(List[UnitID], self.bits)
        )

    def populate_random_bits(
        self,
        bit_list: List[Bit],
        max_n_randomness_qubits: int = 2,
    ) -> None:
        """Populate the given bits with random values. This is achieved
        by initialising hadamard basis plus states and measuring them.

        :param bit_list: List of bits to be populated with randomness.
        :param max_n_randomness_qubits: The maximum number of qubits to use
            to generate randomness. If a number of qubits less than this
            number are actually available then the number available will
            be used., defaults to 2
        :raises Exception: Raised if there are no qubits left to use to
            generate randomness.
        """

        if len(self.available_qubit_list) == 0:
            raise Exception(
                "There are no unused qubits "
                + "which can be used to generate randomness."
            )

        # The number of qubits used is the smaller of the maximum number set
        # by the user, or the number which is available.
        n_randomness_qubits = min(
            len(self.available_qubit_list), max_n_randomness_qubits
        )

        def generate_randomness(list_chunk: List[Bit]) -> None:
            """Write randomness to given bits.

            :param list_chunk: List of bits to write to. Note that this should
                be of length at most the number of qubits to be used to
                generate randomness.
            """
            # Initialise all qubits.
            qubit_list = [
                self.get_qubit(measure_bit=measure_bit) for measure_bit in list_chunk
            ]

            # For each qubit, initialise, rotate, and measure.
            for qubit in qubit_list:
                self.H(qubit=qubit)
                self.managed_measure(qubit=qubit)

        # chunk indicates how many chunks have been completed
        # before the beginning of the following for loop
        # not even chunk 0 had been populated. As such we have -1 here.
        chunk = -1

        # We repeatedly initialise and measure qubits in groups
        # of size n_randomness_qubits. This allows randomness generation
        # to be done in parallel where possible.
        for chunk in range(len(bit_list) // n_randomness_qubits):
            list_chunk = bit_list[
                chunk * max_n_randomness_qubits : (chunk + 1) * n_randomness_qubits
            ]
            generate_randomness(list_chunk=list_chunk)

        # Generate the remaining randomness.
        list_chunk = bit_list[(chunk + 1) * n_randomness_qubits :]
        generate_randomness(list_chunk=list_chunk)

    def _add_vertex(self, qubit: Qubit, measurement_order: Union[int, None]) -> None:
        """Add a new vertex to the graph.
        This requires that the vertex is added to the
        entanglement and flow graphs. A register to save
        the vertex X correction to is also created.

        :param qubit: Qubit to be added.
        :param measurement_order: The order at which this vertex will
            be measured. None if the qubit is an output.
        :return: The vertex in the graphs corresponding to this qubit
        """

        index = len(self.vertex_qubit)
        self.entanglement_graph.add_node(node_for_adding=index)
        self.flow_graph.add_node(node_for_adding=index)

        self.vertex_qubit.append(qubit)
        self.vertex_measured.append(False)
        self.measurement_order_list.append(measurement_order)

    def add_graph_vertex(self, measurement_order: Union[int, None]) -> int:
        """Add a new graph vertex.

        :param measurement_order: The order at which this vertex will
            be measured.

        :return: The index of the vertex added.
        """
        self._add_vertex_check(measurement_order)

        index = len(self.vertex_qubit)

        # TODO: Add a check that vertex_is_dummy is long enough.

        qubit = self.get_qubit(measure_bit=self.vertex_reg[index][0])
        self.H(qubit, condition=BitNot(self.vertex_reg[index][5]))
        self.X(qubit, condition=self.vertex_reg[index][5] & self.vertex_reg[index][6])
        self._add_vertex(qubit=qubit, measurement_order=measurement_order)

        # The graph state is randomly initialised based on the
        # initialisation register.
        self.T(qubit, condition=self.vertex_reg[index][1])
        self.S(qubit, condition=self.vertex_reg[index][2])
        self.Z(qubit, condition=self.vertex_reg[index][3])

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

        if any(
            (not (vertex_is_dummy[vertex_one] or vertex_is_dummy[vertex_two]))
            for vertex_is_dummy in self.vertex_is_dummy_list
        ):
            raise Exception(
                f"Vertex {vertex_one} and vertex {vertex_two} " "have the same colour."
            )

        if vertex_one not in self.entanglement_graph.nodes:
            raise Exception(
                f"There is no vertex with the index {vertex_one}. "
                + f"Existing vertices are {list(self.entanglement_graph.nodes)}."
            )

        if vertex_two not in self.entanglement_graph.nodes:
            raise Exception(
                f"There is no vertex with the index {vertex_two}. "
                + f"Existing vertices are {list(self.entanglement_graph.nodes)}."
            )

        # If edges are in the entanglement_graph they should be in
        # the flow_graph. It would be a bug if not.
        assert (
            vertex_one in self.flow_graph.nodes
        ), f"Vertex {vertex_one} missing from flow graph"
        assert (
            vertex_two in self.flow_graph.nodes
        ), f"Vertex {vertex_two} missing from flow graph"

        # Check that edges only point towards output vertices.
        # This ensures that output vertices do not have flow.
        if (self.measurement_order_list[vertex_one] is None) and (
            self.measurement_order_list[vertex_two] is not None
        ):
            raise Exception(
                "Edges must point towards output qubits. "
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
                + f"{self.measurement_order_list[vertex_two]}. "
                + "Cannot add edge into the past."
            )

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

    def _get_z_correction_expression(self, vertex: int) -> BitLogicExp:
        """Create logical expression by taking the parity of
        the X corrections that have to be applied to the neighbouring
        qubits. If there are no neighbours then None will be returned.

        :param vertex: Vertex to be corrected.
        :return: Logical expression calculating the parity
            of the neighbouring x correction registers.
        """

        neighbour_reg_list = [
            self.vertex_reg[neighbour][4]
            for neighbour in self.entanglement_graph.neighbors(n=vertex)
        ]

        # This happens of this vertex has no neighbours.
        if len(neighbour_reg_list) == 0:
            return BitZero()

        return reduce(lambda a, b: a ^ b, neighbour_reg_list)

    def _apply_classical_z_correction(self, vertex: int) -> None:
        """Apply Z correction on measurement result. This correction is calculated
        using the X corrections that have to be applied to the neighbouring
        qubits.

        :param vertex: Vertex to be corrected.
        """
        condition = self._get_z_correction_expression(vertex=vertex)
        self.add_classicalexpbox_bit(
            expression=self.vertex_reg[vertex][0] ^ condition,
            target=[self.vertex_reg[vertex][0]],
        )

    def _get_dummy_correction_expression(self, vertex: int) -> BitLogicExp:
        """Create correction expressions. A correction is needed is the
        neighbour is a dummy and was initialised in the 1 state.
        The relevant expression is obtained by combining the value for
        all neighbours.

        :param vertex: Vertex to be corrected.
        :type vertex: int
        :return: Logical expression giving correction.
        :rtype: BitLogicExp
        """
        neighbour_reg_list = [
            self.vertex_reg[neighbour][5] & self.vertex_reg[neighbour][6]
            for neighbour in self.entanglement_graph.neighbors(n=vertex)
        ]

        # This happens of this vertex has no neighbours.
        if len(neighbour_reg_list) == 0:
            return BitZero()

        return reduce(lambda a, b: a ^ b, neighbour_reg_list)

    def _apply_dummy_correction(self, vertex: int) -> None:
        """Apply dummy correction expression. This correction is applied
        if the vertex itself is not a dummy vertex.

        :param vertex: _description_
        :type vertex: int
        """
        condition = self._get_dummy_correction_expression(vertex=vertex)
        self.add_classicalexpbox_bit(
            expression=self.vertex_reg[vertex][0]
            ^ (condition & BitNot(self.vertex_reg[vertex][5])),
            target=[self.vertex_reg[vertex][0]],
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

        vertex_measure_order = self.measurement_order_list[vertex]

        if vertex_measure_order is None:
            assert (
                vertex not in self._vertices_with_flow
            ), "Output vertices should not have flow."

        if (vertex not in self._vertices_with_flow) and (
            vertex_measure_order is not None
        ):
            raise Exception(
                f"Vertex {vertex} is not an output and has no flow. "
                "As such it cannot be measured. "
            )

        # List the vertices which have order less than the vertex considered,
        # but which have not yet been measured.
        unmeasured_earlier_vertex_list = [
            earlier_vertex
            for earlier_vertex, earlier_vertex_order in enumerate(
                self.measurement_order_list
            )
            if (earlier_vertex_order is not None)
            and (not self.vertex_measured[earlier_vertex])
        ]

        if vertex_measure_order is not None:
            unmeasured_earlier_vertex_list = [
                earlier_vertex
                for earlier_vertex in unmeasured_earlier_vertex_list
                if cast(int, self.measurement_order_list[earlier_vertex])
                < vertex_measure_order
            ]

        if len(unmeasured_earlier_vertex_list) > 0:
            raise Exception(
                f"The vertices {unmeasured_earlier_vertex_list} are ordered "
                + f"to be measured before vertex {vertex}, "
                + "but are unmeasured."
            )

        if vertex_measure_order is not None:
            measured_later_vertex_list = [
                later_vertex
                for later_vertex, later_vertex_order in enumerate(
                    self.measurement_order_list
                )
                if self.vertex_measured[later_vertex]
                and (
                    later_vertex_order is None
                    or later_vertex_order > vertex_measure_order
                )
            ]

            assert len(measured_later_vertex_list) == 0, (
                f"The vertices {measured_later_vertex_list} are ordered "
                f"to be measured after vertex {vertex}, "
                "but are measured."
            )

        if vertex_measure_order is not None:
            if (vertex_measure_order > 0) and (
                (vertex_measure_order - 1) not in self.measurement_order_list
            ):
                raise Exception(
                    f"Vertex {vertex} has order "
                    + f"{vertex_measure_order} "
                    + f"but there is no vertex with order {vertex_measure_order - 1}."
                )

        # Required to invert random T from initialisation.
        self.T(
            self.vertex_qubit[vertex],
            condition=self.vertex_reg[vertex][1],
        ).S(
            self.vertex_qubit[vertex],
            condition=self.vertex_reg[vertex][1],
        ).Z(
            self.vertex_qubit[vertex],
            condition=self.vertex_reg[vertex][1],
        )

        # Required to invert random S from initialisation.
        self.S(
            self.vertex_qubit[vertex],
            condition=self.vertex_reg[vertex][2],
        ).Z(
            self.vertex_qubit[vertex],
            condition=self.vertex_reg[vertex][2],
        )

        # Required to invert random Z from initialisation.
        self.Z(
            self.vertex_qubit[vertex],
            condition=self.vertex_reg[vertex][3],
        )

        # Rotate measurement basis.
        # Note that measurement angle is inverted if a correction is required.
        # Note that there is no measurement rotation in the case of test rounds.
        # TODO: These measurements should be combined with the above
        # so that the measurement angles are hidden by the initialisation
        # angles.
        inverse_t_multiple = 8 - t_multiple
        inverse_t_multiple = inverse_t_multiple % 8
        if inverse_t_multiple // 4:
            self.Z(self.vertex_qubit[vertex], condition=BitNot(self.is_test_bit))
        if (inverse_t_multiple % 4) // 2:
            self.S(self.vertex_qubit[vertex], condition=BitNot(self.is_test_bit)).Z(
                self.vertex_qubit[vertex],
                condition=self.vertex_reg[vertex][4] & BitNot(self.is_test_bit),
            )
        if inverse_t_multiple % 2:
            self.T(self.vertex_qubit[vertex], condition=BitNot(self.is_test_bit)).S(
                self.vertex_qubit[vertex],
                condition=self.vertex_reg[vertex][4] & BitNot(self.is_test_bit),
            ).Z(
                self.vertex_qubit[vertex],
                condition=self.vertex_reg[vertex][4] & BitNot(self.is_test_bit),
            )
        self.H(self.vertex_qubit[vertex])

        # measure and apply the necessary z corrections
        # classically.
        self.managed_measure(qubit=self.vertex_qubit[vertex])
        self._apply_classical_z_correction(vertex=vertex)
        self._apply_dummy_correction(vertex=vertex)
        self.vertex_measured[vertex] = True

        # Check that the vertex has at most one flow vertex
        assert len(list(self.flow_graph.successors(vertex))) <= 1

        if vertex in self._vertices_with_flow:
            assert (
                vertex_measure_order is not None
            ), f"Vertex {vertex} has flow but is an output. "

            vertex_flow = list(self.flow_graph.successors(vertex))[0]

            # Check that the flow of the vertex being measured has
            # not been measured.
            assert not self.vertex_measured[vertex_flow]

            # Add an x correction to the flow of the
            # measured vertex.
            # Note that dummy vertices should not get nor receive corrections
            # as they are disentangled.
            self.add_classicalexpbox_bit(
                (
                    self.vertex_reg[vertex][0]
                    & BitNot(self.vertex_reg[vertex][5])
                    & BitNot(self.vertex_reg[vertex_flow][5])
                )
                ^ self.vertex_reg[vertex_flow][4],
                [self.vertex_reg[vertex_flow][4]],
            )

    def _add_vertex_check(self, measurement_order: Union[int, None]) -> None:
        """Runs checks that there are enough initialisation
            registers, and that the given order is unique.

        :param measurement_order: The order at which this vertex will
            be measured. None if the qubit is an output.
        :raises Exception: Raised if an insufficient number of initialisation
            registers were created.
        :raises Exception: Raised if the measurement order given
            is already in use. That is to say a vertex measurement order
            must be unique.
        """
        if len(self.vertex_qubit) >= len(self.vertex_reg):
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

    @property
    def output_vertices(self) -> List[int]:
        """Output vertices.

        :return: Output vertices.
        :rtype: List[int]
        """
        return [
            vertex
            for vertex, measurement_order in enumerate(self.measurement_order_list)
            if measurement_order is None
        ]

    def get_output_result(self, result: BackendResult) -> BackendResult:
        """Create result object from measurement bits of output vertices.

        :param result: Result object from running this circuit.
        :raises Exception: Raised if not all output vertices have been
            measured.
        :return: Returns reduced result object.
        :rtype: BackendResult
        """

        unmeasured_outputs = [
            vertex
            for vertex in self.output_vertices
            if not self.vertex_measured[vertex]
        ]
        if len(unmeasured_outputs) > 0:
            raise Exception(
                f"Vertices {unmeasured_outputs} are output vertices but "
                "have not been measured."
            )

        cbits = [self.vertex_reg[vertex][0] for vertex in self.output_vertices]
        counts = result.get_counts(cbits + [self.is_test_bit])

        return BackendResult(
            counts=Counter(
                {
                    OutcomeArray.from_readouts([output[:-1]]): count
                    for output, count in counts.items()
                    if output[-1] == 0
                }
            ),
            c_bits=cbits,
        )

    def get_failure_rate(self, result: BackendResult) -> float:
        """Calculate the failure rate of the test vertices.

        :param result: The result of running this graph circuit.
        :return: Failure rate.
        """

        n_tests = 0
        n_fails = 0

        # Sum the number of test shots, and of those test shots
        # the number of times the test qubits are measured as 0.
        for reg in self.vertex_reg:
            vertex_counts = result.get_counts(cbits=[reg[0], reg[5], self.is_test_bit])
            for shot, count in vertex_counts.items():
                if shot[2] == 1:
                    n_tests += count
                    if shot[1] == 0:
                        n_fails += shot[1] * count

        return n_fails / n_tests
