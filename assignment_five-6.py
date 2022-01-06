""" This program trains and tests a neural network. """
from collections import deque
import random
from enum import Enum
import numpy as np
import math
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class DataMismatchError(Exception):
    pass


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [],
                           MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        """ Print the ID of the node and the IDs of the neighboring
        nodes.
        """
        node_id = id(self)

        upstream_neighbor_nodes = ""
        for neighbor_node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            upstream_neighbor_nodes += str(id(neighbor_node)) + " "

        downstream_neighbor_nodes = ""
        for neighbor_node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            downstream_neighbor_nodes += str(id(neighbor_node)) + " "

        node_id_string = f"ID of node: {node_id}"
        upstream = "\nID of upstream neighboring nodes: " \
                   + upstream_neighbor_nodes
        downstream = "\nID of downstream neighboring nodes: " \
                     + downstream_neighbor_nodes

        print(node_id_string + upstream + downstream)

    @abstractmethod
    def _process_new_neighbor(self, node, side: Side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """ Reset the nodes that link into this node.

        Key Arguments:
        nodes (list): a list of neighboring nodes
        side (Enum): a Side Enum specifying whether the neighboring
        nodes are upstream or downstream
        """
        self._neighbors[side] = nodes.copy()

        for node in nodes:
            self._process_new_neighbor(node, side)

        decimal_form = int((2 ** len(nodes)) - 1)
        self._reference_value[side] = decimal_form


class Neurode(MultiLinkNode):

    def __init__(self, node_type, learning_rate=.05):
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}
        super().__init__()

    def _process_new_neighbor(self, node, side):
        """ Add node reference to self._weights if the node is an
        upstream neighbor.

        Key Arguments:
        node: a neighboring node object
        side (Enum): a Side Enum specifying whether the neighboring node
        is upstream or downstream
        """
        if side == MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side):
        """ Update the instance attribute reporting nodes once node has
        information available. Compare the instance attributes
        reference_value and reporting_nodes to determine whether all
        neighboring nodes have reported.

        Key Arguments:
        node: a neighboring node object
        side (Enum): a Side Enum that specifies the node's side
        """
        node_index = self._neighbors[side].index(node)
        bit_position = 1 << node_index
        reporting_nodes = self._reporting_nodes[side] | bit_position
        self._reporting_nodes[side] = reporting_nodes

        if self._reference_value[side] == self._reporting_nodes[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    def get_weight(self, node):
        return self._weights[node]


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value: float):
        """ Return result of the sigmoid function at value.

        Key Argument:
        value (float): the value that we wish to apply the sigmoid
        function to (to limit it to numbers between 1 and 0)
        """
        sigmoid_function_result = 1 / (1 + math.exp(-value))
        return sigmoid_function_result

    def _calculate_value(self):
        """ Calculate weighted sum of the upstream nodes' values and
        pass the weighted sum through the sigmoid function, storing
        the value into self._value.
        """
        weighted_sum = 0
        for neurode in self._weights:
            weighted_sum += neurode.value * self._weights[neurode]

        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        """ Call data_ready_upstream on each of the node's downstream
        neighbors.
        """
        for down_stream_neighbor \
                in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            down_stream_neighbor.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """ Register that the node has data. Collect data when all
        upstream nodes have data and make available to the next layer.

        Key Argument:
        node: an upstream neurode
        """
        nodes_have_data = self._check_in(node, MultiLinkNode.Side.UPSTREAM)
        if nodes_have_data:
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """ Set the value of an input layer neurode directly through
        client.

        Key Argument:
        input_value: the value the client wants to set the value of an
        input layer neurode to
        """
        self._value = input_value

        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)


class BPNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value):
        """ Calculate the sigmoid derivative using the simplified
        formula: f(x) * (1 - f(x)).

        Key Arguments:
        value: f(x), the calculated value of the sigmoid function at x
        """
        sigmoid_derivative_result = value * (1 - value)
        return sigmoid_derivative_result

    def _calculate_delta(self, expected_value=None):
        """ Calculate the delta of this neurode.

        Key Arguments:
        expected_value: the expected value for the neurode, which is
        only applicable for Output layer nodes
        """
        if self._node_type == LayerType.OUTPUT:
            delta = (expected_value - self.value) * \
                    self._sigmoid_derivative(self.value)
            self._delta = delta
        elif self.node_type == LayerType.HIDDEN or \
                self.node_type == LayerType.INPUT:
            weighted_sum = 0
            for neurode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                weighted_sum += neurode.get_weight(self) * neurode.delta
            self._delta = weighted_sum * self._sigmoid_derivative(self.value)

    def data_ready_downstream(self, node):
        """ Register that the node has data and check whether all
        downstream nodes have data.

        Key Arguments:
        node: the Downstream node that has data ready
        """
        data_is_ready = self._check_in(node, MultiLinkNode.Side.DOWNSTREAM)
        if data_is_ready:
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """ Set the value of an Output Layer neurode according to the
        client.

        Key Arguments:
        expected_value: the expected value of the Output Layer Neurode
        """
        if self._node_type == LayerType.OUTPUT:
            self._calculate_delta(expected_value)
            for neurode in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
                neurode.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        """ Adjust the weight of the node.

        Key Arguments:
        node: upstream neurode
        adjustment: the value to change the weight
        """
        self._weights[node] = self.get_weight(node) + adjustment

    def _update_weights(self):
        """ Adjust the weights of the Downstream neighbors. """
        for neurode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = (self.value * neurode.delta * neurode.learning_rate)
            neurode.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """ Call data_ready_downstream on each of the node's Upstream
        neighbors.
        """
        for neurode in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            neurode.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta


class FFBPNeurode(FFNeurode, BPNeurode):
    def __init__(self, my_type):
        super().__init__(my_type)


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    class EmptyListError(Exception):
        pass

    def __init__(self):
        self._head = None
        self._tail = None
        self._curr = None

    def add_to_head(self, data):
        """ Add head to Doubly Linked List. If list is empty, set the
        tail to the head. If list is not empty, set the head to the
        new node and set up the previous and next pointers accordingly.

        Key Arguments:
        data: the data of the node
        """
        new_node = Node(data)
        if self._head is None:
            self._head = new_node
            self._head.next = None
            self._head.prev = None
            self._tail = self._head
            self.reset_to_head()
        else:
            self._head.prev = new_node
            new_node.next = self._head
            self._head = new_node
            self._head.prev = None
            self.reset_to_head()

    def remove_from_head(self):
        """ Remove a node from the head and return its data. However, if
        the Doubly Linked List is empty, raise an EmptyListError.
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        if self._head.next is None:
            self._head = None
        else:
            self._head = self._head.next
        self._head.prev = None
        self.reset_to_head()
        return ret_val

    def reset_to_head(self):
        """ Set the current node to the head. """
        self._curr = self._head
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def reset_to_tail(self):
        """ Set the current node to the tail. """
        self._curr = self._tail
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def move_forward(self):
        """ Move the current pointer forward. """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        if self._curr.next is None:
            self._tail = self._curr
            raise IndexError
        else:
            self._curr = self._curr.next
            return self._curr.data

    def move_back(self):
        """ Move the current pointer backwards. """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        elif self._curr.prev is None:
            raise IndexError
        else:
            self._curr = self._curr.prev
            return self._curr.data

    def add_after_curr(self, data):
        """ Add a node after the current node. If the list is empty,
        use the add_to_head method.

        Key arguments:
        data: the data of the node
        """
        if self._curr is None:
            self.add_to_head(data)
            return

        new_node = Node(data)
        if self._curr.next is None:
            new_node.prev = self._curr
            self._curr.next = new_node
            new_node.next = None
            self._tail = new_node
        else:
            self._curr.next.prev = new_node
            new_node.next = self._curr.next
            self._curr.next = new_node
            new_node.prev = self._curr

    def remove_after_cur(self):
        """ Remove the node after the current node and return its value.
        If the list is empty, return None.
        """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        elif self._curr.next is None:
            raise IndexError
        ret_val = self._curr.next.data
        self._curr.next = self._curr.next.next
        self._curr.next.prev = self._curr
        return ret_val

    def get_current_data(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        return self._curr.data

    def __iter__(self):
        self._curr = self._head
        return self

    def __next__(self):
        if self._curr is None:
            raise StopIteration
        ret_val = self._curr.data
        self.move_forward()
        return ret_val


class LayerList(DoublyLinkedList):

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        input_layer = [FFBPNeurode(LayerType.INPUT) for _ in range(inputs)]
        output_layer = [FFBPNeurode(LayerType.OUTPUT) for _ in range(outputs)]
        self.add_to_head(input_layer)
        self.add_after_curr(output_layer)
        self._link_with_next()

    def _link_with_next(self):
        """ Create bidirectional neurode links. """
        for node in self._curr.data:
            node.reset_neighbors(self._curr.next.data,
                                 FFBPNeurode.Side.DOWNSTREAM)
        for node in self._curr.next.data:
            node.reset_neighbors(self._curr.data,
                                 FFBPNeurode.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        """ Create a hidden layer of neurodes after the current layer.

        Key Arguments:
        num_nodes (int): the number of neurodes in the Hidden Layer
        """
        if self._head.data == self._tail.data:
            raise IndexError

        hidden_layer = [FFBPNeurode(LayerType.HIDDEN) for _ in
                        range(num_nodes)]

        self.add_after_curr(hidden_layer)
        self._link_with_next()
        self.move_forward()
        self._link_with_next()
        self.move_back()
        pass

    def remove_layer(self):
        """ Remove the layer after the current layer. Raise an
        IndexError if the user attempts to remove the output layer.
        """
        if self._head.data == self._tail.data or self._curr.next == self._tail:
            raise IndexError
        self.remove_after_cur()
        self._link_with_next()

    @property
    def input_nodes(self):
        return self._head.data

    @property
    def output_nodes(self):
        return self._tail.data


class NNData:
    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        """ Limit the training factor value.

        Key Arguments:
            percentage (float): the training factor (i.e., the
            percentage of data we want to use as our training set)
        """
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        else:
            return percentage

    def load_data(self, features=None, labels=None):
        """ Assign features and labels to self._features and
        self._labels accordingly, if the lists are the same length and
        have valid data values.

        Key Arguments:
            features (list): a list of lists, each row represents
            features of one example from the data
            labels (list): a list of lists, each row represents one
            label from our data
        """
        if features is None:
            self._features = None
            self._labels = None
            return None

        if len(features) != len(labels):
            self._features = None
            self._labels = None
            raise DataMismatchError

        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError

    def split_set(self, new_train_factor=None):
        """ Randomly sample from the indices of the dataset and assign
        the indices to self._train_indices. Then assign the remaining
        indices to self._test_indices. Sort the two lists.

        Key Arguments:
            new_train_factor (float): the proportion of data that will
            be used for the training set
        """
        if self._features is None:
            self._train_indices = []
            self._test_indices = []
            return

        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)

        total_examples = len(self._features)
        num_examples_train = int(total_examples * self._train_factor)

        indices_list = [i for i in range(total_examples)]

        self._train_indices = random.sample(indices_list, k=num_examples_train)
        self._train_indices.sort()

        set_test_indices = set(indices_list) - set(self._train_indices)
        self._test_indices = list(set_test_indices)
        self._test_indices.sort()

    def prime_data(self, target_set=None, order=None):
        """ Load one or both deques to be used as indirect indices.

        Key Arguments:
            target_set (Enum): an enum that helps identify whether a
            training or testing dataset is requested
            order (Enum): an enum that helps identify whether a
            randomized or sequential order of the dataset is requested
        """
        train_indices = self._train_indices.copy()
        test_indices = self._test_indices.copy()

        if target_set == NNData.Set.TRAIN:
            self._train_pool = deque(train_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(self._train_pool)
        elif target_set == NNData.Set.TEST:
            self._test_pool = deque(test_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(self._test_pool)
        elif target_set is None:
            self._train_pool = deque(train_indices)
            self._test_pool = deque(test_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(self._train_pool)
                random.shuffle(self._test_pool)

    def get_one_item(self, target_set=None):
        """ Return exactly one feature/label pair as a tuple.

        Key Arguments:
            target_set (Enum): an enum that helps identify whether a
            training or testing dataset is requested
        """
        if target_set == NNData.Set.TRAIN or target_set is None:
            if self._train_pool == deque():
                return None
            else:
                index = self._train_pool.popleft()
                return self._features[index], self._labels[index]
        elif target_set == NNData.Set.TEST:
            if self._test_pool == deque():
                return None
            else:
                index = self._test_pool.popleft()
                return self._features[index], self._labels[index]

    def number_of_samples(self, target_set=None):
        """ Return the total number of testing examples.

        Key Arguments:
            target_set (Enum): a enum that helps identify whether a
            training or testing dataset is requested
        """
        if target_set == NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set == NNData.Set.TRAIN:
            return len(self._train_indices)
        elif target_set is None:
            return len(self._test_indices) + len(self._train_indices)

    def pool_is_empty(self, target_set=None):
        """ Check whether the pools are empty or not.

        Key Arguments:
            target_set (Enum): a enum that represents a training or
            testing dataset
        """
        if target_set == NNData.Set.TRAIN and self._train_pool == deque():
            return True
        elif target_set == NNData.Set.TEST and self._test_pool == deque():
            return True
        elif target_set is None and self._train_pool == deque():
            return True
        else:
            return False

    def __init__(self, features=None, labels=None, train_factor=0.9):
        if features is None:
            features = []

        if labels is None:
            labels = []

        self._features = None
        self._labels = None

        self._train_factor = NNData.percentage_limiter(train_factor)

        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            pass

        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        self.split_set()


class FFBPNetwork:
    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        self._network = LayerList(num_inputs, num_outputs)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

    def add_hidden_layer(self, num_nodes: int, position=0):
        """ Add a hidden layer with the given number of nodes. If the
        position is greater than zero, advance that many layers, then
        insert the hidden layer.

        Key Arguments:
        num_nodes (int) = number of nodes in hidden layer
        position (int) = the position, which determines how many layers
        we will move before adding a hidden layer
        """
        self._network.reset_to_head()
        for i in range(position):
            self._network.move_forward()
        self._network.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2,
              order=NNData.Order.RANDOM):
        """ Train the network.

        Key Arguments:
        data_set (NNData): the data set containing the features and
        labels
        epochs (int): the epoch for the training
        verbosity (int): the verbosity for the training
        order (Enum): an enum that determines whether a randomized or
        sequential order of the dataset is requested
        """
        error_sum = 0
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        for epoch in range(epochs):
            error_sum = 0
            data_set.prime_data(order=order)
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                pair = data_set.get_one_item(NNData.Set.TRAIN)
                input_nodes = self._network.input_nodes
                output_nodes = self._network.output_nodes
                output_list = []
                for i in range(len(pair[0])):
                    input_nodes[i].set_input(pair[0][i])
                for i in range(len(pair[1])):
                    output_nodes[i].set_expected(pair[1][i])
                    output_list.append(output_nodes[i].value)
                    error = output_nodes[i].value - pair[1][i]
                    error_sum += math.pow(error, 2)
                if verbosity > 1 and (epoch % 1000 == 0):
                    print(f"Sample: {pair[0]} expected: "
                          f"{pair[1]}\nproduced: {output_list}")
            if epoch % 100 == 0 and verbosity > 0:
                quotient = error_sum / \
                           (data_set.number_of_samples(NNData.Set.TRAIN) *
                            self._num_outputs)
                rmse = math.sqrt(quotient)
                print(f"Epoch {epoch} RMSE = ", rmse)
        quotient = error_sum / (data_set.number_of_samples(NNData.Set.TRAIN) *
                                self._num_outputs)
        final_rmse = math.sqrt(quotient)
        print("Final Epoch RMSE = ", final_rmse)

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """ Test the Neural Network.

        Key Arguments:
        data_set (NNData): the data set containing the features and
        labels
        order (Enum): an enum that determines whether a randomized or
        sequential order of the dataset is requested
        """
        error_sum = 0
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        data_set.prime_data(order=order)
        input_list = []
        output_list = []
        while not data_set.pool_is_empty(NNData.Set.TEST):
            pair = data_set.get_one_item(NNData.Set.TEST)
            input_nodes = self._network.input_nodes
            output_nodes = self._network.output_nodes
            for i in range(len(pair[0])):
                input_list.append(pair[0][i])
                input_nodes[i].set_input(pair[0][i])
            for i in range(len(pair[1])):
                output_list.append(output_nodes[i].value)
                error = output_nodes[i].value - pair[1][i]
                error_sum += math.pow(error, 2)
            print(f"Sample: {pair[0]} expected: {pair[1]}\nproduced: "
                  f"{output_list}")
        quotient = error_sum / (data_set.number_of_samples(NNData.Set.TEST) *
                                self._num_outputs)
        rmse = math.sqrt(quotient)
        print(f"RMSE = ", rmse)
        return input_list, output_list


def run_iris():
    """ Initialize the network with one input, hidden, and output layer.
    Get the training dataset from the iris data set. Train then test
    the network.
    """
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2], [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
              [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
              [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2],
              [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
              [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2],
              [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4],
              [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
              [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1],
              [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2],
              [5.1, 3.4, 1.5, 0.2], [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3],
              [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4],
              [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3],
              [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6],
              [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4],
              [5, 2, 3.5, 1], [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1],
              [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5],
              [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3],
              [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3],
              [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7],
              [6, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1],
              [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2], [6, 2.7, 5.1, 1.6],
              [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3],
              [5.5, 2.6, 4.4, 1.2], [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2],
              [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1],
              [5.7, 2.8, 4.1, 1.3], [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9],
              [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3, 5.8, 2.2],
              [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2],
              [6.4, 2.7, 5.3, 1.9], [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2],
              [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5],
              [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2],
              [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6, 1.8],
              [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2],
              [6.4, 2.8, 5.6, 2.2], [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4],
              [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8],
              [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3],
              [6.7, 3.3, 5.7, 2.5], [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9],
              [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    """ Initialize the network with one input, hidden, and output layer.
    Get the training dataset from the sin data set. Train then test
    the network.
    """
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12], [0.13], [0.14], [0.15],
             [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23],
             [0.24], [0.25], [0.26], [0.27], [0.28], [0.29], [0.3], [0.31],
             [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38], [0.39],
             [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47],
             [0.48], [0.49], [0.5], [0.51], [0.52], [0.53], [0.54], [0.55],
             [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63],
             [0.64], [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71],
             [0.72], [0.73], [0.74], [0.75], [0.76], [0.77], [0.78], [0.79],
             [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87],
             [0.88], [0.89], [0.9], [0.91], [0.92], [0.93], [0.94], [0.95],
             [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16], [1.17], [1.18], [1.19],
             [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27],
             [1.28], [1.29], [1.3], [1.31], [1.32], [1.33], [1.34], [1.35],
             [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42], [1.43],
             [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51],
             [1.52], [1.53], [1.54], [1.55], [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331],
             [0.0299955002024957], [0.0399893341866342], [0.0499791692706783],
             [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175],
             [0.119712207288919], [0.129634142619695], [0.139543114644236],
             [0.149438132473599], [0.159318206614246], [0.169182349066996],
             [0.179029573425824], [0.188858894976501], [0.198669330795061],
             [0.2084598998461], [0.218229623080869], [0.227977523535188],
             [0.237702626427135], [0.247403959254523], [0.257080551892155],
             [0.266731436688831], [0.276355648564114], [0.285952225104836],
             [0.29552020666134], [0.305058636443443], [0.314566560616118],
             [0.324043028394868], [0.333487092140814], [0.342897807455451],
             [0.35227423327509], [0.361615431964962], [0.370920469412983],
             [0.380188415123161], [0.389418342308651], [0.398609327984423],
             [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068],
             [0.461779175541483], [0.470625888171158], [0.479425538604203],
             [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883],
             [0.539632048733969], [0.548023936791874], [0.556361022912784],
             [0.564642473395035], [0.572867460100481], [0.581035160537305],
             [0.58914475794227], [0.597195441362392], [0.60518640573604],
             [0.613116851973434], [0.62098598703656], [0.628793024018469],
             [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145],
             [0.681638760023334], [0.688921445110551], [0.696135238627357],
             [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859], [0.751280405140293], [0.757842562895277],
             [0.764328937025505], [0.770738878898969], [0.777071747526824],
             [0.783326909627483], [0.78950373968995], [0.795601620036366],
             [0.801619940883777], [0.807558100405114], [0.813415504789374],
             [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015],
             [0.852108021949363], [0.857298989188603], [0.862404227243338],
             [0.867423225594017], [0.872355482344986], [0.877200504274682],
             [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883], [0.912763940260521], [0.916803108771767],
             [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068],
             [0.942488801931697], [0.945783999449539], [0.948984619355586],
             [0.952090341590516], [0.955100855584692], [0.958015860289225],
             [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319],
             [0.975723357826659], [0.977864602435316], [0.979908061398614],
             [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588], [0.993868363411645],
             [0.994924349777581], [0.99588084453764], [0.996737752043143],
             [0.997494986604054], [0.998152472497548], [0.998710143975583],
             [0.999167945271476], [0.999525830605479], [0.999783764189357],
             [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    input_list, output_list = network.test(data)
    x = np.arange(0, (np.pi/2), 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.scatter(input_list, output_list)
    plt.show()


def load_XOR():
    """ Set up XOR into features and labels. Initialize NNData object
     using features and labels.
    """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    dataset = NNData(features, labels, 1)
    return dataset


def run_XOR():
    """ Initialize the network with one input, hidden, and output layer.
    Get the training dataset from the XOR data set. Train then test
    the network.
    """
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    data = load_XOR()
    network.train(data, 20001, order=NNData.Order.RANDOM)
    data.split_set(0)
    network.test(data)


def main():
    """ Train and test the neural network on the iris, sin, and XOR
    datasets.
    """
    run_iris()
    run_sin()
    run_XOR()


if __name__ == "__main__":
    main()

"""
Sample Run of run_iris():
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9989357490717369, 0.9999856646887164, 0.9993855375027781]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9999963533365441, 0.9999997962859115, 0.9999616618584978]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9999540179944627, 0.9999976085283098, 0.9998062630470235]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9999914344368965, 0.9999995862737447, 0.9999302101952259]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9982631341720825, 0.9999648334584874, 0.9989489413686894]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9999586118349241, 0.9999981557032452, 0.9998209782194596]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.9999931557263672, 0.9999994652274917, 0.9999170174703855]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999993526885376, 0.9999999363990779, 0.9999788723062868]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9999676943625878, 0.9999977932532821, 0.9998261057947936]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.998441878555883, 0.999977028668369, 0.9991674007585186]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.9999553831597073, 0.999996384026582, 0.9997683218209513]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.9999888454377382, 0.9999992476177649, 0.9998938085494782]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999993675247781, 0.9999999738484118, 0.999988925112676]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.998815428402769, 0.9999929214267238, 0.9995637671493709]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.9999979253449217, 0.9999998686242294, 0.9999642389656335]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9999901924983082, 0.9999992093699359, 0.9998866206987342]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9987065489069571, 0.9999889078644114, 0.999512003841954]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.998099535435712, 0.9999676942014101, 0.9989435932339348]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9999980763513511, 0.999999860491381, 0.9999698874566328]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9997892966574421, 0.9999888312081848, 0.9994122095875303]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.999988012898542, 0.999998716618358, 0.9998490487052413]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999990409472147, 0.9999999252121268, 0.9999787001417214]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9999814978408322, 0.9999992456929728, 0.9999016864278542]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9999942977044621, 0.9999996213906356, 0.9999401349692926]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9999729709502708, 0.9999982882630061, 0.9998433076497585]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9999755964215022, 0.9999984831577018, 0.9998681813430566]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9999342530514034, 0.9999955331097556, 0.999712658007302]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9998006844091241, 0.9999857446042864, 0.9993888031647976]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9999970000540535, 0.9999997637438268, 0.9999500064826342]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9999969755787481, 0.9999998297690095, 0.9999602082993884]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9990398034973746, 0.9999858889049125, 0.9993054860271462]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9988754091277877, 0.9999908760694188, 0.9994747602256085]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.999996309895357, 0.9999997817698937, 0.9999499859139434]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9985170176118682, 0.9999791428338642, 0.9992083104432128]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9983771644255944, 0.999979936090161, 0.9992298024489455]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.999958905185718, 0.9999977195010383, 0.9998123567396277]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.999218986888876, 0.9999923007770913, 0.9995678277214676]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999989067057551, 0.9999999508553756, 0.9999797313956978]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.999954721528366, 0.9999970079509242, 0.9997762345576319]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9973384630722294, 0.9999463844037865, 0.9985721228118862]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9998958141492609, 0.9999931824126883, 0.9996252016767565]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9999787506746314, 0.9999988862159905, 0.9998816751380694]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9999936916254811, 0.9999996887114756, 0.9999376069299827]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.9999258591492575, 0.9999952057443804, 0.9996995993293213]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9973768298069537, 0.9999328610754297, 0.9983514772346032]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9999704612335896, 0.9999986395590775, 0.9998604472790887]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9980593830522702, 0.9999638236388335, 0.998915875261145]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9999990416783351, 0.9999999341342402, 0.9999795898580394]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.998335030745904, 0.9999790508268424, 0.9991509900995951]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999973461550403, 0.9999998369048171, 0.999954718631904]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9999702210197642, 0.999997928847467, 0.999826501786553]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9971344992075537, 0.9999648296754727, 0.998842346119718]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9989126408734512, 0.9999927403572296, 0.9995937001853531]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9999825992437814, 0.9999980649193798, 0.9998311361480698]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9999641853165109, 0.9999956426674338, 0.9997034858197642]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9999902857526378, 0.9999989391290568, 0.9999035390173444]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9999942481945038, 0.9999995229633543, 0.9999317077640003]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9970828356261907, 0.9999004924732957, 0.9976161471721168]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9992874777182851, 0.9999959715538648, 0.9997024626518349]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.9999956991164813, 0.9999996343166515, 0.9999337137265801]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9985545090428211, 0.9999817785099848, 0.9992318561546876]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.999805409179089, 0.9999858472456165, 0.9993795969340151]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9999847087367755, 0.9999988373151278, 0.9998739282824824]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9988165486281478, 0.9999815763299026, 0.9992320834425205]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9999960644434325, 0.9999996645189584, 0.9999346301005154]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9984524395661571, 0.99998022582685, 0.9992078452126867]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.9971954601972857, 0.9999358583735276, 0.998371917438345]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9988155941537766, 0.9999880436283721, 0.9994384552329874]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9999266409141091, 0.9999968013923202, 0.9997391078624456]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9999374721110542, 0.9999949214971058, 0.9996673239994005]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.9999702014244954, 0.9999979844652681, 0.9998148418987493]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9999875191274684, 0.999999211160349, 0.9998898713002071]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9998924463436181, 0.9999948775533171, 0.9996714076007135]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.998429182522848, 0.9999732879434784, 0.9991184845153009]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9999771740928622, 0.9999987226434798, 0.9998630222225514]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999970049407712, 0.9999998167555748, 0.9999540894174102]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.999999108213533, 0.9999999710553724, 0.9999881525828421]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9999894373988688, 0.999999219528089, 0.99990151074795]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.999953997805069, 0.9999974064589222, 0.9997901975601158]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9999450358717752, 0.9999968062774811, 0.9997594898292135]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.99998148525453, 0.9999993755179838, 0.9999096271815198]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9999879910162747, 0.999999618856326, 0.9999363135079313]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9999976438890986, 0.9999998706168004, 0.9999626216480898]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9987781844346557, 0.9999866364210572, 0.9993961105209517]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9980587116914349, 0.9999638154719744, 0.9989133945588617]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9980586736047706, 0.9999638147728726, 0.998913212390975]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.9980914613554915, 0.9999619440412236, 0.998761912235]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9999344427682624, 0.9999954632227954, 0.9996894277582519]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.997703199471431, 0.999954464454213, 0.9987163252276237]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9999988152889101, 0.9999999384825442, 0.9999762759303139]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9984146300499471, 0.999980076543024, 0.9992167006802861]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9999890992645372, 0.9999989748023556, 0.9998766925973408]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9983031154758998, 0.9999650850763281, 0.998929745880439]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9999869918074258, 0.9999994921448175, 0.9999222490189801]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9988061065478683, 0.9999765006850686, 0.999238416079353]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9990375240595069, 0.9999839586647059, 0.9992701951320211]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9999977500016003, 0.9999998600577455, 0.9999639286681149]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9999824986145982, 0.9999993296391312, 0.9999070060633567]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.9999962515141094, 0.9999998108343311, 0.999953036325341]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9999979524114699, 0.9999998623633509, 0.999966928993539]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9982458824892193, 0.9999737648647585, 0.9990617289878462]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.999980092913873, 0.999999249921186, 0.9998986775118387]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9999900925605558, 0.9999996124329745, 0.9999357592265211]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9987775577852143, 0.9999860026671007, 0.9994141999078249]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9999412732849314, 0.9999966592872487, 0.9997774679696423]
Epoch 0 RMSE =  0.8163284687128448
Epoch 100 RMSE =  0.8160762128571543
Epoch 200 RMSE =  0.6899038323785491
Epoch 300 RMSE =  0.6765348091820762
Epoch 400 RMSE =  0.6690809084155026
Epoch 500 RMSE =  0.6682626973605326
Epoch 600 RMSE =  0.6675386601077195
Epoch 700 RMSE =  0.6668203240912958
Epoch 800 RMSE =  0.6664243113285169
Epoch 900 RMSE =  0.6669951768059216
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9932817935479308, 0.999923408898464, 0.002723583300426962]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999976157207853, 0.999999949415764, 0.8252330837779867]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.99986531921485, 0.9999930923940643, 0.07385141218334738]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999981980899718, 0.9999999399728701, 0.9926206172798242]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9965606239427682, 0.9999652282709742, 0.003916088233382186]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.999978118058451, 0.9999989546710757, 0.8563572375427153]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.999287069104391, 0.9999688034799399, 0.005302908613242413]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9997228131620559, 0.9999927810346121, 0.010993478609137772]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9999962573154474, 0.9999998279104941, 0.9858506451798056]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.999797006225816, 0.9999929131747599, 0.014635091724464578]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9999234187522773, 0.9999968792394, 0.16916657985683078]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9957998156219972, 0.9999742162120628, 0.005102221989600513]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9974182603297846, 0.9999911731868764, 0.009359624466419652]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9998885297660698, 0.9999969321789326, 0.021986093026170515]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9998430619207157, 0.9999958263652965, 0.0161488748887437]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9945859344190421, 0.9999577738998161, 0.003675723263125445]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.999755157654582, 0.9999900270280564, 0.01120145357653554]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9932805709049679, 0.9999234074593225, 0.0027270931249827927]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9960045334377163, 0.9999802879024392, 0.005423974747117993]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9997781957994175, 0.999992464833927, 0.013201505580740907]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.9933986464303429, 0.9999194508388225, 0.002396027718498469]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.999924830358953, 0.9999976060455492, 0.043645132286657166]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9999834423566757, 0.9999986722773218, 0.9536770150256316]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.999996752744716, 0.9999998265898055, 0.9866861773926161]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9998503078415989, 0.9999949919814044, 0.021375094038796998]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9997897910972869, 0.999990937235144, 0.03428500876899705]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.999996613933708, 0.9999998407269934, 0.987914282237453]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9999288896872011, 0.9999985730747503, 0.029975148640736846]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9999417340168142, 0.9999947347069346, 0.9013949325221927]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9999980212251149, 0.9999999241230053, 0.9904171249850203]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9956633116577064, 0.9999711796123973, 0.004756089310685293]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9999911238412328, 0.9999996679852922, 0.7520526945637872]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.990482857621801, 0.9998662723004627, 0.0018841436260929748]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9998319297565326, 0.999994735577153, 0.020137671115862534]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9999518328917543, 0.9999988706385479, 0.047105524703274164]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9999751157191807, 0.9999988908948416, 0.835349223398647]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.9999807101828543, 0.9999984606290038, 0.9542270099034378]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.999993342204716, 0.9999997558247448, 0.9692123501701094]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999989355535912, 0.9999999675659991, 0.9953069775668211]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9996144767473504, 0.9999848564104673, 0.008788956269736494]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9910620833895138, 0.999859663409913, 0.0018562694441764604]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9938981644271013, 0.9999442334480453, 0.003160202160757709]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999956928233746, 0.9999998030773461, 0.9854875903345032]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9958081278685252, 0.9999847428502343, 0.006584306267498327]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9999638318182094, 0.9999991471464799, 0.06369288941219964]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9999961297617345, 0.999999842377266, 0.9868255539253307]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9940806337861001, 0.999925712653634, 0.0027723563831967414]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9941881313376149, 0.999955342869204, 0.0034638648136327043]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9999028919184166, 0.9999957971970788, 0.10184400849280197]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9949204787568551, 0.9999609721480869, 0.003798159317595864]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9992163397718027, 0.9999750963708888, 0.0049013322377865105]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999989536124001, 0.9999999234421882, 0.993518450722935]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.9999035636921646, 0.9999959438345475, 0.055171416362211456]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9944865902743422, 0.9999429235528662, 0.0033338265574645153]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.99999844124042, 0.9999999203472291, 0.9933968125518708]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.99972937051069, 0.999989457126865, 0.012638961688181752]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9999875516460909, 0.9999995656668793, 0.9091441203217625]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9999822865495618, 0.9999987614431026, 0.9596724831667313]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9954072676995732, 0.9999760798816851, 0.005892897356533765]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9909208743998897, 0.9998878212331526, 0.0021349035819357787]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9999762838242294, 0.9999992582270958, 0.36731586384127224]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9999951292149006, 0.9999997146931806, 0.9837692056777819]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9999238779695572, 0.9999982918421971, 0.02768656637547333]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.9999935870555513, 0.99999972283747, 0.9720942388418756]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9965620413140723, 0.9999693735304894, 0.004082122564390515]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999951157031666, 0.9999997779846712, 0.9845912190901351]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9939350254298642, 0.9999251619536901, 0.0027873134730667937]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9902687563018536, 0.9999265215591191, 0.0026052123898200304]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9932791060397813, 0.9999233991109218, 0.002722823706357594]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9947914223079433, 0.9999553597585316, 0.003656050760357056]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9999831668446758, 0.9999990051529825, 0.9426906544708944]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9995931765056356, 0.9999884313069257, 0.008283336011350121]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.999920802487779, 0.9999963977046918, 0.23353058533014998]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.9999929846557708, 0.999999556895041, 0.9775366998617834]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9957999226030263, 0.999960277801443, 0.003723617347710134]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9997992560688104, 0.9999898444528956, 0.03153840970463747]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9998409721995581, 0.9999936053862978, 0.023696755629426063]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9998298745000643, 0.9999942404411483, 0.016787058005126143]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9944516304703244, 0.999957460409556, 0.003705119685533698]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9992634336315618, 0.9999683895404162, 0.005005148507078071]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9999902100855913, 0.9999994035601749, 0.9675709460403423]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9999936114153003, 0.9999995949604141, 0.9787282507307798]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9999692832316893, 0.9999975147232456, 0.8957394185764651]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9999891772423543, 0.9999994868584914, 0.9341777381032285]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9943154581866951, 0.9999571494273649, 0.003761638407736134]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9921261669013737, 0.9999042878284284, 0.0023281194306252724]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9945349880644384, 0.9999509034909653, 0.0034750317186528134]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9901913673561966, 0.9997936698894977, 0.0012846045715587367]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999984290254925, 0.9999999090918801, 0.9926336713421323]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9962079408797468, 0.9999689945003477, 0.004613052371343051]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9999964279269269, 0.9999998247006728, 0.9810567462083302]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9971775742817393, 0.9999831743348622, 0.00645792074882452]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9961200113588202, 0.9999842733265254, 0.006954648022057932]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9999806105953494, 0.9999989680309325, 0.9172216828849185]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.99993354461487, 0.9999984814391004, 0.03209085978564808]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9933899639654057, 0.9999314194509433, 0.00277983309883431]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9999623211751144, 0.9999980880025499, 0.4426421781390488]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.999909035301002, 0.9999966117811804, 0.025713147863787186]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9999535004653968, 0.9999991239036529, 0.03981754863857471]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9999176595091591, 0.9999972146686021, 0.03209647134161567]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9956563786540333, 0.9999698040033482, 0.004833993919547404]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9999942206764084, 0.9999997680761308, 0.9529948681330751]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9957505543108081, 0.9999493172608136, 0.0037337466899100545]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.9999874785861967, 0.9999992946038954, 0.9309076695827452]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.999930124909481, 0.9999982984506803, 0.03139230727223973]
Epoch 1000 RMSE =  0.6658062263289488
Epoch 1100 RMSE =  0.666730860466623
Epoch 1200 RMSE =  0.6667121939456514
Epoch 1300 RMSE =  0.6669505899105974
Epoch 1400 RMSE =  0.6667474993192498
Epoch 1500 RMSE =  0.6661643529591716
Epoch 1600 RMSE =  0.6670994206694475
Epoch 1700 RMSE =  0.6665917724061342
Epoch 1800 RMSE =  0.6666695271998058
Epoch 1900 RMSE =  0.666985809640871
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9917192729386006, 0.9999801099144713, 0.0038002226881134397]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9870135141366265, 0.9999294813669752, 0.0017027632219965493]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9999603294095916, 0.9999987813824331, 0.9587232435196327]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9998396634293151, 0.9999961767951484, 0.16751952142152185]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9999278886985967, 0.999997829836496, 0.6820218286708192]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9999183136358268, 0.99999887947881, 0.023547861934632708]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.999991217399149, 0.999999799169936, 0.9853397222442464]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9884641669587603, 0.9999465943372804, 0.0019630343650899346]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9995193227528453, 0.9999903937032476, 0.006543042478095178]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9907484146561057, 0.9999635487002826, 0.002539767267681807]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9907371910443902, 0.9999618161300644, 0.002621173608388378]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9998040713751039, 0.9999957054623979, 0.014127072699430919]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999890541265695, 0.9999997199589012, 0.9849690938274818]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9939686967634891, 0.9999787193300073, 0.003468819156045975]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9998293284457324, 0.9999968453823196, 0.015799152283681714]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9882601558643364, 0.9999278322190663, 0.0017684868410728024]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9998927131486108, 0.9999985312507201, 0.019227801175847145]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9998353511630558, 0.9999978255643098, 0.01406914047815106]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9983112414322936, 0.9999683649331216, 0.0024891903403939]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9870942107395922, 0.9999053858112087, 0.0014890078767441617]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.9999838315034387, 0.9999996708463821, 0.932097246601136]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.999985728082465, 0.9999996513805202, 0.972468643349152]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9999627393051498, 0.9999983166453272, 0.9505419489653526]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9999784501998987, 0.9999992567636613, 0.9717536041231105]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9998497414011362, 0.9999978454125522, 0.017374879473261898]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9998556514896619, 0.9999980614674988, 0.015939177233724504]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9997569789279108, 0.9999960731199594, 0.010519187237368274]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.9999923970919229, 0.9999997988469431, 0.9880627339820931]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9997552726558216, 0.9999940318185826, 0.017983448259891253]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9883593296961721, 0.9999379181675618, 0.0018639668497931307]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9995090742260173, 0.999987836106587, 0.010568573771339256]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9810516230980793, 0.9998226329015669, 0.0009821695198535358]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9996598870597546, 0.9999946804529717, 0.008052046862140855]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9999373271182561, 0.999998911056411, 0.06330719098834829]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999964828747295, 0.9999998854323786, 0.9929011165236068]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9999625451249636, 0.9999987494199768, 0.9440908860286804]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.999544440547681, 0.9999866712510157, 0.012646395766290671]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9991556023028473, 0.9999805437696774, 0.003970751474956527]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999942016634948, 0.9999999305692123, 0.5587677187454362]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999976318957531, 0.9999999592889542, 0.9956127233809555]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9999956397922871, 0.9999999054952551, 0.9919238437054968]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9909359499655009, 0.9999359197102748, 0.0020452394717819476]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999976403584588, 0.999999903041546, 0.9934626187389842]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9881789796170632, 0.9999462034921716, 0.002012426715805542]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9807530243223577, 0.9998582139964421, 0.0011428643429125182]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9999003160468316, 0.9999988923541426, 0.022424963199241085]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9999923941580797, 0.9999997870102463, 0.9876730667423332]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.985704399524852, 0.9999031462208084, 0.0014715084381420472]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9919037221916606, 0.9999607851022903, 0.002503537175591748]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.9859540304743222, 0.9998981407387604, 0.001292825725888972]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.9999747409329308, 0.9999991765820865, 0.9702923788422407]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.9997051170540323, 0.9999912124135958, 0.04567115567617219]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9999352334665822, 0.9999970003562675, 0.934155148430994]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9996735631479273, 0.9999935972295734, 0.01072635571864854]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9999603602055888, 0.9999984396958462, 0.9613272488144977]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9991261579466768, 0.9999853474703122, 0.004443377194045455]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.999404495101899, 0.9999864183343316, 0.005695267583081552]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.9798301215831166, 0.9998309696876767, 0.0010104532988949276]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9999458995272383, 0.9999986221675182, 0.8443773639078235]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9999887727053238, 0.9999997367300639, 0.9852331023697237]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9832774445067514, 0.99987899252416, 0.0012639593386572512]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9984537198374289, 0.9999602174044543, 0.002655049760926371]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9888950967954373, 0.9999435426791803, 0.0019826049344879105]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.999956631195887, 0.9999980548515544, 0.9550033207461122]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9996306360470859, 0.9999932281888375, 0.009348261499531505]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9926479401637328, 0.9999560130166414, 0.0021199942369142456]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9995577255052813, 0.9999909405419214, 0.007239903710763445]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9792210781691989, 0.9997392168711615, 0.0006968821989553706]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999959958657296, 0.9999999248047294, 0.9934535466897563]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9878894486536082, 0.9999458067923216, 0.0020452162865643034]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9857004492410951, 0.9999031370247305, 0.0014754253506504326]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9996650817872124, 0.9999920568875049, 0.017200221469998573]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9926539369098575, 0.9999612589150757, 0.0022151357212343097]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9998468010052696, 0.9999981896640476, 0.016058103324832363]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9999917278553824, 0.9999997852121707, 0.9878400264104422]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9999928922691365, 0.9999997850672964, 0.9893781833955607]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.979371114514734, 0.9999070957716394, 0.0014112375781774635]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9910474257396406, 0.9999806962529202, 0.003545910498526731]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9984148827706619, 0.9999599130801874, 0.00267910902597486]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9914636307969252, 0.9999750601295698, 0.002941181712809722]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9901987037183612, 0.9999697421290359, 0.0031729254714342758]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9910337731671099, 0.9999497575671528, 0.002027520500995089]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9999731715188929, 0.9999994644657281, 0.9221041369161495]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9910289716353662, 0.9999673791819162, 0.0027670893596597975]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.999984359126, 0.9999994435411329, 0.9796481087755371]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9873919255986052, 0.999906054652475, 0.0014899556109011713]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9996355969945084, 0.9999927343006995, 0.00979984777263108]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.999870781236545, 0.9999934122703805, 0.9125378393637477]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9994691606280639, 0.9999872932384221, 0.005731812854568192]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9999964986698391, 0.9999998994209819, 0.9935870357262285]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9856977490930077, 0.9999031323070184, 0.00147727260832795]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9999890815135486, 0.9999996402924521, 0.9846368593765025]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9999822138637697, 0.999999609912427, 0.846861588300473]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9999783695375439, 0.9999994045811925, 0.9729416188154991]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9998476530893632, 0.9999958828667408, 0.41799861338236716]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9891647046951814, 0.9999506312905619, 0.0020304130348000895]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999903081511416, 0.999999751003948, 0.9855081157369685]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9944737298364247, 0.9999888305755517, 0.005051159376082358]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9859360718107365, 0.9999132849260645, 0.001498834712698604]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9999856850221666, 0.999999489423631, 0.9797671468844986]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9876143496237697, 0.9999435124618069, 0.0018495757801430862]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9994009574321264, 0.9999908192582123, 0.005630803469560696]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9998305352008282, 0.9999965995778913, 0.028189192651616315]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.999771853615028, 0.9999945020013167, 0.015425132336683878]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.9999522754400932, 0.9999986995631829, 0.8641852822639006]
Epoch 2000 RMSE =  0.6665461410986666
Epoch 2100 RMSE =  0.6661780601877243
Epoch 2200 RMSE =  0.6663025093070863
Epoch 2300 RMSE =  0.6662638546791491
Epoch 2400 RMSE =  0.6665044011895739
Epoch 2500 RMSE =  0.6667039180992806
Epoch 2600 RMSE =  0.6664976484452667
Epoch 2700 RMSE =  0.6664273731317895
Epoch 2800 RMSE =  0.6669845818805828
Epoch 2900 RMSE =  0.6658715870387405
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9993850229023722, 0.9999920447337547, 0.018891650454927387]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9995444645017695, 0.9999938930128478, 0.18033680063641588]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9957225283422021, 0.9999569458464034, 0.0019954285344075846]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9999676348115935, 0.9999996391748395, 0.9826875369637984]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9651024423515427, 0.9998821016270376, 0.0011958590885424827]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9999751515603785, 0.9999997293634757, 0.9861821826106986]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9586647177581997, 0.9998354622799079, 0.0010017210715837443]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9998849827715292, 0.999998317275524, 0.9471711956607389]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.9998660639317825, 0.999998225211626, 0.8479069313219552]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.970989280815489, 0.99991555645551, 0.0015010788771362357]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9991532404773084, 0.9999911537181634, 0.00718450223279566]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9489917686721158, 0.9996455119563523, 0.0005531128646647316]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9995817516100793, 0.9999970398093296, 0.011331990297527702]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9996330941794218, 0.999997360803962, 0.012833641309754635]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9775982061868586, 0.9999737390414855, 0.0028204391749176746]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.9999275577975268, 0.9999988767194071, 0.9673584114864995]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9786323562913211, 0.9999660734187978, 0.0023390828992823107]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9691900620693531, 0.999923176104639, 0.0014799055842247696]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9526436622514549, 0.9998072318828123, 0.0009095735459670807]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9998425419169165, 0.999998557064548, 0.07157931100830106]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9792532149179223, 0.9999729360940789, 0.003012419593214065]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.964548283997586, 0.999868274190776, 0.0011714783648964883]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9997465759196732, 0.9999984900810747, 0.01760276458312642]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9775871707740035, 0.9999316609763507, 0.0016107687215770263]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9998968398306968, 0.999997763709238, 0.960240590669323]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9705703952693567, 0.9999268173844341, 0.0016030802567694451]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.999993225390048, 0.9999998678521703, 0.9929129077558286]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9998414465014511, 0.9999980384289897, 0.7401084601534251]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9987773480854949, 0.9999839098059233, 0.012240951136281056]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9999899594802214, 0.9999998630029784, 0.9930898647522891]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9995071384797249, 0.999994225786839, 0.013339792597599919]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.999958365922461, 0.9999995846222087, 0.9693148483966418]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9999487959498209, 0.9999994454779464, 0.7431477245273949]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9997943467171925, 0.9999984883037916, 0.02140513478262335]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9998171501362484, 0.9999959569404392, 0.9376681060640516]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9999686846031993, 0.9999995101131611, 0.9835962420194305]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9723254961070151, 0.9999232030417883, 0.0015917366859742096]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.9504662498726643, 0.9997701986239015, 0.0008118277655910699]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9677190219991337, 0.9999040655644762, 0.001362557962008351]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999687375256479, 0.9999996198915427, 0.9849018906322927]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.9999610668193218, 0.9999995427427141, 0.9813125782386973]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999880457749066, 0.999999924504427, 0.9421896061715481]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.984844067659692, 0.9999710423815183, 0.0028268428485149627]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9698799704025663, 0.9999262844949182, 0.0016422181081461526]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9989216846670604, 0.9999832298725544, 0.026785164400460644]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.99957881001335, 0.9999958313923131, 0.018448305042669318]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9998081363258005, 0.9999971405095399, 0.7399274423209765]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9960565825484025, 0.9999455427532418, 0.0019590793379722373]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9775671396054724, 0.9999556222957448, 0.002190275294289475]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9984796010650394, 0.999987492435154, 0.004427822063561829]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9645423534476966, 0.9998682632455362, 0.001167350818592143]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9998862822591527, 0.9999978735063664, 0.9579229249169516]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9993824263441649, 0.9999946505111527, 0.008376299299334419]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9986403974857925, 0.9999825740663832, 0.0041432368869374625]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9815666831140112, 0.9999401569373161, 0.0016788354516288483]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.999955096140266, 0.9999992415468507, 0.9777088584709425]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.976829874994833, 0.999948047040221, 0.0021050716299415143]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9996172382032144, 0.9999970661334912, 0.013977402085949147]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9707551973502548, 0.9999018273322612, 0.0014194466756747735]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.986105362587657, 0.9999848028526961, 0.004031494904790899]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.9999782712413816, 0.999999726821426, 0.9876957377734056]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.995958248766184, 0.9999451442088371, 0.001983393873658747]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9729769556809975, 0.9999328436330385, 0.0016187439811734072]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.9998754759063978, 0.999997349007698, 0.9509790666153956]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999899410207383, 0.9999998443702862, 0.992665396229545]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9991360568563792, 0.9999927549855373, 0.006424867656128796]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9533434913093336, 0.9997588292983751, 0.0007872089439342119]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.999046754333033, 0.9999906666916039, 0.006405784278229215]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9645333976142807, 0.9998682544283933, 0.0011678488986840117]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.998779309071226, 0.9999869176858139, 0.005230390146638228]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9998954979863103, 0.9999983282548882, 0.9515630200890194]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9815780890695355, 0.9999472925135721, 0.0017551408186317259]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9493319900259442, 0.999873653152507, 0.001117079372381232]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9999190683550587, 0.9999992196594499, 0.8241954147568773]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.968660427036821, 0.9998722244464272, 0.0011819248486791961]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.967922746992252, 0.9998712947089855, 0.00119848415971379]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9755269280812217, 0.9999588336616475, 0.0025207269497435104]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9797227975367625, 0.9999466388476784, 0.0019941416492600824]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.9984735403752446, 0.9999813656816415, 0.004134779538613623]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9991006483461756, 0.9999887676026018, 0.008860405791271531]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9999762575806334, 0.9999997069213273, 0.9865613457869846]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9999781867872527, 0.9999997092290198, 0.9863376372018803]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999722408395545, 0.9999996611563189, 0.984629532126004]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9999394502215063, 0.9999990028141666, 0.9747680180767081]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9996104550230026, 0.9999975314012746, 0.012518564530672396]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9999589438755735, 0.999999304626744, 0.9783518907074438]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.999727407158706, 0.9999980011454004, 0.015618611515092966]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9999874856990233, 0.9999998710773561, 0.991144804469512]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.9994050542508024, 0.999992400941519, 0.010440776555519986]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9768559063188732, 0.9999503973732922, 0.0020461173843565926]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9978576562975292, 0.9999735048444239, 0.0031807820458653034]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.999937269745465, 0.9999991761969002, 0.9645049172458663]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.965138222357371, 0.9998614364208203, 0.001029140794520163]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.9991841427906497, 0.9999874262868542, 0.021304142890050483]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9977839635160917, 0.9999800477325949, 0.003520382775003927]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9995559880736101, 0.9999953035640359, 0.019460130036194217]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9773276134208, 0.999912819413148, 0.0016281749987646953]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9988630182040482, 0.9999875498597456, 0.005148695714181446]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9999795883467615, 0.9999997065723165, 0.9881846138406066]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999885011595808, 0.9999998974814367, 0.9928470842498813]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9990529234565715, 0.9999899416228804, 0.006402214307853537]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9712262158421513, 0.9999273260006465, 0.00158128925486625]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9995313744761833, 0.9999943386064839, 0.057974390158707534]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9996294841301762, 0.9999910267361386, 0.9057761687187266]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999932048784255, 0.9999999443921053, 0.9951064533863224]
Epoch 3000 RMSE =  0.6661048744097736
Epoch 3100 RMSE =  0.6663642004309618
Epoch 3200 RMSE =  0.6663950000655603
Epoch 3300 RMSE =  0.6663950971752374
Epoch 3400 RMSE =  0.6667981486967303
Epoch 3500 RMSE =  0.6660347970997913
Epoch 3600 RMSE =  0.6660815336175857
Epoch 3700 RMSE =  0.6653575037484586
Epoch 3800 RMSE =  0.6660708681429491
Epoch 3900 RMSE =  0.665373539680466
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9995355205156856, 0.9999961397898011, 0.010704465776382018]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9733753338059064, 0.9999306307386124, 0.0018809053352825643]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9998484028820145, 0.999996530949298, 0.9603439012353268]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9994146430179682, 0.9999907443229614, 0.2207539448950282]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999829070332871, 0.9999998395574325, 0.9920251355081087]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9725405026792557, 0.9999224789038317, 0.0017448202080414747]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9949059238174158, 0.9999326675136138, 0.0017005812287573452]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9781178688239828, 0.9999176195386978, 0.001500639215802906]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9995613354636727, 0.9999958663485463, 0.010757262771282677]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.999938663046646, 0.9999993426212848, 0.9607480456007651]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.9587220049913405, 0.9997835077750803, 0.0008772921796746438]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999899949765332, 0.999999913370853, 0.9948264008776672]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999898926433805, 0.9999997929023203, 0.9919916349298906]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9998889840209296, 0.9999988065774951, 0.8458254302676214]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.99998143929083, 0.9999997985822197, 0.9903642923560898]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9834778992695246, 0.9999762409159246, 0.0034577056586254]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.998468398099771, 0.9999743163096194, 0.00820209879364703]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9724978217934935, 0.9999187951227355, 0.0018044911193685934]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999850244992187, 0.9999997563864523, 0.9918773483876003]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9580143789372025, 0.9997941397127108, 0.0010005397180393961]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9994496493220478, 0.9999859673793065, 0.8970867505882884]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9952922325355207, 0.9999148083789313, 0.0016683809305588781]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.9999418890630563, 0.9999992798230888, 0.9773902035313348]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.957999984647961, 0.9997941351818608, 0.0010015384184659205]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999533448189705, 0.9999994042174541, 0.9827665985593911]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9994005947668976, 0.9999908879491493, 0.010230506152465554]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9653226066394297, 0.9998465780201016, 0.0012176884490762672]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9999296992035525, 0.99999911958746, 0.6798633844800538]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9679549210898799, 0.9998950280933869, 0.0013986402398658403]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9650987431669421, 0.9998855999456235, 0.0013802904516729175]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9973594737275331, 0.9999688059172385, 0.00303307509542614]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9746284978039463, 0.99994695172828, 0.0020138528499890652]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9997319701152686, 0.9999955528022325, 0.7306059601953616]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9671705918363597, 0.9998799495594789, 0.001337143742970793]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9819717677931816, 0.9999547243538228, 0.0023747114247265897]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9989692969703191, 0.9999886640376848, 0.005437224449982685]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9659011714450646, 0.9998864129653315, 0.0013425636130292816]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9988623291151507, 0.9999842006644983, 0.005151306493079671]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.9414904824506537, 0.9996410074317894, 0.0006815732064103487]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9402043348062638, 0.9998025761786783, 0.0009506858596820234]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.9998145932346968, 0.9999958473273532, 0.9451381881264193]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.9998928247299009, 0.9999982439222904, 0.9641032745188779]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9999533084839317, 0.9999992319778493, 0.9810809180422688]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.9999676529344494, 0.9999995722121195, 0.9862328470327737]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9999697867987443, 0.9999995413206643, 0.9869694532013792]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9986401605888546, 0.9999804830951341, 0.004245842206566056]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9999850336820181, 0.9999997852716108, 0.9920959867285832]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9635021611873379, 0.9998798921837626, 0.0012561978206232873]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9981871567452499, 0.9999804321975214, 0.0037582009928512276]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9995010062431242, 0.9999953664092579, 0.009551974107749987]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.9981727855036141, 0.9999707903674101, 0.0034122119369262324]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9642798716734379, 0.9998847639775101, 0.0013796619380842174]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9734166162769351, 0.9999589306877373, 0.002395362889496713]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9781079969144223, 0.9999064388369457, 0.0014306803642140972]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9753667433329157, 0.9999576777865211, 0.002559293143057014]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9580292625553826, 0.9997940964123452, 0.0009946348474066775]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9985443933923975, 0.9999795306285211, 0.004430422604204223]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9999647563811987, 0.99999954131128, 0.984991307996192]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999815564833606, 0.9999998627695854, 0.6958996233835187]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9617491314665753, 0.9998500276554969, 0.0011524523326414775]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9994296492442292, 0.9999915913112737, 0.09096216389248864]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9998311445699632, 0.9999966772463444, 0.9546036337329321]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.9999331538942242, 0.9999988124853425, 0.9753413775149553]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9998469987146616, 0.9999974133102744, 0.9532439921400635]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9974450228882296, 0.9999585625344212, 0.0027139565654217476]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9994810936569954, 0.9999933597722597, 0.012622258805925103]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9999632204441291, 0.9999995779825792, 0.9854295273128619]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9619942859164905, 0.9997988613587403, 0.0010241369511126198]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9997880683145569, 0.9999971083732828, 0.8394989418690341]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.9992863723362493, 0.9999882554005872, 0.010365911005175255]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9998116718551495, 0.9999978629656509, 0.11495731277205667]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9985976294441031, 0.9999727574071863, 0.014690079664838242]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9998340057931113, 0.9999974237197603, 0.9549373345460459]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.999817228798762, 0.9999973726925376, 0.9076925298227577]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9759118019886633, 0.999916573626048, 0.0017081243920223834]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.999258790017783, 0.9999878736938146, 0.022264781036372205]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9398062486360489, 0.9994461015516618, 0.0004730365048159237]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9997521499239538, 0.9999976193564006, 0.016824982930833447]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.998988391291368, 0.9999862021227813, 0.00638707916412055]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9999679186092758, 0.999999548894393, 0.9864111081577213]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9983782571896851, 0.9999727624901301, 0.0035908317996602135]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.999697693132792, 0.9999976381183889, 0.015116716252736886]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9999103820308725, 0.999998446041868, 0.9735561251511973]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9997289088733576, 0.9999936715480833, 0.930645668195318]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9709517394789943, 0.9999356376797156, 0.0021601773752624744]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9996744244313507, 0.9999968764303367, 0.013555869828866942]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9992632166936253, 0.999991638781448, 0.007273999504013493]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9628586717430733, 0.9998002827246728, 0.0010123897261606841]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9999388003037295, 0.9999989105341546, 0.975860957389865]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9656199026696657, 0.9998679527243972, 0.0012851171120492938]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9449046924923468, 0.9996231885138918, 0.0006759342342563741]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9731118483809408, 0.9998637192165445, 0.0013946315706208484]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9440876355543224, 0.9996987288995713, 0.0007782783217296716]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9988635293584521, 0.9999854328907207, 0.005661065686876723]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9999097998450246, 0.9999987403569051, 0.9701997665174673]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9734041160579189, 0.9998931268291223, 0.0013798584277467246]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9511711585379677, 0.9997427471036743, 0.0008593457868594889]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.999466091334897, 0.9999927960041363, 0.021251408166165]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9587368410363161, 0.9998156502471481, 0.001025919321832709]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.9990157350939612, 0.9999810745052194, 0.029232196904557292]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999586983969447, 0.9999994694331691, 0.9830005561839461]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9989201262223927, 0.9999826127233402, 0.008657942453865336]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9995436769824948, 0.9999954418914432, 0.013101439274206416]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.995183853044129, 0.9999142251182712, 0.0017099176072210424]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.999952620610177, 0.9999994418141196, 0.9835011557238001]
Epoch 4000 RMSE =  0.666280577092989
Epoch 4100 RMSE =  0.6662106624450624
Epoch 4200 RMSE =  0.6666065546188561
Epoch 4300 RMSE =  0.6665114193719944
Epoch 4400 RMSE =  0.6666188056641019
Epoch 4500 RMSE =  0.6662505613165127
Epoch 4600 RMSE =  0.6663278540536898
Epoch 4700 RMSE =  0.6660523627111334
Epoch 4800 RMSE =  0.6654083365157142
Epoch 4900 RMSE =  0.6655616801170678
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9772941760600966, 0.9997859736447368, 0.0014045413005243244]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9995437661841097, 0.9999905135225741, 0.010227948443197556]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9998759750850019, 0.999996960742907, 0.9472932685809357]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9984171314495743, 0.9999321206856879, 0.005173745453783319]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9999746213391731, 0.9999995328719861, 0.9894038003820166]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9951053124154793, 0.9998047027937549, 0.0015925381454409424]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9988351560831645, 0.9999585991325105, 0.005220694365305844]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9750067451873465, 0.9998091545417096, 0.0016639046890621096]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9972575004497509, 0.9999285870394425, 0.002931586773712944]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9629433073645189, 0.9997364372295032, 0.0013546811374574794]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9420236670087042, 0.9993118464899149, 0.0007573829418914828]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999795213380039, 0.9999994393970303, 0.9918921835579172]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9983109372566817, 0.9999374521891488, 0.00335898542147871]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9715042512331309, 0.9998226021860787, 0.001706953557870334]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9981160195280447, 0.9999551913565181, 0.0036740409143292585]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9996858548750216, 0.9999945902533405, 0.014620155035254465]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9989367898262131, 0.999968009580127, 0.005327326419697808]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9997792691642715, 0.9999944253069382, 0.025359759835570303]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9947063331714767, 0.9998458473414912, 0.0016599230478013735]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9996597844465127, 0.9999928050842194, 0.01224284809946975]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9723679824834229, 0.9997555761569368, 0.0013431554158960864]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9994102576171056, 0.9999824080877465, 0.00954644232273983]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.9999205435702399, 0.9999983128658156, 0.9719496780410372]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9621076404695635, 0.9997252479199646, 0.00123342342975678]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9744181876203668, 0.9999031455704179, 0.002513401911986698]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9571529628731666, 0.9995786098506559, 0.0009985201914511322]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9376098177017821, 0.9987348491711874, 0.0004605414406808469]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9995172260771827, 0.9999911546012784, 0.010379026515932909]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.972090049313803, 0.9996883208489514, 0.0013575172030182486]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9988108352177024, 0.9999664027151309, 0.005034001593257299]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9999518535358078, 0.9999989395221462, 0.9842709643130007]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.9393714975177446, 0.9991795904459816, 0.0006692377961052119]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9698740138150477, 0.9998527133774409, 0.002103264259054922]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.999746427334557, 0.9999904486688955, 0.9461245402102644]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9828559798666255, 0.999945590788774, 0.003376931411857122]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9614965756696068, 0.9995432372018054, 0.000985751618661401]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9999496688185194, 0.9999990226109572, 0.9843309226593687]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9999587649365006, 0.9999989374051526, 0.9860426311837209]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9564848406394574, 0.9995290064182019, 0.0009767765322408258]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9996289317188929, 0.9999851254034745, 0.9117999735213717]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9997408386451825, 0.9999945029052687, 0.014868053160599368]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9999560087559639, 0.9999989423385511, 0.9829954209583057]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9973423562819547, 0.9999049294894649, 0.002588245046379836]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9997109615503011, 0.9999922574882183, 0.4455976659880691]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9949862450432785, 0.9998034110173019, 0.0016599291798551883]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9380174784897667, 0.9995484552289852, 0.0009475351121966332]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9997936450338881, 0.9999920385100235, 0.9624849599101271]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.9999557758058243, 0.9999990168797586, 0.9868992823097708]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9998776717800879, 0.9999964268283346, 0.9743256876067677]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9988142177904733, 0.9999638293649674, 0.005172731526477567]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9984867047307242, 0.9999531203845483, 0.004413694259102841]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9997915472499204, 0.9999940490222647, 0.9544438772450242]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9723629395982026, 0.9998411535159573, 0.0018613790056538069]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9989289980945163, 0.9999740321107449, 0.005418909785345512]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9993027366701009, 0.9999787461984718, 0.22172999057014822]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9999140794042103, 0.9999979278088068, 0.6123876099021582]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9564534833681047, 0.9995289535935188, 0.0009966584228059828]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.9980997223526453, 0.9999332570709131, 0.003520907054370682]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9992482696969092, 0.9999677800008022, 0.9025153790985481]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9714591593908989, 0.9998140777645692, 0.0017993926870798112]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9994531609808143, 0.9999848519609421, 0.013285149267583351]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.9992404592838323, 0.9999730847417684, 0.010323050722785516]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9983717916535294, 0.9999429903860044, 0.012140856026501163]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9772822150029317, 0.9998113621926872, 0.0014996230746326094]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9640234615057895, 0.9996488814043978, 0.0012127441535670083]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9603020383088798, 0.9996567815355059, 0.001148440544342093]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.9998545125593019, 0.9999959959406883, 0.9691333120314755]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9994813554823643, 0.9999893913320652, 0.009652197696122466]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999796418084496, 0.9999997346317846, 0.9505851824870377]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9605690206244691, 0.9995398406526091, 0.0010210057109548483]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.9989121241484887, 0.9999575725996064, 0.038296224592595685]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9993730291691706, 0.9999794226421188, 0.012205557466670841]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9999795315947999, 0.9999995065939665, 0.9925791876101837]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9992331729626651, 0.9999808425573107, 0.0072139127910056475]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9999162599391211, 0.9999974945894848, 0.9767843344779824]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.9999085044415853, 0.9999972692371407, 0.9764256561399556]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.999775287018372, 0.9999941186732723, 0.9602243025305363]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9646208896797275, 0.9997399405494188, 0.0013478475138032362]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9643277870378401, 0.999697793043434, 0.0012784986955097345]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9637952676001804, 0.9997380700185045, 0.001365933871269974]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9724120069592662, 0.9999059332967893, 0.002404100201889901]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999434748202035, 0.999998780253938, 0.9837113669381131]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999864602321089, 0.9999998030119135, 0.9957092352502153]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9736800221530089, 0.9998784939330032, 0.0019932865920691805]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9997691717583496, 0.9999923633496346, 0.9568974764445771]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9996596773144812, 0.9999898619004423, 0.753751939323884]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9659520149129894, 0.999725127277835, 0.0013240088154722763]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999861712331499, 0.9999995234333439, 0.9921911147738044]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9985843050606495, 0.9999552489852658, 0.0041638121233021]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.999918630496532, 0.9999984965221234, 0.9640705286912489]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9991802232145057, 0.9999709907212081, 0.013163069383584993]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.966760317475419, 0.9997596205478123, 0.0013653698341053557]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9995188433336936, 0.9999894350498815, 0.0111749385527701]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999766410493443, 0.9999996310872343, 0.9923038547094908]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.999332755873926, 0.9999793955456681, 0.04255250463227573]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9493395144236181, 0.9994115522431786, 0.0008429513553319456]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.956469277949758, 0.9995287561881422, 0.0009841822002794158]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9999361272702677, 0.999998233200652, 0.9817059664172391]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.95721284662193, 0.9995043802381882, 0.0008647370599887226]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9998587994829911, 0.9999972690696179, 0.8551238502153338]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9429151160398914, 0.9991384519534392, 0.000664135214314546]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.9997562984851363, 0.9999938169985939, 0.8831819618881978]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9813035358337302, 0.9998962633049814, 0.0023587591433570595]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9999354236741103, 0.9999987140041862, 0.983583499596994]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999362556495976, 0.9999986291149422, 0.9832490015622293]
Epoch 5000 RMSE =  0.6668556454480853
Epoch 5100 RMSE =  0.6660749691556739
Epoch 5200 RMSE =  0.6653647648848631
Epoch 5300 RMSE =  0.6666533828716622
Epoch 5400 RMSE =  0.6664463788581024
Epoch 5500 RMSE =  0.6663440449533616
Epoch 5600 RMSE =  0.6656019062561451
Epoch 5700 RMSE =  0.6658089833406904
Epoch 5800 RMSE =  0.5747895077827012
Epoch 5900 RMSE =  0.506610208171427
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9999055551825781, 0.02624902446846103, 0.9967597106637893]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9982973504527924, 0.4284773705486605, 0.02782300441426087]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.999785629819594, 0.013196570246911366, 0.993998049465906]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9984151016793286, 0.669209726655824, 0.02180626293779889]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9992523033605245, 0.6461529145285331, 0.07114115042341995]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999655011259875, 0.12119863543074677, 0.9986275687077446]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9947439962128317, 0.32811284526865675, 0.008189624126252201]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9544459318375972, 0.17256292204778248, 0.004982254228809166]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9762136959765771, 0.3135641635957728, 0.007185020517809492]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9988843289946958, 0.7471431088556125, 0.02608468924763948]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9998129633070059, 0.30516832147282547, 0.8800848988235201]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9999257803740913, 0.049808744965630085, 0.99749033446374]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9544509570076997, 0.17097861479684195, 0.004988161132639892]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9738437871427706, 0.3362135257469962, 0.008513893531188728]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9997281500741372, 0.9447545516195655, 0.07196603007777494]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9603906610470588, 0.259072093591493, 0.006306444537467365]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.9998841810595523, 0.03277622460746403, 0.9962232096683868]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9352284247137409, 0.17505548396180892, 0.004757663288907729]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9987539530388175, 0.7165448069812036, 0.023760064644156777]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9470462341183657, 0.14043259721812046, 0.004266558001290343]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.997215994469636, 0.5001039600278631, 0.013160830694027443]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999795467748852, 0.09308478232474893, 0.9986268070272103]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.9996249602310981, 0.005105911199862179, 0.9903354703145709]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9710970861126033, 0.3786776245643731, 0.009371091641483146]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9994946020367151, 0.9154093505791516, 0.05138993319177633]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9993333265210143, 0.8019842390119503, 0.03983680140039235]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9732562581077204, 0.49842441696728296, 0.012814684912700899]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9998221602678177, 0.024591548493085136, 0.9939078989891331]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9711509526134493, 0.502721380763868, 0.012010896486842372]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9994190436971871, 0.8450576278737295, 0.043601228492744525]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.9998836580358085, 0.06300125305926955, 0.9932666868292792]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9948700714869599, 0.32335946767462864, 0.008108198751444572]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9711008531018892, 0.2809691681333612, 0.00686505462875492]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9996708106282495, 0.010836553879551795, 0.9912609164083237]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9994566459408707, 0.8990704122647257, 0.046223556710524256]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9544555689110664, 0.16816707282196675, 0.004981288368107508]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9552009057681344, 0.1839943680566833, 0.005088319509489075]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9999291596191774, 0.04544535390056944, 0.9975542717861513]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9707947433696633, 0.23296219872424195, 0.006921032968184538]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9998762012552672, 0.018633212415526772, 0.9957829211395914]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9991145203773565, 0.6706620768483456, 0.04246855154319958]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999743496693743, 0.9674304249623247, 0.8136883569950673]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9762416440203814, 0.3339429360072292, 0.007524516825158325]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9997604482532956, 0.8586714588819454, 0.1757187320367928]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9985154996649762, 0.6732364796734673, 0.02046291551795082]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9804192056711983, 0.4772470685170265, 0.011902546861496551]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9988891148032218, 0.0014867215674958605, 0.9810996488091501]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9980265330668925, 0.6755097044533377, 0.01865330771318839]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9987528445850496, 0.7326848328737936, 0.02521278567944826]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9996931835896602, 0.00880293157207275, 0.9919426713911688]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9643679320991739, 0.256496779591317, 0.006709314056876153]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.936592173476848, 0.10349109430090123, 0.003398445173074306]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.959708644330785, 0.1711987607778725, 0.005026840017611361]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999056723239015, 0.033574969534938805, 0.9970249746786478]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9995216826146873, 0.9071993527965563, 0.05048521242609846]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9724689536313992, 0.43620490458791394, 0.00996009737944988]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.999934587667099, 0.045769515144434975, 0.9976684403388174]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9996231126735504, 0.04756059251039278, 0.9438497088070104]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9999393809495111, 0.0448186573104025, 0.9979466597543849]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9982192443097466, 0.49336285618411646, 0.024838722491118286]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9996586966372119, 0.006275620538268119, 0.9920075570787323]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9999626343716689, 0.09674524426343901, 0.9983977198225239]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9994936136294086, 0.89365291146493, 0.049508151219642985]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.999187085531045, 0.7311232454043157, 0.032157350533819955]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9626899666362447, 0.23872784354407298, 0.0063851897144839315]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9684738577019566, 0.3901989420540371, 0.010714777992404967]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9623817338261649, 0.21035963091361284, 0.006051898396727825]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9820566368609698, 0.63078246482546, 0.0171590829181702]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9348108628218883, 0.06784963564732549, 0.002341537745158891]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9994585887280112, 0.003634061456383163, 0.9876127835115306]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9999697475965507, 0.08603207244057524, 0.9986683073385519]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.9987749178963733, 0.47307408398694223, 0.04520031966587948]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9701965132051534, 0.3433979317113326, 0.008707772024501022]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9587814237688793, 0.16709374151465087, 0.0050895159198393975]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9982312758361023, 0.5926103147411481, 0.017009567404550438]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9999050482352817, 0.04133387816052831, 0.9969027421433081]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9403046624658483, 0.09768550803463166, 0.0033380504397038284]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9971304347304892, 0.5643670482828587, 0.014931028569080348]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9995537710240577, 0.1572790521866286, 0.6713380066783247]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9585158852366414, 0.21652032584163908, 0.005653921324418983]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9621630686517088, 0.2650761029660332, 0.006722685420174739]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.9998648792056509, 0.017625622704907883, 0.9954919302279003]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9999165453985949, 0.03986740221022063, 0.9967893957043001]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999697490352296, 0.08108864396334813, 0.9985224886225852]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9993773599541138, 0.8325450191068112, 0.035582696460369975]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.9996784939878849, 0.38254919409402816, 0.4949516637285347]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9987718029280462, 0.6677967021262179, 0.02382476523275451]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9999357013117682, 0.052065848361425054, 0.9973720286017533]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9652523152065204, 0.28099668173169373, 0.006899882351366702]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.999167804634187, 0.4454968693416204, 0.12219856076808039]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9701655501079067, 0.33738106690422165, 0.00894753351928649]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9630353804179671, 0.2654360108816439, 0.006711191325103129]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.9552719372956736, 0.158921947639772, 0.004360721334790299]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9996713803234312, 0.9450380573642024, 0.07081006145420915]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9991970712057192, 0.8286257583156987, 0.03379179719006007]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9944626388267148, 0.377205153308176, 0.008446773429027724]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999802511016214, 0.31790942325018334, 0.9984667190964687]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9612591537747945, 0.26336859161954057, 0.006880729001743213]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9996439633554637, 0.928185792598541, 0.059065485531579844]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9996968809511794, 0.007477862260946412, 0.9918800088702829]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9988787362136158, 0.7825957720508142, 0.026545912724915598]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9393957778464388, 0.12072839431438877, 0.0038264994278811586]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.998007024671999, 0.5815365098334903, 0.01620747056102003]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9998197147256401, 0.015007636899039126, 0.9948870781419268]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9998966593428981, 0.8834840712385783, 0.4054578864126429]
Epoch 6000 RMSE =  0.5058141671369338
Epoch 6100 RMSE =  0.5079969115636365
Epoch 6200 RMSE =  0.5054678193276424
Epoch 6300 RMSE =  0.505775539575424
Epoch 6400 RMSE =  0.506441913316992
Epoch 6500 RMSE =  0.5048578998560188
Epoch 6600 RMSE =  0.5046525083624027
Epoch 6700 RMSE =  0.5040269904447876
Epoch 6800 RMSE =  0.5043672795843597
Epoch 6900 RMSE =  0.5034754455652174
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9993558327425524, 0.021299223709195222, 0.8710254737994315]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9694927385932497, 0.5069569210777864, 0.013211399336588818]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.9997843381152538, 0.0056104552320749195, 0.9917113628699143]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9993852258490624, 0.8568193467256843, 0.043884372071660835]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9998494888966406, 0.013712356508061727, 0.9943362775454726]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9717158113731724, 0.49673986578160384, 0.014088121564116032]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.999149025809423, 0.8308342489907143, 0.03722676677386127]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999722707640846, 0.9894786250797836, 0.5005727830297422]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.9994019118720108, 0.0015953950577908414, 0.981609143927138]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9561392215834981, 0.2160986554081685, 0.0063875377930631356]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9574191351980088, 0.17105658517101274, 0.0055886780065785135]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9694579168420446, 0.2777137088531204, 0.007639196784608314]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9519094418727017, 0.16564092752236342, 0.00553813403827935]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9983223806312727, 0.6652060968340664, 0.024199969680891442]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9998817761268408, 0.015631349611310814, 0.9953108186679408]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.999819131560352, 0.018661953443580025, 0.9903150243701699]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9997464763375199, 0.14718789690865056, 0.869892777456134]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9995152586722554, 0.0019450987181745952, 0.9880006649443585]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9684467476789381, 0.33536364970538934, 0.010014635250832933]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9986806609729186, 0.7339175786974111, 0.027532296543695028]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9519055088967178, 0.1657767077464982, 0.005545597192349622]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999683041465628, 0.0769057584292697, 0.9985106863858513]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9519154406236177, 0.1653845994761905, 0.0055455814022052005]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9990591861550697, 0.7158000052807723, 0.03532158189860036]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9312076412267085, 0.068932541469882, 0.0026014486671706275]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9996231885674596, 0.9283433405255124, 0.0656553399526072]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9317017572086823, 0.17182324298944943, 0.005284789715649569]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9994566863618557, 0.0019754181732702905, 0.9849573804090265]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9691539923208444, 0.23060678068972634, 0.007702848099783391]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9998668913962941, 0.012157731091525376, 0.9943888873154259]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9793171514534206, 0.4729764597382165, 0.013262990147052477]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9748865342473543, 0.3003883691739463, 0.008004432735301414]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.999658362838675, 0.0037448038647532335, 0.989622628006431]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.998814070298207, 0.7784841372552412, 0.02968324733279648]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9602900810610226, 0.20717442150334323, 0.006736803508176264]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9994256960547683, 0.8958582172992514, 0.051228621560299156]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9527353558116223, 0.17838643609719892, 0.005657435425979169]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.9527825434031771, 0.1555933880914006, 0.00487196099765159]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9990748326286489, 0.4990294228499412, 0.08090162711484473]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9441562345384719, 0.1355614358469425, 0.004736448853369484]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9994653152864739, 0.8956556823335527, 0.05245841889731155]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999674418920901, 0.030234832639720537, 0.9973605255542649]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9997127405412437, 0.004236052868692872, 0.9914372226433302]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9998496184556146, 0.008345840903745315, 0.9937980743041662]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9945787912515234, 0.3201011918902399, 0.008965187222101424]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9992062558126058, 0.7450105854893679, 0.04278709229005705]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9723835060392729, 0.32982319481375894, 0.009461345707935711]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9998498052021876, 0.010893518240441312, 0.994304691474394]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9600433378498826, 0.2627133928039745, 0.0075929244972186925]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9999405194537665, 0.03179920754357206, 0.997026055370654]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9992953196894897, 0.808751821919982, 0.04039818614031945]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9969639460755679, 0.565724920918555, 0.016625001821866575]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9694719668970511, 0.3719915224702289, 0.010417443665666696]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.9998973682797121, 0.014710890407544875, 0.9958230320990586]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9994881216590855, 0.044402166494791825, 0.8427121464748142]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.9331569548101731, 0.10211386143767301, 0.0037740986238499717]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9981752335984773, 0.47713706861857, 0.022454526523521173]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9685220555323965, 0.347194993244424, 0.00971008953077057]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9999034932249009, 0.014470429802389356, 0.9962101923995362]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9994943090756944, 0.9078169823986582, 0.055519297957701046]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.998697538583689, 0.5834326320032044, 0.02934679143110779]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9970550640965321, 0.49865631292790735, 0.01458942261638632]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9994855698190548, 0.0038073622300305704, 0.9834027734056204]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9979129520103001, 0.6813145319546301, 0.020755320632430973]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9998680199570368, 0.5707713815157011, 0.7202963836649666]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9667005326660957, 0.39255883635647987, 0.01200529345133156]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9997162372615125, 0.0058595353643872295, 0.9920218363850506]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9709126431627543, 0.43706556881122094, 0.011165669300486947]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9988195483956472, 0.7411073873282806, 0.02884755210272431]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.9998151813233627, 0.009319075785604184, 0.9937643553746898]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9986808618148963, 0.7169851017705845, 0.025976529815549553]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9999517649827411, 0.029830761044464343, 0.9974777263917715]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9994650910112659, 0.914193293114222, 0.05727102273720847]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9632760645163739, 0.281974183554231, 0.007749053905230581]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9982305512845884, 0.00046964936761320977, 0.9650441020032444]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9998870006126273, 0.01425326665522145, 0.995562408508325]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9991352576476218, 0.0010584434201581134, 0.9795819042667115]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9748832112052455, 0.33219330618817006, 0.008406536864553157]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9999449706570168, 0.03918595729268226, 0.9974887339805737]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9581727594656301, 0.253055336919024, 0.007054498957938856]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9997300180449319, 0.8237998381287818, 0.1892449308219868]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9810286489964479, 0.6306332147531892, 0.019169940110955866]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9623821432171346, 0.24993530723619034, 0.007490440415777656]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.9998957402577267, 0.014700924014931097, 0.995572436389068]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9997123664912281, 0.9425261620394226, 0.07886999134479478]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9996519860715499, 0.9439356105592706, 0.07888407730452084]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.9994579063863608, 0.0038482623219057797, 0.9816978888713871]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9984291057746775, 0.6689592490566151, 0.022534952365581604]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9605970555723129, 0.23315724147064046, 0.0071210631004116395]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9998026477415789, 0.005837952954116658, 0.9919682807634558]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.994439213558162, 0.3168146796714074, 0.009104823634436593]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.9991227882611105, 0.6983923759716293, 0.03882041539301053]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9999516858991107, 0.02612972169523412, 0.9973525176481334]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9981256715824602, 0.5971642136636478, 0.018936930807105744]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9986923118663086, 0.6698912515073105, 0.026094630117466277]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9941313252619822, 0.3811986014932841, 0.009470790111507366]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9563842365004427, 0.173299892425914, 0.005669871845586541]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9359545270307195, 0.1226323139248522, 0.004275272438706879]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9993317992537585, 0.8134824105679089, 0.04553167893453786]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9590338461957126, 0.26754647759049094, 0.007705377854311166]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9609032443917255, 0.2692959801206068, 0.007514164256345406]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.9978879936468497, 0.5853437978620213, 0.018132050684150372]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.936921350223468, 0.10063080721771117, 0.0037099529559287453]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9981058235381741, 0.5570279452485899, 0.020572024012902763]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9995105031697252, 0.002687254631063644, 0.9859980799614666]
Epoch 7000 RMSE =  0.503040778177455
Epoch 7100 RMSE =  0.5018204469991107
Epoch 7200 RMSE =  0.5019750375777106
Epoch 7300 RMSE =  0.5039049222412424
Epoch 7400 RMSE =  0.5046665462971613
Epoch 7500 RMSE =  0.5017743666337345
Epoch 7600 RMSE =  0.5023061560065782
Epoch 7700 RMSE =  0.5015874067540623
Epoch 7800 RMSE =  0.5015997760886102
Epoch 7900 RMSE =  0.5018441366459562
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.9997030703830665, 0.01298508523626794, 0.9908771585854168]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.9997463321967813, 0.31821995227087657, 0.8086705644848301]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.9999142318356811, 0.027698133349670542, 0.9946375429833437]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.9984289816355935, 0.0014200725361697957, 0.9632479861226568]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.9340698661460789, 0.1417734533531857, 0.004528564479570549]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9565501239571896, 0.2876121232110016, 0.007367237240468229]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.9987257806502423, 0.00170945492332973, 0.9761320947975766]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.9998433378425213, 0.028459870411812353, 0.9940134433623733]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9930220692902574, 0.3847182825231296, 0.00904395993341447]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.9986918237252601, 0.009169484876828017, 0.9222104828796613]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9526836943107382, 0.2720239080643545, 0.007270243239821625]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9481872222603085, 0.22093637853181394, 0.006111827093041318]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.9987145336242373, 0.002312451834051416, 0.9719873003592624]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.9991001054356199, 0.0034014716113639192, 0.9793368363616802]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.999729427394097, 0.013018030551129602, 0.9916875634357448]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9485065573099167, 0.1740513544554772, 0.005415018907936969]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9984423209330052, 0.687247600863589, 0.02368750321864721]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9993149203093595, 0.9016285326106449, 0.049092737974868114]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.998428178184311, 0.7267698139567069, 0.024500791640337336]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9537690503513091, 0.27369338662269893, 0.007178029197878247]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9996043825749126, 0.009843634057932255, 0.9885034777838454]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9626267649495799, 0.34390515244757325, 0.009569433669234288]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9975124908580096, 0.6832017500851905, 0.01985603104685725]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9638663420199841, 0.5086882498172229, 0.01279788667772134]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9515909828804597, 0.26712996450825377, 0.007359897451987388]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.9999164737738078, 0.06528934042604123, 0.9971274565867025]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9626796879294475, 0.35004767206501153, 0.009279275709810915]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.9997456743055343, 0.012548126382325367, 0.9923898379208951]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9638125559757763, 0.3738262247539138, 0.009976076823239092]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.9196448078610033, 0.17230110678292315, 0.005049650622046539]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9977431540175958, 0.5750917761309051, 0.017871802862024208]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9638477009169577, 0.2792599026464195, 0.007308109963507403]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.9996497036227134, 0.010680019720377222, 0.9886403196333877]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.9996890426005436, 0.013460008449255267, 0.9905965095920392]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9935412060675602, 0.3249984040041226, 0.008578899758868733]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9933796383020204, 0.3267148689286666, 0.008686042067527908]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.9977271576783229, 0.0009480687340473901, 0.9590951847155981]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9554472937912314, 0.2609989524708935, 0.00714390288809631]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9996569905969029, 0.9459170303754194, 0.07508275717485621]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9977668808390984, 0.6061298953332215, 0.018023541159247813]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9431766418073786, 0.17148241418070342, 0.005294497704795916]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.952969935156634, 0.2169387329306167, 0.006436296363003505]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9981276807533779, 0.6831474482321556, 0.021450634090298865]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9431942574677478, 0.17158717078342975, 0.005294224134405645]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.9999521592703704, 0.9431344981906501, 0.7868964581199446]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.949676500405463, 0.174907198689326, 0.005361442998712649]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.970211678880409, 0.338696729384581, 0.008018577289814538]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.9985697767081346, 0.0017512576719085047, 0.9700036769798324]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.95338857792875, 0.24087457159126682, 0.0068196243764765295]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9982612831150348, 0.0036507705498093, 0.9403163803132626]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.9994326286080643, 0.004857102206734621, 0.9834338551675992]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9991954230797377, 0.819603448468503, 0.0411830282901609]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9754240964854635, 0.4801233358165057, 0.012655007119450024]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9985928990542169, 0.7453694092824997, 0.02718287774067492]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.9212739224606332, 0.10381300151485334, 0.0035945061620208966]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.9985956235580936, 0.0030377002547935777, 0.9680426401986204]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.998585950435404, 0.785108218193772, 0.02834507501500133]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.9953544602810295, 0.00041347479113048593, 0.9306333672995346]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.9993356397501478, 0.022329969264555907, 0.9539065960435357]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9988421084739681, 0.7007586886153555, 0.0357929867057349]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.9974838697659087, 0.5875016314615158, 0.01725788016190093]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9431779875585949, 0.1705585827790483, 0.0052871766836669574]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.9992553063782117, 0.005055060459366028, 0.9842445406762016]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.9996032299329862, 0.007580190421640439, 0.9874224796589187]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9963781219207954, 0.5749088729440998, 0.01589812472041296]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9989855027174734, 0.8354362970107713, 0.036038477467003824]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.9996006300059986, 0.010759353875422055, 0.9898567111749508]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.9998728213847989, 0.02694432872872706, 0.9948196574154091]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9964852320028372, 0.5069279675871418, 0.013925887732488823]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9991526301744782, 0.8145256823663369, 0.03941692175144907]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.999724996987796, 0.013914676545740726, 0.9909545291142127]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9993957841535657, 0.9126467777765812, 0.05307252746561277]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.950468306321825, 0.26560086657589016, 0.006691756354655114]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.9440954285376626, 0.16641318235951263, 0.004646282020592305]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9979990837379555, 0.6776446478815441, 0.023119619615669183]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9701486302941007, 0.31693942935662356, 0.007633817220116123]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9992644606469224, 0.8591593804090224, 0.042875356449022094]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9633791589081295, 0.24075655254976513, 0.0073433048709216475]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.9440635121142716, 0.18928157843044316, 0.005393961230237443]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.9986206402774345, 0.3473948446151134, 0.11688926454843945]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9774464635277118, 0.6467090713115368, 0.018225972053175565]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.9996828514641202, 0.9174351287605232, 0.09290213090177733]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9993618459747774, 0.9173496810537893, 0.054352034053318586]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9672002050420502, 0.3397454395468469, 0.009014282842398831]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.9994799166502089, 0.005412327433208929, 0.9836192466417917]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.999550305040743, 0.9311099797759396, 0.06255840957162098]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.9246004090296127, 0.12410914310615676, 0.004058564220810008]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.9190338629326171, 0.07146509843434676, 0.0024704378117931234]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.9989561972797812, 0.7492345425227688, 0.030446398786552992]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.999022709094463, 0.747463163372743, 0.04077397900370476]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.9998551348496025, 0.036113757111820924, 0.9948203933837836]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.9986402555908003, 0.0026777584765962886, 0.9730671718686196]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9984279557717326, 0.7446460529146661, 0.026164729031575216]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9664792383825492, 0.5048150977667268, 0.013575512192197239]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.9605429821087436, 0.3983678292592215, 0.01135288999184185]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.9983866442129457, 0.5731294069377478, 0.028819936111359506]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9993616221856597, 0.9011454063607036, 0.049968991118038104]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.9995146848838434, 0.008548442955897156, 0.9870937914876959]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.9998727797612489, 0.023814809445748014, 0.9945659291624444]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9257365252321862, 0.10190626305688268, 0.0035203177627460835]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9995851784080851, 0.9474664831488918, 0.0751244833486719]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.9995336097925505, 0.013741729905194172, 0.9841973543951009]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.96551338111252, 0.4456339490273921, 0.010550760371118567]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.9977537576346768, 0.4899455750874358, 0.020376477811520594]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.9992421034469059, 0.003850561774363278, 0.9828068001988607]
Epoch 8000 RMSE =  0.5018927424794346
Epoch 8100 RMSE =  0.501569504922359
Epoch 8200 RMSE =  0.5018244462899848
Epoch 8300 RMSE =  0.5013280938772963
Epoch 8400 RMSE =  0.5009526140199229
Epoch 8500 RMSE =  0.5002219937048706
Epoch 8600 RMSE =  0.39230978755089435
Epoch 8700 RMSE =  0.3790956758356105
Epoch 8800 RMSE =  0.37401031820740116
Epoch 8900 RMSE =  0.3746310820121185
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9955244114180328, 0.7926795231742738, 0.004078792373261214]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [0.00019067398530791258, 0.0024894433444141934, 0.700772452996513]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [0.0032317317082911053, 0.07555961247673963, 0.9506961139031518]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9111673135466337, 0.31763098229937015, 0.0010856315674079558]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [0.0006274134509218137, 0.01424720355999006, 0.8655542155784088]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [0.0018211916986038267, 0.05372481237155452, 0.9364247652088313]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [0.003486589052272725, 0.07092716594969405, 0.9582910642418582]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.842019132761891, 0.19004945584561414, 0.0007673368998281929]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.014470860730632432, 0.3319435517476771, 0.9769349367009674]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.932029590172143, 0.6428110900254839, 0.002625287830705689]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [0.0015704585504609546, 0.02755976849044152, 0.912966519687725]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9011962206352577, 0.49892848499208464, 0.0019521472884333109]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9939976330367948, 0.6778651959162396, 0.0031156292984893556]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.992683622576603, 0.600458323701945, 0.0026372528540173435]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [0.0011912906988838742, 0.026825871882538856, 0.9205225371798821]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9879323579983076, 0.7206004628780983, 0.011554131309755593]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [0.006116438722113178, 0.1751574271698203, 0.9715854660822939]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9113640010630762, 0.34001312725578847, 0.0011384574934240727]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9969334764007892, 0.8429621526941957, 0.0070906326974520695]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.7975352588091263, 0.10087549254287292, 0.0005022747137529767]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9263511363217368, 0.48222106754572286, 0.0018136106281660552]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [0.010563412398363517, 0.2806051150154422, 0.9842368411727729]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [0.006999362648708701, 0.13316514596756127, 0.971414618211552]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.7955781558314133, 0.12088939242582095, 0.0005790694871467472]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9954208929745368, 0.7429566075200441, 0.004003626694049695]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.859371341361258, 0.2569245547187909, 0.000954234679222028]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8679389089303486, 0.2667334519636051, 0.0010215390460835542]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [0.0020751076346857827, 0.049276661095080145, 0.9450577526480202]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [0.010382415892538431, 0.13612069366348836, 0.970381662400949]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.8188073582125861, 0.13765295698779206, 0.0006434603529190939]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.8938913853686021, 0.23120712292100065, 0.0010483602353037107]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.8863101159965954, 0.38797447880903607, 0.0016312091131527506]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.8758240389288352, 0.2779977271865334, 0.0010494170888658554]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.7393438235645372, 0.29486460947364296, 0.04628286403286053]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9978322552233284, 0.8977498495196595, 0.007257408184922996]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.998075077550266, 0.907294594429222, 0.007920093292073466]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8526985774921562, 0.21415602088828756, 0.0008689298821934154]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [0.0006441403995516858, 0.012681582915428284, 0.8765373073384652]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [0.003284931336595219, 0.067508185218658, 0.954811055958304]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.861671916228113, 0.26100917529750073, 0.001049400750187117]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [0.002995721198549978, 0.0661271402369368, 0.9509330176369135]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.8988436905443424, 0.43271218395606714, 0.001518486545670379]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.9918213644369712, 0.5727552481679054, 0.002529308513884957]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9886838856344741, 0.5645463224114469, 0.0022883974578251603]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.8404364849583476, 0.165785539059319, 0.0007540575240389837]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [0.002864739066912021, 0.0716488121659173, 0.9492025394830775]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9985222717489891, 0.9269294839620724, 0.009618109652542919]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9967842088440704, 0.8286815002329567, 0.0053050760681268554]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.8540377801700015, 0.16909561837148812, 0.0007708223812725698]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [0.0012154122911879994, 0.03287794316336534, 0.9135558446302197]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9671137124722167, 0.4154415723186319, 0.005164575471553353]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [0.0006925106117134532, 0.012711755321153683, 0.860758595615945]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9986924861451925, 0.9459502410319991, 0.011479012966073543]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9937046875374028, 0.6684952224389216, 0.003364090423973294]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9988009027322842, 0.9429292464142334, 0.01196556188064345]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [0.005657515356195015, 0.14225599606449305, 0.9672547754260804]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9782980004271397, 0.38199891794394125, 0.0012955210325087927]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9901276929889183, 0.7105115977839382, 0.010563914985855126]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9979032210091802, 0.8993717761371348, 0.0075936447347408655]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9792775382664588, 0.33054573679618576, 0.001245017594994061]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.06559593773659558, 0.047152442570685876, 0.13476390893065968]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.8420075281251651, 0.166927033078643, 0.0006633474426423197]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.8905305382131817, 0.3475891587331637, 0.0013687510665373111]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9948013144457263, 0.7246232528773798, 0.003625887014859714]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.988921699029637, 0.5089528448457894, 0.002007218577053008]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.0006199461780241642, 0.01038367823782098, 0.8510418498084622]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.8713633146278906, 0.26687006080925635, 0.0010215473002076186]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9979852192063401, 0.9183419655265872, 0.008141922901723995]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [0.0018237728869534876, 0.04661085764377524, 0.9333642147449496]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [0.13778826358284624, 0.05705763114117333, 0.05885713814417476]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.8562231916292374, 0.1796505480167488, 0.000763268080838499]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.8650056620296009, 0.22131460762748426, 0.0009203742758279996]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.01732582692330774, 0.03677866448811921, 0.3310005565588826]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.8939952000514186, 0.2897221278202135, 0.0010378075806677808]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9950273337746441, 0.7458951190573978, 0.0038069679306236134]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [0.0022283306459033787, 0.059131270152615614, 0.9456754173023862]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [0.006964767398113237, 0.12480389148035967, 0.9701250420241454]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.8402883846722894, 0.17432576992975227, 0.0007511031267876237]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9798686500641328, 0.33629620488526224, 0.0012210634407696225]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.7875933403186163, 0.10927745175849127, 0.000510132999041375]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.992178427797528, 0.6568565513070228, 0.00415861519564072]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.9722996861285699, 0.5537131776035551, 0.011602593104652573]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.7837791441340289, 0.1849708382835887, 0.0007158841665937519]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.7831580871945291, 0.07469118748149424, 0.00035128452348372213]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.8947232932289562, 0.39141674022842726, 0.001420406363340794]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8655018735411086, 0.2787585764723124, 0.00103240590105584]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [0.002265639893620316, 0.043717870175764485, 0.9327659207371105]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [0.0011858732381323526, 0.022175620647867968, 0.9107123810415558]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9922664235300007, 0.691770611628106, 0.0028547214267040403]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [0.0009972096419043161, 0.019971403304014496, 0.8935318466799588]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.8918547328251503, 0.36332120940712587, 0.0013232335832878952]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [0.0007043383940053633, 0.010095346715001109, 0.8788757270011058]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [0.0007039405256420334, 0.015154789286276855, 0.8499514976505228]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [0.0005389192983723398, 0.01003952351892718, 0.8367756285873198]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [0.0017304473544120012, 0.031192225117582524, 0.9134199595056431]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [0.0005716486657409478, 0.008378250392286273, 0.8209307633918573]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [0.0025608956700938377, 0.06204660647523843, 0.9385524126113932]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9042897233702732, 0.34477540376428334, 0.001285617321311978]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [0.0022719582354044777, 0.05517486640878177, 0.9378807501677056]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.7467555986076697, 0.529747457789994, 0.15527440907540163]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [0.00038867406459595417, 0.005614630819810181, 0.8049403288780774]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.5690379672702147, 0.22474786282607973, 0.05349879170855731]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.8925923333195717, 0.5208730890322251, 0.0018242947032650796]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.8644509995153421, 0.25034224854142617, 0.0009640353313794855]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.8383907429117211, 0.17571238876807097, 0.0007501325868621063]
Epoch 9000 RMSE =  0.36980961327979606
Epoch 9100 RMSE =  0.3684805088671891
Epoch 9200 RMSE =  0.3693938613769353
Epoch 9300 RMSE =  0.3681347653329152
Epoch 9400 RMSE =  0.37013441799946034
Epoch 9500 RMSE =  0.37684049945773007
Epoch 9600 RMSE =  0.3661348407757445
Epoch 9700 RMSE =  0.36638672003825884
Epoch 9800 RMSE =  0.36708930560899566
Epoch 9900 RMSE =  0.36533683120324156
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.]
produced: [9.656060277234437e-07, 0.0025375661701859857, 0.6962511580958353]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.]
produced: [0.9961428884370859, 0.7916860834098457, 0.0008744749577562647]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8700835491726777, 0.22408733824913657, 0.00018394861388020998]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.]
produced: [1.9790845509884166e-06, 0.005666359698461363, 0.801771953925561]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.]
produced: [2.210959024137636e-05, 0.053062351554768565, 0.8567405988388533]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.]
produced: [1.1393598387559503e-05, 0.05636286045780959, 0.9370210825776418]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.858902651598749, 0.17317537750258347, 0.00016017715638321704]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8782917212347527, 0.2717877615283352, 0.00022302057380944756]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.]
produced: [0.0003170618491952753, 0.4291581524127852, 0.9587136654021331]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.]
produced: [0.15589314325907538, 0.47310433101607724, 0.0890859796532545]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.]
produced: [0.33281774499009725, 0.2006924969246981, 0.009523314963822768]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.]
produced: [0.8099416976405666, 0.10803449740914173, 0.00010870128003505716]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.9406230801418821, 0.6460951192874942, 0.00056006645738997]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.]
produced: [2.8500415006857494e-06, 0.008426434386355392, 0.819424476232494]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.]
produced: [4.1613150049359e-06, 0.015626707943188975, 0.8396889786897054]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.881956521409697, 0.2446534273431848, 0.00020676907633308532]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.]
produced: [1.1285017380814335e-05, 0.05974902575858087, 0.9450852171183999]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.858666806241117, 0.17130520576995284, 0.00016090688775903596]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.]
produced: [1.513076063292159e-05, 0.07145031998327872, 0.9502469492861506]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8831818554670027, 0.27192411153314033, 0.00021850410392746077]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.]
produced: [0.9357322825623584, 0.4823456626408588, 0.00038821361185507506]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9821254929786043, 0.3261991727706629, 0.0002649964725322425]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.]
produced: [3.540895295528775e-05, 0.12531562903434354, 0.9698306581363965]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.]
produced: [0.9070559077808903, 0.28342031909897214, 0.00022253500363061428]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.]
produced: [5.939251684968869e-06, 0.021883688510628777, 0.9098354596760259]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.]
produced: [0.9945523392732982, 0.6718682365572888, 0.0007179787986096699]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.]
produced: [0.9061143941441817, 0.23702886507841994, 0.00022434481025368516]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.]
produced: [0.9981317605564556, 0.9007691905470431, 0.0015598640959737526]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.818198140974624, 0.12292392570357129, 0.00012391220090092939]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8810164500465267, 0.2690822068720467, 0.00022146189303841004]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.]
produced: [0.9230036513871451, 0.33706169794052343, 0.00024376135880616983]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.]
produced: [0.996062149203655, 0.7454240732317289, 0.0008554046471254291]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.]
produced: [0.8876482856141984, 0.25817691416335514, 0.00021762430172146945]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.]
produced: [1.4542444275380247e-05, 0.07532296897111494, 0.9485469854787841]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.]
produced: [0.9932786605455222, 0.6796533076762414, 0.000612714772390897]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.]
produced: [0.9229889865541444, 0.309032034558187, 0.00023331341995185546]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.]
produced: [0.8762803005825335, 0.25713318566791155, 0.00020431612156248408]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9983609812691094, 0.9087016445568236, 0.0016987697298631122]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.]
produced: [5.2801959889126034e-05, 0.14132222057320643, 0.9700808507654921]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.]
produced: [0.874429725614083, 0.17152163343050217, 0.00016260707504089838]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.]
produced: [0.9049159431439301, 0.3366917234237304, 0.0002923211813530442]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.]
produced: [0.9974349875435412, 0.8448967297646757, 0.00148764419854906]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.]
produced: [0.9905102350140045, 0.4961887208264227, 0.00042888881705404014]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.]
produced: [2.8926998953481804e-05, 0.1458071997832331, 0.9667226052500006]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.]
produced: [0.9118148806420434, 0.43854118620998106, 0.0003247231053491444]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.]
produced: [0.8619675983642133, 0.18205488404444514, 0.00016454571061192425]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.]
produced: [3.581237873423355e-05, 0.13552343006472692, 0.9711473778122017]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.]
produced: [0.9936252382752288, 0.755482391245492, 0.0018645144558350913]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.]
produced: [0.9902753944241419, 0.5678172489452918, 0.0004897052525564582]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8217250513578556, 0.09920383307268105, 0.00010753958469783926]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.]
produced: [0.8091426903616653, 0.17345572128877215, 0.00015360138204496295]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.]
produced: [0.9990262564081408, 0.9440013591251779, 0.0025051614197946176]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.]
produced: [5.452707077984157e-05, 0.2850568988839865, 0.984081093621012]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.]
produced: [0.9935238829184517, 0.5938490949019862, 0.0005740537552750739]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.]
produced: [3.151076448309391e-05, 0.17523017380187847, 0.9713238821966931]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.]
produced: [0.8724796132052484, 0.17069389277473218, 0.00016472548453020023]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.]
produced: [5.07004414340332e-06, 0.01911794541353384, 0.8925953852404911]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.]
produced: [8.793869229285474e-06, 0.03007689045988694, 0.9136213286827578]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.]
produced: [0.9827524287643222, 0.3254475858694379, 0.00026265728412077266]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.9081451262955211, 0.37642595983609733, 0.00030550315127260625]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.]
produced: [1.662954321207328e-05, 0.07366037208727139, 0.9506436742582433]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.]
produced: [9.64844738280383e-06, 0.05311963889156846, 0.935458278403637]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.]
produced: [8.04168292707078e-06, 0.027793878162837137, 0.9126669193342436]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.]
produced: [0.9957572961378023, 0.7377825328910397, 0.0008213683034293619]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.]
produced: [0.8626589970006169, 0.1615145771691406, 0.00014205796486499865]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.]
produced: [1.6879194878973753e-05, 0.07067235463639314, 0.9544991299573309]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.]
produced: [0.9988853051030169, 0.9462014536650794, 0.0024758572087626224]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.]
produced: [0.9165309076315442, 0.33271794704437513, 0.0002775654540405463]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.]
produced: [3.377682031495178e-06, 0.014041470188269076, 0.8635125900717855]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.]
produced: [3.3667060457358083e-06, 0.013287266162478223, 0.8753916569381973]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.]
produced: [0.9925451114280915, 0.6486004553391603, 0.0009183269133353235]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.]
produced: [3.5475965629448828e-06, 0.013132892666813445, 0.8601886799404783]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.]
produced: [3.5758304495404883e-06, 0.00981465721009076, 0.8789353288829855]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.]
produced: [0.8829115365682405, 0.2134209681834507, 0.00019796484232973888]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.]
produced: [0.8912452204238744, 0.28317695625071543, 0.00022678403334280335]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.]
produced: [0.4675150863587864, 0.3116570684490538, 0.011386777286555318]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.]
produced: [0.9670842443694799, 0.6798059185043966, 0.0038129108251131515]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.]
produced: [1.2838914993438176e-05, 0.06120521954915116, 0.939112641963911]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.]
produced: [0.47626903232092016, 0.3063057028040137, 0.011404065355950453]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.]
produced: [9.120080953605463e-06, 0.04723351585335945, 0.9331913223956628]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.]
produced: [1.1204274269160405e-05, 0.043742993468236575, 0.933192379441541]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.]
produced: [0.8372108483323447, 0.14254539299445723, 0.00013916600550240718]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.9971274628425568, 0.8340543034908543, 0.0011600062488891288]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.]
produced: [9.378884435905263e-06, 0.05065129926691855, 0.9476663263746458]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.]
produced: [0.9036685149105544, 0.3553817370858232, 0.00028649354440117797]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.8986241737811024, 0.39718110128736867, 0.00035317392389140796]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.]
produced: [0.9067274385483529, 0.5055499729551012, 0.00039650042820944673]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.]
produced: [1.753002030074898e-05, 0.06983820205401972, 0.9586471677844169]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.]
produced: [0.9942523065696215, 0.6704069218694433, 0.0007045068741577922]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.]
produced: [0.9981142196157671, 0.8970968195772974, 0.0016674938802064348]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.]
produced: [6.097918865973063e-05, 0.009899267904505888, 0.22191184338130152]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [3.1126733182683424e-06, 0.010257792664621441, 0.8505978537120579]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.]
produced: [0.9421673463219727, 0.41290758463232635, 0.0013943334652510853]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.]
produced: [0.9982410741837974, 0.9164952984590455, 0.001761550408531077]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.]
produced: [5.908676861926679e-06, 0.02766214812847927, 0.9210269982853835]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.]
produced: [0.9809710730333708, 0.38656862383402, 0.0002789174843896546]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.]
produced: [0.9986072537636075, 0.9294180814364531, 0.0021479276949075326]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.9125380016471266, 0.5045322862837285, 0.00042273562181226437]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [5.316682250855232e-05, 0.018321972075505365, 0.43895163679254356]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.]
produced: [0.8043382846175312, 0.07154774997518079, 7.52485726757e-05]
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.]
produced: [0.992599052938322, 0.5864904433329005, 0.0005471385823022982]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.]
produced: [2.3266991604219592e-06, 0.00949799775552682, 0.8435948574946783]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.]
produced: [0.0833484956844109, 0.20510680555867214, 0.034678551320366305]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.]
produced: [0.9954270595540526, 0.7286389050460721, 0.0007667991393877112]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.]
produced: [0.85756186165466, 0.1757340188369987, 0.00015914938311525377]
Epoch 10000 RMSE =  0.36284941813718513
Final Epoch RMSE =  0.36284941813718513
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855]
Sample: [4.7 3.2 1.3 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931]
Sample: [5.1 3.7 1.5 0.4] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212]
Sample: [5.  3.  1.6 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045]
Sample: [5.4 3.4 1.5 0.4] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248]
Sample: [5.1 3.8 1.9 0.4] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987]
Sample: [4.6 3.2 1.4 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363]
Sample: [6.1 2.9 4.7 1.4] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402]
Sample: [6.2 2.2 4.5 1.5] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338]
Sample: [6.  2.7 5.1 1.6] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462]
Sample: [5.5 2.5 4.  1.3] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402]
Sample: [6.7 2.5 5.8 1.8] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515]
Sample: [5.8 2.8 5.1 2.4] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299, 3.7608174281035716e-06, 0.014665358065007964, 0.8636096040893242]
Sample: [6.  3.  4.8 1.8] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299, 3.7608174281035716e-06, 0.014665358065007964, 0.8636096040893242, 5.148166420441002e-06, 0.01723485356501731, 0.8254043331753776]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299, 3.7608174281035716e-06, 0.014665358065007964, 0.8636096040893242, 5.148166420441002e-06, 0.01723485356501731, 0.8254043331753776, 9.048021465527466e-06, 0.06012719610674018, 0.9259334153086257]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299, 3.7608174281035716e-06, 0.014665358065007964, 0.8636096040893242, 5.148166420441002e-06, 0.01723485356501731, 0.8254043331753776, 9.048021465527466e-06, 0.06012719610674018, 0.9259334153086257, 1.5831140595757064e-05, 0.08436181850737923, 0.9472833279365618]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299, 3.7608174281035716e-06, 0.014665358065007964, 0.8636096040893242, 5.148166420441002e-06, 0.01723485356501731, 0.8254043331753776, 9.048021465527466e-06, 0.06012719610674018, 0.9259334153086257, 1.5831140595757064e-05, 0.08436181850737923, 0.9472833279365618, 8.00277984163274e-06, 0.04666110321657383, 0.9203555180754475]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299, 3.7608174281035716e-06, 0.014665358065007964, 0.8636096040893242, 5.148166420441002e-06, 0.01723485356501731, 0.8254043331753776, 9.048021465527466e-06, 0.06012719610674018, 0.9259334153086257, 1.5831140595757064e-05, 0.08436181850737923, 0.9472833279365618, 8.00277984163274e-06, 0.04666110321657383, 0.9203555180754475, 5.658712930595864e-06, 0.030474903200337973, 0.90879119805224]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.]
produced: [0.8549969718651702, 0.1694535881058837, 0.00014494918500665855, 0.8373422484714422, 0.16201755218517355, 0.0001421282187472041, 0.8508448956499771, 0.14845992259111934, 0.00013899906858445314, 0.7568915000347969, 0.08602346644768227, 9.26985780898255e-05, 0.9370216526695228, 0.5192420196248017, 0.0004130096334557931, 0.9090071488410152, 0.36040166670977425, 0.0002666630611540212, 0.8804771281635742, 0.19538200703029668, 0.00016668485647432807, 0.8933057103576413, 0.30444199106438463, 0.00024056753612452045, 0.9148487129795438, 0.36534283728121786, 0.0002567650399550709, 0.8449483543505698, 0.1980944367745188, 0.0001577684743800828, 0.8903824012268513, 0.34962664370673446, 0.0002553381885405248, 0.9355045555039848, 0.41887570752251374, 0.00033977661214997987, 0.8426120467555258, 0.15465402534551853, 0.00014144769314855462, 0.9020925182257064, 0.3645395956597304, 0.0002864987413615684, 0.995096777786655, 0.8541411908744242, 0.002324627280305282, 0.9985438971122724, 0.9225966229348475, 0.001970942076864506, 0.9981818755263488, 0.8948325900294477, 0.001578939129802014, 0.9234231909856538, 0.19902791492714195, 0.00047284065749429304, 0.9970044299953782, 0.8281572485677708, 0.0010256921602071363, 0.9187627634434421, 0.630180326926188, 0.006677399407584402, 1.6244783646073916e-06, 0.006011284836450254, 0.7693179586576773, 7.02847451279048e-05, 0.043080587556171834, 0.6241267559498604, 0.997461622851953, 0.8629636100519782, 0.0012364919890265652, 0.9974757592177185, 0.8992869884995781, 0.0024158923024162225, 0.991285787699964, 0.5379498434749842, 0.0004633621039687871, 0.9945087824851825, 0.7006007674427178, 0.0006669984932003338, 2.630889497893533e-06, 0.010093243034883294, 0.8540328935151462, 0.9685098743650082, 0.4839970014634512, 0.0011416847925371512, 0.9958537078177019, 0.7570369088795003, 0.0008768453483143136, 1.6969589986908554e-05, 0.06626893705733378, 0.9439636109685667, 3.1003283432495323e-06, 0.010778044371379003, 0.846967233041402, 7.682456545092677e-06, 0.02535135583878267, 0.9118099050506515, 4.942504565283219e-06, 0.01795589456666775, 0.8633554658294309, 7.918114013889404e-06, 0.04364245236281236, 0.9218701489567451, 6.076171494211528e-06, 0.028817907235439918, 0.9177206579395242, 2.6467203507020848e-06, 0.009808299228189347, 0.8316916585170013, 1.3868199806153627e-05, 0.07743593958409627, 0.9572570009299012, 4.413525230056565e-06, 0.017910626998270146, 0.8547515548097299, 3.7608174281035716e-06, 0.014665358065007964, 0.8636096040893242, 5.148166420441002e-06, 0.01723485356501731, 0.8254043331753776, 9.048021465527466e-06, 0.06012719610674018, 0.9259334153086257, 1.5831140595757064e-05, 0.08436181850737923, 0.9472833279365618, 8.00277984163274e-06, 0.04666110321657383, 0.9203555180754475, 5.658712930595864e-06, 0.030474903200337973, 0.90879119805224, 8.23049433540037e-06, 0.044711292917736595, 0.9267787569176379]
RMSE =  0.378885824045311

Sample Run of run_sin():
Sample: [0.97] expected: [0.82488571]
produced: [0.7879791336833528]
Sample: [1.13] expected: [0.90441219]
produced: [0.807114345243044]
Sample: [0.2] expected: [0.19866933]
produced: [0.674028344808951]
Sample: [0.38] expected: [0.37092047]
produced: [0.702903127452415]
Sample: [1.3] expected: [0.96355819]
produced: [0.8243195434340881]
Sample: [0.24] expected: [0.23770263]
produced: [0.6793793414250676]
Sample: [1.34] expected: [0.97348454]
produced: [0.8278610130279128]
Sample: [0.23] expected: [0.22797752]
produced: [0.6770295651685889]
Sample: [1.12] expected: [0.90010044]
produced: [0.8033546210575696]
Sample: [1.47] expected: [0.99492435]
produced: [0.8402391218399383]
Sample: [0.44] expected: [0.42593947]
produced: [0.7108692390508364]
Sample: [1.22] expected: [0.93909936]
produced: [0.8144617444086172]
Sample: [1.05] expected: [0.86742323]
produced: [0.7952200680116367]
Sample: [1.36] expected: [0.9778646]
produced: [0.829518219942977]
Sample: [0.48] expected: [0.46177918]
produced: [0.717175170442466]
Epoch 0 RMSE =  0.2590728539947286
Epoch 100 RMSE =  0.24324473853894787
Epoch 200 RMSE =  0.24164988675633187
Epoch 300 RMSE =  0.23917498683751598
Epoch 400 RMSE =  0.23498831551379012
Epoch 500 RMSE =  0.227707333329023
Epoch 600 RMSE =  0.21589828243680717
Epoch 700 RMSE =  0.1998969648231568
Epoch 800 RMSE =  0.1823529640799749
Epoch 900 RMSE =  0.16576708315851124
Sample: [1.13] expected: [0.90441219]
produced: [0.7819693091701393]
Sample: [1.36] expected: [0.9778646]
produced: [0.8277196222585559]
Sample: [0.24] expected: [0.23770263]
produced: [0.44818091633847207]
Sample: [1.05] expected: [0.86742323]
produced: [0.7627053446231745]
Sample: [1.47] expected: [0.99492435]
produced: [0.8450431155215836]
Sample: [0.38] expected: [0.37092047]
produced: [0.51482566271517]
Sample: [1.22] expected: [0.93909936]
produced: [0.8017060274260552]
Sample: [0.2] expected: [0.19866933]
produced: [0.4289002073663146]
Sample: [1.34] expected: [0.97348454]
produced: [0.8240951600369743]
Sample: [0.48] expected: [0.46177918]
produced: [0.5604404861195129]
Sample: [1.12] expected: [0.90010044]
produced: [0.7797432194839123]
Sample: [0.44] expected: [0.42593947]
produced: [0.5422759129353657]
Sample: [0.97] expected: [0.82488571]
produced: [0.741192129934123]
Sample: [0.23] expected: [0.22797752]
produced: [0.44282387233506526]
Sample: [1.3] expected: [0.96355819]
produced: [0.8167194371013542]
Epoch 1000 RMSE =  0.15114131372430814
Epoch 1100 RMSE =  0.13851269081931747
Epoch 1200 RMSE =  0.12762072682868686
Epoch 1300 RMSE =  0.11816867332426059
Epoch 1400 RMSE =  0.10990263400592518
Epoch 1500 RMSE =  0.10261858969440286
Epoch 1600 RMSE =  0.09615586962297798
Epoch 1700 RMSE =  0.09038725556590439
Epoch 1800 RMSE =  0.08521331891749004
Epoch 1900 RMSE =  0.08054967716523874
Sample: [1.47] expected: [0.99492435]
produced: [0.8982744080380839]
Sample: [1.12] expected: [0.90010044]
produced: [0.8377942474113483]
Sample: [0.48] expected: [0.46177918]
produced: [0.5170586719984442]
Sample: [1.3] expected: [0.96355819]
produced: [0.8743900293020652]
Sample: [0.23] expected: [0.22797752]
produced: [0.31621879986392143]
Sample: [0.2] expected: [0.19866933]
produced: [0.2934235883802151]
Sample: [0.38] expected: [0.37092047]
produced: [0.4360510067242925]
Sample: [1.22] expected: [0.93909936]
produced: [0.859539934905199]
Sample: [0.97] expected: [0.82488571]
produced: [0.7937415721488016]
Sample: [1.34] expected: [0.97348454]
produced: [0.8806617029974328]
Sample: [0.24] expected: [0.23770263]
produced: [0.3236667123552469]
Sample: [1.36] expected: [0.9778646]
produced: [0.8836438518218087]
Sample: [1.13] expected: [0.90441219]
produced: [0.8401084343575658]
Sample: [1.05] expected: [0.86742323]
produced: [0.8191599144994283]
Sample: [0.44] expected: [0.42593947]
produced: [0.48517632334188304]
Epoch 2000 RMSE =  0.07632971149721597
Epoch 2100 RMSE =  0.07249765878474364
Epoch 2200 RMSE =  0.06900642882850268
Epoch 2300 RMSE =  0.06581811774747581
Epoch 2400 RMSE =  0.06289712702623623
Epoch 2500 RMSE =  0.06021598822788662
Epoch 2600 RMSE =  0.057749616167951795
Epoch 2700 RMSE =  0.05547618911639781
Epoch 2800 RMSE =  0.05337693241062084
Epoch 2900 RMSE =  0.051435329496788466
Sample: [1.22] expected: [0.93909936]
produced: [0.884312654893272]
Sample: [1.36] expected: [0.9778646]
produced: [0.9067761271634861]
Sample: [0.48] expected: [0.46177918]
produced: [0.5025544989808827]
Sample: [1.13] expected: [0.90441219]
produced: [0.8652723723450841]
Sample: [1.3] expected: [0.96355819]
produced: [0.8981077396898389]
Sample: [1.12] expected: [0.90010044]
produced: [0.8629957004136559]
Sample: [1.05] expected: [0.86742323]
produced: [0.8444336913371532]
Sample: [1.47] expected: [0.99492435]
produced: [0.9199969846909057]
Sample: [0.44] expected: [0.42593947]
produced: [0.4653441546042734]
Sample: [1.34] expected: [0.97348454]
produced: [0.9041282307663918]
Sample: [0.23] expected: [0.22797752]
produced: [0.2720137795258257]
Sample: [0.2] expected: [0.19866933]
produced: [0.24751342543482407]
Sample: [0.97] expected: [0.82488571]
produced: [0.8185481192803877]
Sample: [0.24] expected: [0.23770263]
produced: [0.2802612748173239]
Sample: [0.38] expected: [0.37092047]
produced: [0.40786307347019096]
Epoch 3000 RMSE =  0.04963666801326939
Epoch 3100 RMSE =  0.04796876715584866
Epoch 3200 RMSE =  0.04641966269165311
Epoch 3300 RMSE =  0.04497916956326236
Epoch 3400 RMSE =  0.04363828938805076
Epoch 3500 RMSE =  0.042388745471386685
Epoch 3600 RMSE =  0.04122328488779608
Epoch 3700 RMSE =  0.0401350950811841
Epoch 3800 RMSE =  0.039118218926328774
Epoch 3900 RMSE =  0.03816727308795575
Sample: [1.36] expected: [0.9778646]
produced: [0.9189240575409952]
Sample: [1.3] expected: [0.96355819]
produced: [0.9106968582837484]
Sample: [1.34] expected: [0.97348454]
produced: [0.9163765950120852]
Sample: [0.24] expected: [0.23770263]
produced: [0.2593937599204807]
Sample: [0.2] expected: [0.19866933]
produced: [0.2258285662760839]
Sample: [0.38] expected: [0.37092047]
produced: [0.39345867461380374]
Sample: [1.47] expected: [0.99492435]
produced: [0.9312478549134954]
Sample: [1.05] expected: [0.86742323]
produced: [0.8579903158753301]
Sample: [1.12] expected: [0.90010044]
produced: [0.8765265007856747]
Sample: [1.22] expected: [0.93909936]
produced: [0.8975534701378561]
Sample: [0.48] expected: [0.46177918]
produced: [0.4953632238147694]
Sample: [0.44] expected: [0.42593947]
produced: [0.45472067128580934]
Sample: [0.97] expected: [0.82488571]
produced: [0.831840487276224]
Sample: [1.13] expected: [0.90441219]
produced: [0.878809742410875]
Sample: [0.23] expected: [0.22797752]
produced: [0.25061488120344655]
Epoch 4000 RMSE =  0.037277108945050604
Epoch 4100 RMSE =  0.03644326765063487
Epoch 4200 RMSE =  0.035661613707181655
Epoch 4300 RMSE =  0.03492810548520346
Epoch 4400 RMSE =  0.03423981046842891
Epoch 4500 RMSE =  0.03359301210479786
Epoch 4600 RMSE =  0.03298486552469715
Epoch 4700 RMSE =  0.03241282176777248
Epoch 4800 RMSE =  0.03187414968642744
Epoch 4900 RMSE =  0.031366755383267594
Sample: [1.12] expected: [0.90010044]
produced: [0.8846754309340059]
Sample: [1.47] expected: [0.99492435]
produced: [0.9381559510037002]
Sample: [1.13] expected: [0.90441219]
produced: [0.8870773718331726]
Sample: [0.24] expected: [0.23770263]
produced: [0.24786656003940719]
Sample: [0.38] expected: [0.37092047]
produced: [0.38505119831798856]
Sample: [0.97] expected: [0.82488571]
produced: [0.8399941922209078]
Sample: [1.36] expected: [0.9778646]
produced: [0.9263355518334546]
Sample: [0.48] expected: [0.46177918]
produced: [0.49073881886901444]
Sample: [0.23] expected: [0.22797752]
produced: [0.23904509168485194]
Sample: [1.3] expected: [0.96355819]
produced: [0.9182963835146057]
Sample: [0.2] expected: [0.19866933]
produced: [0.21408775197317784]
Sample: [1.05] expected: [0.86742323]
produced: [0.8661798160703399]
Sample: [1.34] expected: [0.97348454]
produced: [0.9237914659970295]
Sample: [1.22] expected: [0.93909936]
produced: [0.9054403433700557]
Sample: [0.44] expected: [0.42593947]
produced: [0.4485762792550624]
Epoch 5000 RMSE =  0.030888382913199498
Epoch 5100 RMSE =  0.030437034450486542
Epoch 5200 RMSE =  0.03001085812870153
Epoch 5300 RMSE =  0.02960839467199867
Epoch 5400 RMSE =  0.029227752820855938
Epoch 5500 RMSE =  0.028867828824549396
Epoch 5600 RMSE =  0.028526889084383392
Epoch 5700 RMSE =  0.02820379463309011
Epoch 5800 RMSE =  0.027897608490155112
Epoch 5900 RMSE =  0.027606984965166774
Sample: [0.24] expected: [0.23770263]
produced: [0.24110815823990125]
Sample: [1.05] expected: [0.86742323]
produced: [0.8715155618571964]
Sample: [1.12] expected: [0.90010044]
produced: [0.8899978086975049]
Sample: [1.3] expected: [0.96355819]
produced: [0.923359824181127]
Sample: [1.13] expected: [0.90441219]
produced: [0.8923830246183325]
Sample: [1.36] expected: [0.9778646]
produced: [0.9312487579952361]
Sample: [0.23] expected: [0.22797752]
produced: [0.23248801390087678]
Sample: [1.34] expected: [0.97348454]
produced: [0.9287892616305222]
Sample: [0.2] expected: [0.19866933]
produced: [0.20755645091707248]
Sample: [0.44] expected: [0.42593947]
produced: [0.44464525325319487]
Sample: [0.97] expected: [0.82488571]
produced: [0.8451808377447857]
Sample: [1.22] expected: [0.93909936]
produced: [0.9106465281105288]
Sample: [0.38] expected: [0.37092047]
produced: [0.37975397055081284]
Sample: [1.47] expected: [0.99492435]
produced: [0.9427617464167619]
Sample: [0.48] expected: [0.46177918]
produced: [0.4877281326372944]
Epoch 6000 RMSE =  0.02733096642149385
Epoch 6100 RMSE =  0.027068682246718503
Epoch 6200 RMSE =  0.02681917223313593
Epoch 6300 RMSE =  0.02658161752259427
Epoch 6400 RMSE =  0.02635554969627602
Epoch 6500 RMSE =  0.02613986481558371
Epoch 6600 RMSE =  0.025934146545300473
Epoch 6700 RMSE =  0.025737728539788996
Epoch 6800 RMSE =  0.025550000082658328
Epoch 6900 RMSE =  0.025370427567488293
Sample: [0.38] expected: [0.37092047]
produced: [0.3761326829766525]
Sample: [0.48] expected: [0.46177918]
produced: [0.4852401745772914]
Sample: [0.44] expected: [0.42593947]
produced: [0.44142488852611195]
Sample: [1.47] expected: [0.99492435]
produced: [0.9459247277689921]
Sample: [1.3] expected: [0.96355819]
produced: [0.9267856276116825]
Sample: [1.22] expected: [0.93909936]
produced: [0.9141423859561686]
Sample: [1.13] expected: [0.90441219]
produced: [0.895903624444425]
Sample: [0.2] expected: [0.19866933]
produced: [0.20354302409173994]
Sample: [1.36] expected: [0.9778646]
produced: [0.9346082622872527]
Sample: [0.23] expected: [0.22797752]
produced: [0.2284204263445263]
Sample: [0.97] expected: [0.82488571]
produced: [0.848432406012226]
Sample: [1.05] expected: [0.86742323]
produced: [0.8749985150042573]
Sample: [0.24] expected: [0.23770263]
produced: [0.23705835158752578]
Sample: [1.12] expected: [0.90010044]
produced: [0.8935467926477713]
Sample: [1.34] expected: [0.97348454]
produced: [0.9321517204136367]
Epoch 7000 RMSE =  0.025198509588605825
Epoch 7100 RMSE =  0.025033929896378364
Epoch 7200 RMSE =  0.02487611664364971
Epoch 7300 RMSE =  0.024724732190753808
Epoch 7400 RMSE =  0.0245793292347774
Epoch 7500 RMSE =  0.02443964830425931
Epoch 7600 RMSE =  0.02430531726737324
Epoch 7700 RMSE =  0.024175995311104523
Epoch 7800 RMSE =  0.024051453828115223
Epoch 7900 RMSE =  0.023931412816809553
Sample: [0.48] expected: [0.46177918]
produced: [0.4834109422164047]
Sample: [0.97] expected: [0.82488571]
produced: [0.8505069826124017]
Sample: [0.2] expected: [0.19866933]
produced: [0.2012418236312212]
Sample: [0.24] expected: [0.23770263]
produced: [0.23461235563870306]
Sample: [1.3] expected: [0.96355819]
produced: [0.9292805598983501]
Sample: [1.22] expected: [0.93909936]
produced: [0.9166492350027085]
Sample: [1.05] expected: [0.86742323]
produced: [0.877347846243026]
Sample: [1.12] expected: [0.90010044]
produced: [0.895996710917387]
Sample: [0.38] expected: [0.37092047]
produced: [0.37361336043888577]
Sample: [1.13] expected: [0.90441219]
produced: [0.8983521883400425]
Sample: [0.44] expected: [0.42593947]
produced: [0.4393195951195307]
Sample: [1.34] expected: [0.97348454]
produced: [0.9346015271200351]
Sample: [1.36] expected: [0.9778646]
produced: [0.9370598096537816]
Sample: [1.47] expected: [0.99492435]
produced: [0.9483486319866107]
Sample: [0.23] expected: [0.22797752]
produced: [0.22606545538555867]
Epoch 8000 RMSE =  0.02381558188652001
Epoch 8100 RMSE =  0.023703959346486853
Epoch 8200 RMSE =  0.02359600059816145
Epoch 8300 RMSE =  0.02349167962992882
Epoch 8400 RMSE =  0.023390729251951226
Epoch 8500 RMSE =  0.023293053457502745
Epoch 8600 RMSE =  0.02319845641014814
Epoch 8700 RMSE =  0.023106735276772292
Epoch 8800 RMSE =  0.02301785318778183
Epoch 8900 RMSE =  0.022931562541867397
Sample: [1.22] expected: [0.93909936]
produced: [0.9185236671684742]
Sample: [0.38] expected: [0.37092047]
produced: [0.372006280832028]
Sample: [0.44] expected: [0.42593947]
produced: [0.43778017638537536]
Sample: [1.13] expected: [0.90441219]
produced: [0.9001388033807449]
Sample: [1.47] expected: [0.99492435]
produced: [0.9501652000922678]
Sample: [1.34] expected: [0.97348454]
produced: [0.9365115783716154]
Sample: [0.23] expected: [0.22797752]
produced: [0.2247587606453749]
Sample: [0.97] expected: [0.82488571]
produced: [0.8520235310520353]
Sample: [1.3] expected: [0.96355819]
produced: [0.9311949904192673]
Sample: [0.2] expected: [0.19866933]
produced: [0.20013377421785425]
Sample: [1.36] expected: [0.9778646]
produced: [0.9389553475868276]
Sample: [0.48] expected: [0.46177918]
produced: [0.48197289859988507]
Sample: [1.05] expected: [0.86742323]
produced: [0.8790090147887281]
Sample: [0.24] expected: [0.23770263]
produced: [0.23331508387382618]
Sample: [1.12] expected: [0.90010044]
produced: [0.8977793699437138]
Epoch 9000 RMSE =  0.02284777347467275
Epoch 9100 RMSE =  0.02276638011119428
Epoch 9200 RMSE =  0.02268723603463689
Epoch 9300 RMSE =  0.022610266840024987
Epoch 9400 RMSE =  0.022535382448480485
Epoch 9500 RMSE =  0.022462425561176034
Epoch 9600 RMSE =  0.022391390499173443
Epoch 9700 RMSE =  0.02232216328027156
Epoch 9800 RMSE =  0.02225464489557706
Epoch 9900 RMSE =  0.022188743541105997
Sample: [1.22] expected: [0.93909936]
produced: [0.9198883827334589]
Sample: [1.36] expected: [0.9778646]
produced: [0.9403713349559]
Sample: [0.97] expected: [0.82488571]
produced: [0.85291006387322]
Sample: [1.34] expected: [0.97348454]
produced: [0.9379211858141883]
Sample: [0.48] expected: [0.46177918]
produced: [0.4806065304129592]
Sample: [1.47] expected: [0.99492435]
produced: [0.9515769341687255]
Sample: [1.13] expected: [0.90441219]
produced: [0.9014152067778536]
Sample: [1.05] expected: [0.86742323]
produced: [0.8801203224646904]
Sample: [0.23] expected: [0.22797752]
produced: [0.2239703844222145]
Sample: [0.38] expected: [0.37092047]
produced: [0.3707087825149813]
Sample: [1.12] expected: [0.90010044]
produced: [0.8990233043128568]
Sample: [0.24] expected: [0.23770263]
produced: [0.23255371986863377]
Sample: [0.2] expected: [0.19866933]
produced: [0.19950349168122616]
Sample: [1.3] expected: [0.96355819]
produced: [0.9326016700711101]
Sample: [0.44] expected: [0.42593947]
produced: [0.43647811439693057]
Epoch 10000 RMSE =  0.022124537295811583
Final Epoch RMSE =  0.022124537295811583
Sample: [0.] expected: [0.]
produced: [0.08411152663924558]
Sample: [0.01] expected: [0.00999983]
produced: [0.08411152663924558, 0.08806967710163104]
Sample: [0.02] expected: [0.01999867]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386]
Sample: [0.03] expected: [0.0299955]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648]
Sample: [0.04] expected: [0.03998933]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757]
Sample: [0.05] expected: [0.04997917]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103]
Sample: [0.06] expected: [0.05996401]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223]
Sample: [0.07] expected: [0.06994285]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254]
Sample: [0.08] expected: [0.07991469]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041]
Sample: [0.09] expected: [0.08987855]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423]
Sample: [0.1] expected: [0.09983342]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824]
Sample: [0.11] expected: [0.1097783]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262]
Sample: [0.12] expected: [0.11971221]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425]
Sample: [0.13] expected: [0.12963414]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382]
Sample: [0.14] expected: [0.13954311]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923]
Sample: [0.15] expected: [0.14943813]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865]
Sample: [0.16] expected: [0.15931821]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827]
Sample: [0.17] expected: [0.16918235]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963]
Sample: [0.18] expected: [0.17902957]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636]
Sample: [0.19] expected: [0.18885889]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483]
Sample: [0.21] expected: [0.2084599]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892]
Sample: [0.22] expected: [0.21822962]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225]
Sample: [0.25] expected: [0.24740396]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068]
Sample: [0.26] expected: [0.25708055]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777]
Sample: [0.27] expected: [0.26673144]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084]
Sample: [0.28] expected: [0.27635565]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507]
Sample: [0.29] expected: [0.28595223]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696]
Sample: [0.3] expected: [0.29552021]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667]
Sample: [0.31] expected: [0.30505864]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907]
Sample: [0.32] expected: [0.31456656]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813]
Sample: [0.33] expected: [0.32404303]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665]
Sample: [0.34] expected: [0.33348709]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334]
Sample: [0.35] expected: [0.34289781]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816]
Sample: [0.36] expected: [0.35227423]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877]
Sample: [0.37] expected: [0.36161543]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648]
Sample: [0.39] expected: [0.38018842]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058]
Sample: [0.4] expected: [0.38941834]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896]
Sample: [0.41] expected: [0.39860933]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563]
Sample: [0.42] expected: [0.40776045]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933]
Sample: [0.43] expected: [0.4168708]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534]
Sample: [0.45] expected: [0.43496553]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454]
Sample: [0.46] expected: [0.44394811]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875]
Sample: [0.47] expected: [0.45288629]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565]
Sample: [0.49] expected: [0.47062589]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374]
Sample: [0.5] expected: [0.47942554]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826]
Sample: [0.51] expected: [0.48817725]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435]
Sample: [0.52] expected: [0.49688014]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197]
Sample: [0.53] expected: [0.50553334]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675]
Sample: [0.54] expected: [0.51413599]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529]
Sample: [0.55] expected: [0.52268723]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892]
Sample: [0.56] expected: [0.5311862]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567]
Sample: [0.57] expected: [0.53963205]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463]
Sample: [0.58] expected: [0.54802394]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199]
Sample: [0.59] expected: [0.55636102]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817]
Sample: [0.6] expected: [0.56464247]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857]
Sample: [0.61] expected: [0.57286746]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914]
Sample: [0.62] expected: [0.58103516]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393]
Sample: [0.63] expected: [0.58914476]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207]
Sample: [0.64] expected: [0.59719544]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738]
Sample: [0.65] expected: [0.60518641]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406]
Sample: [0.66] expected: [0.61311685]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141]
Sample: [0.67] expected: [0.62098599]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907]
Sample: [0.68] expected: [0.62879302]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724]
Sample: [0.69] expected: [0.63653718]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561]
Sample: [0.7] expected: [0.64421769]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308]
Sample: [0.71] expected: [0.65183377]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077]
Sample: [0.72] expected: [0.65938467]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137]
Sample: [0.73] expected: [0.66686964]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427]
Sample: [0.74] expected: [0.67428791]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803]
Sample: [0.75] expected: [0.68163876]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622]
Sample: [0.76] expected: [0.68892145]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559]
Sample: [0.77] expected: [0.69613524]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784]
Sample: [0.78] expected: [0.70327942]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867]
Sample: [0.79] expected: [0.71035327]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685]
Sample: [0.8] expected: [0.71735609]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664]
Sample: [0.81] expected: [0.72428717]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528]
Sample: [0.82] expected: [0.73114583]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215]
Sample: [0.83] expected: [0.73793137]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018]
Sample: [0.84] expected: [0.74464312]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153]
Sample: [0.85] expected: [0.75128041]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329]
Sample: [0.86] expected: [0.75784256]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166]
Sample: [0.87] expected: [0.76432894]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592]
Sample: [0.88] expected: [0.77073888]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498]
Sample: [0.89] expected: [0.77707175]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678]
Sample: [0.9] expected: [0.78332691]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194]
Sample: [0.91] expected: [0.78950374]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098]
Sample: [0.92] expected: [0.79560162]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859]
Sample: [0.93] expected: [0.80161994]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997]
Sample: [0.94] expected: [0.8075581]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169]
Sample: [0.95] expected: [0.8134155]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681]
Sample: [0.96] expected: [0.81919157]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704]
Sample: [0.98] expected: [0.83049737]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659]
Sample: [0.99] expected: [0.83602598]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482]
Sample: [1.] expected: [0.84147098]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243]
Sample: [1.01] expected: [0.84683184]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001]
Sample: [1.02] expected: [0.85210802]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658]
Sample: [1.03] expected: [0.85729899]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195]
Sample: [1.04] expected: [0.86240423]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377]
Sample: [1.06] expected: [0.87235548]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875]
Sample: [1.07] expected: [0.8772005]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246]
Sample: [1.08] expected: [0.88195781]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465]
Sample: [1.09] expected: [0.88662691]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892]
Sample: [1.1] expected: [0.89120736]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998]
Sample: [1.11] expected: [0.89569869]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061]
Sample: [1.14] expected: [0.9086335]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768]
Sample: [1.15] expected: [0.91276394]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397]
Sample: [1.16] expected: [0.91680311]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772]
Sample: [1.17] expected: [0.9207506]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624]
Sample: [1.18] expected: [0.92460601]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892]
Sample: [1.19] expected: [0.92836897]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038]
Sample: [1.2] expected: [0.93203909]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027]
Sample: [1.21] expected: [0.935616]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507]
Sample: [1.23] expected: [0.9424888]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614]
Sample: [1.24] expected: [0.945784]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291]
Sample: [1.25] expected: [0.94898462]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611]
Sample: [1.26] expected: [0.95209034]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998]
Sample: [1.27] expected: [0.95510086]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597]
Sample: [1.28] expected: [0.95801586]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288]
Sample: [1.29] expected: [0.96083506]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911]
Sample: [1.31] expected: [0.96618495]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017]
Sample: [1.32] expected: [0.9687151]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525]
Sample: [1.33] expected: [0.97114838]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684]
Sample: [1.35] expected: [0.97572336]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423]
Sample: [1.37] expected: [0.97990806]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993]
Sample: [1.38] expected: [0.98185353]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925]
Sample: [1.39] expected: [0.98370081]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254]
Sample: [1.4] expected: [0.98544973]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826]
Sample: [1.41] expected: [0.9871001]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307]
Sample: [1.42] expected: [0.98865176]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591]
Sample: [1.43] expected: [0.99010456]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204]
Sample: [1.44] expected: [0.99145835]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749]
Sample: [1.45] expected: [0.99271299]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911]
Sample: [1.46] expected: [0.99386836]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989]
Sample: [1.48] expected: [0.99588084]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734]
Sample: [1.49] expected: [0.99673775]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049]
Sample: [1.5] expected: [0.99749499]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324]
Sample: [1.51] expected: [0.99815247]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324, 0.9549053444747452]
Sample: [1.52] expected: [0.99871014]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324, 0.9549053444747452, 0.9556812344972768]
Sample: [1.53] expected: [0.99916795]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324, 0.9549053444747452, 0.9556812344972768, 0.9564370899844842]
Sample: [1.54] expected: [0.99952583]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324, 0.9549053444747452, 0.9556812344972768, 0.9564370899844842, 0.9571735365719446]
Sample: [1.55] expected: [0.99978376]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324, 0.9549053444747452, 0.9556812344972768, 0.9564370899844842, 0.9571735365719446, 0.9578911784375138]
Sample: [1.56] expected: [0.99994172]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324, 0.9549053444747452, 0.9556812344972768, 0.9564370899844842, 0.9571735365719446, 0.9578911784375138, 0.9585905990599585]
Sample: [1.57] expected: [0.99999968]
produced: [0.08411152663924558, 0.08806967710163104, 0.09219483398523386, 0.09649170911808648, 0.10096492181613757, 0.10561897414708103, 0.11045822477923223, 0.11548686149492254, 0.12070887247154041, 0.12612801645841423, 0.13174779200396824, 0.1375714059146262, 0.143601741154425, 0.1498413244217382, 0.1562922936663923, 0.1629563658361865, 0.16983480516576827, 0.17692839234228963, 0.18423739490057636, 0.19176153921496483, 0.20745129895516892, 0.21561343917320225, 0.24133485185034068, 0.2503070743688777, 0.25947023172061084, 0.26881836738787507, 0.27834487160412696, 0.28804249359861667, 0.29790335881993907, 0.3079189911378813, 0.31808033995579665, 0.3283778120970334, 0.33880130826019816, 0.34934026377069877, 0.359983693291648, 0.3815382224577058, 0.3924256976376896, 0.40337050797172563, 0.41436034345020933, 0.42538279923005534, 0.4474758309269454, 0.4585216506364875, 0.4695506922703565, 0.4915106428447374, 0.502418308886826, 0.5132628058578435, 0.5240333758592197, 0.5347196790743675, 0.5453118278741529, 0.5558004166418892, 0.5661765472228567, 0.5764318499630463, 0.5865585003585199, 0.5965492313895817, 0.6063973416620857, 0.6160966995210914, 0.6256417433393, 0.6350274782140207, 0.644249469331738, 0.6533038322787406, 0.6621872205899141, 0.6708968108359907, 0.6794302855526724, 0.687785814313561, 0.6959620332432308, 0.7039580232576077, 0.7117732873066137, 0.7194077268793427, 0.7268616180153803, 0.7341355870477622, 0.7412305862839559, 0.7481478698115784, 0.7548889695956867, 0.7614556720147685, 0.7678499949632664, 0.7740741656298528, 0.7801305990429215, 0.786021877458018, 0.7917507306463153, 0.7973200171288329, 0.8027327063879166, 0.807991862075592, 0.8131006262277498, 0.818062204483678, 0.8228798523022194, 0.8275568621587098, 0.8320965517007859, 0.8365022528360997, 0.8407773017208169, 0.8449250296144681, 0.8489487545641704, 0.8566373573569659, 0.8603087412160482, 0.8638691227010243, 0.8673216553123001, 0.8706694446239658, 0.8739155446488195, 0.8770629547120377, 0.8830734133202875, 0.8859421653218246, 0.8887236310027465, 0.8914205046138892, 0.8940354156444998, 0.8965709282899061, 0.9037257341258768, 0.905967984076397, 0.9081426748386772, 0.9102519801017624, 0.9122980102969892, 0.9142828134696038, 0.9162083762456027, 0.9180766248803507, 0.9216485896611614, 0.9233558668085291, 0.925012954304611, 0.9266214943396998, 0.9281830761245597, 0.929699237222288, 0.9311714648898911, 0.9339898255064017, 0.9353386935433525, 0.9366491009976684, 0.9391595151898423, 0.9415306146357993, 0.9426667295048925, 0.9437713094236254, 0.9448453751746826, 0.9458899126056307, 0.9469058737730591, 0.9478941780594204, 0.9488557132620749, 0.9497913366541911, 0.9507018760172989, 0.952450872320734, 0.9532908462611049, 0.9541087720393324, 0.9549053444747452, 0.9556812344972768, 0.9564370899844842, 0.9571735365719446, 0.9578911784375138, 0.9585905990599585, 0.959272361952509]
RMSE =  0.035595628802967114

Sample Run of run_XOR():
Sample: [0. 1.] expected: [1.]
produced: [0.7274184087931534]
Sample: [0. 0.] expected: [0.]
produced: [0.6252742932352657]
Sample: [1. 0.] expected: [1.]
produced: [0.84430176575002]
Sample: [1. 1.] expected: [0.]
produced: [0.8905655406130873]
Epoch 0 RMSE =  0.5662635402181919
Epoch 100 RMSE =  0.5231740243940889
Epoch 200 RMSE =  0.5124183584857172
Epoch 300 RMSE =  0.5110662881643
Epoch 400 RMSE =  0.5107347385285976
Epoch 500 RMSE =  0.5104658188356271
Epoch 600 RMSE =  0.5101875671885522
Epoch 700 RMSE =  0.5098777731325879
Epoch 800 RMSE =  0.5095535966525869
Epoch 900 RMSE =  0.5091839060152121
Sample: [0. 0.] expected: [0.]
produced: [0.3776070868810177]
Sample: [1. 0.] expected: [1.]
produced: [0.5599355904324891]
Sample: [1. 1.] expected: [0.]
produced: [0.6306397529263786]
Sample: [0. 1.] expected: [1.]
produced: [0.45087206574903427]
Epoch 1000 RMSE =  0.5087955834066731
Epoch 1100 RMSE =  0.5083609381611318
Epoch 1200 RMSE =  0.5078837836924904
Epoch 1300 RMSE =  0.507364454526752
Epoch 1400 RMSE =  0.5067894988569517
Epoch 1500 RMSE =  0.50615744842536
Epoch 1600 RMSE =  0.5054538598225057
Epoch 1700 RMSE =  0.5046815439037688
Epoch 1800 RMSE =  0.503831278241074
Epoch 1900 RMSE =  0.5028875283315443
Sample: [0. 1.] expected: [1.]
produced: [0.5026051162867287]
Sample: [1. 0.] expected: [1.]
produced: [0.5387489713394322]
Sample: [1. 1.] expected: [0.]
produced: [0.6398470954515024]
Sample: [0. 0.] expected: [0.]
produced: [0.37121655254147423]
Epoch 2000 RMSE =  0.5018366806599214
Epoch 2100 RMSE =  0.5006903229505416
Epoch 2200 RMSE =  0.49941761578242044
Epoch 2300 RMSE =  0.49801514296243204
Epoch 2400 RMSE =  0.49648139328243757
Epoch 2500 RMSE =  0.49480854183114786
Epoch 2600 RMSE =  0.4929827212439211
Epoch 2700 RMSE =  0.4910274956919509
Epoch 2800 RMSE =  0.4889185380102787
Epoch 2900 RMSE =  0.48668655758267615
Sample: [0. 0.] expected: [0.]
produced: [0.3536184800310566]
Sample: [0. 1.] expected: [1.]
produced: [0.5829500823888738]
Sample: [1. 1.] expected: [0.]
produced: [0.6210527909273695]
Sample: [1. 0.] expected: [1.]
produced: [0.4964155403161712]
Epoch 3000 RMSE =  0.48432441099759016
Epoch 3100 RMSE =  0.4818330080995766
Epoch 3200 RMSE =  0.47925601754693403
Epoch 3300 RMSE =  0.4765814001001243
Epoch 3400 RMSE =  0.47383191550544
Epoch 3500 RMSE =  0.4710139936333577
Epoch 3600 RMSE =  0.46815162004108785
Epoch 3700 RMSE =  0.4652552236532569
Epoch 3800 RMSE =  0.462320833340822
Epoch 3900 RMSE =  0.45935059933533284
Sample: [1. 1.] expected: [0.]
produced: [0.5716855287966935]
Sample: [1. 0.] expected: [1.]
produced: [0.4602723134148703]
Sample: [0. 1.] expected: [1.]
produced: [0.6730629520724135]
Sample: [0. 0.] expected: [0.]
produced: [0.32861040620053067]
Epoch 4000 RMSE =  0.4563449769289057
Epoch 4100 RMSE =  0.4532926704585072
Epoch 4200 RMSE =  0.4501748264206178
Epoch 4300 RMSE =  0.4469745759554457
Epoch 4400 RMSE =  0.44365793051379804
Epoch 4500 RMSE =  0.44019047411645734
Epoch 4600 RMSE =  0.4365180294683271
Epoch 4700 RMSE =  0.4325861464737913
Epoch 4800 RMSE =  0.42832923440144416
Epoch 4900 RMSE =  0.4236753498122094
Sample: [0. 0.] expected: [0.]
produced: [0.2962248034485183]
Sample: [1. 1.] expected: [0.]
produced: [0.5128354840980076]
Sample: [0. 1.] expected: [1.]
produced: [0.7197266394694871]
Sample: [1. 0.] expected: [1.]
produced: [0.47901611174376735]
Epoch 5000 RMSE =  0.4185471109605825
Epoch 5100 RMSE =  0.41289355492482177
Epoch 5200 RMSE =  0.4066452625948095
Epoch 5300 RMSE =  0.39977574297128166
Epoch 5400 RMSE =  0.3922958761427248
Epoch 5500 RMSE =  0.38423848571704106
Epoch 5600 RMSE =  0.37565576251998456
Epoch 5700 RMSE =  0.36663557848410583
Epoch 5800 RMSE =  0.357282675056383
Epoch 5900 RMSE =  0.3477011538849734
Sample: [1. 1.] expected: [0.]
produced: [0.4086912430746791]
Sample: [0. 0.] expected: [0.]
produced: [0.2404083671439436]
Sample: [1. 0.] expected: [1.]
produced: [0.6092777452889789]
Sample: [0. 1.] expected: [1.]
produced: [0.7180758110929291]
Epoch 6000 RMSE =  0.3379977232823164
Epoch 6100 RMSE =  0.3282838483709489
Epoch 6200 RMSE =  0.318642115831183
Epoch 6300 RMSE =  0.3091605584931896
Epoch 6400 RMSE =  0.2999119321866461
Epoch 6500 RMSE =  0.2909449475100087
Epoch 6600 RMSE =  0.28229955770419607
Epoch 6700 RMSE =  0.27400638224027407
Epoch 6800 RMSE =  0.26607568905642726
Epoch 6900 RMSE =  0.2585141645102997
Sample: [0. 0.] expected: [0.]
produced: [0.19871734817703812]
Sample: [1. 0.] expected: [1.]
produced: [0.7301588351410757]
Sample: [0. 1.] expected: [1.]
produced: [0.761587573992237]
Sample: [1. 1.] expected: [0.]
produced: [0.28897257993693637]
Epoch 7000 RMSE =  0.25132074901868506
Epoch 7100 RMSE =  0.2444855858478172
Epoch 7200 RMSE =  0.23800214969656414
Epoch 7300 RMSE =  0.2318530201115002
Epoch 7400 RMSE =  0.22602642251178046
Epoch 7500 RMSE =  0.22050146012733035
Epoch 7600 RMSE =  0.2152645931196474
Epoch 7700 RMSE =  0.21029794327419096
Epoch 7800 RMSE =  0.20558680847927904
Epoch 7900 RMSE =  0.20111239639764286
Sample: [0. 1.] expected: [1.]
produced: [0.8068950379719829]
Sample: [1. 0.] expected: [1.]
produced: [0.7949461585442956]
Sample: [1. 1.] expected: [0.]
produced: [0.2145836615280919]
Sample: [0. 0.] expected: [0.]
produced: [0.17215361305984173]
Epoch 8000 RMSE =  0.19686265424144284
Epoch 8100 RMSE =  0.19282285104450955
Epoch 8200 RMSE =  0.1889782312636713
Epoch 8300 RMSE =  0.18531689152979686
Epoch 8400 RMSE =  0.18182688994499063
Epoch 8500 RMSE =  0.1784985613997357
Epoch 8600 RMSE =  0.1753197829424382
Epoch 8700 RMSE =  0.1722824106311265
Epoch 8800 RMSE =  0.16937803211033456
Epoch 8900 RMSE =  0.16659745287808778
Sample: [1. 1.] expected: [0.]
produced: [0.16988165689680867]
Sample: [0. 1.] expected: [1.]
produced: [0.837745224027346]
Sample: [0. 0.] expected: [0.]
produced: [0.15428611492711888]
Sample: [1. 0.] expected: [1.]
produced: [0.8311635919780892]
Epoch 9000 RMSE =  0.16393316291173854
Epoch 9100 RMSE =  0.16137842204663455
Epoch 9200 RMSE =  0.15892717108451224
Epoch 9300 RMSE =  0.1565726658805625
Epoch 9400 RMSE =  0.15430990608702302
Epoch 9500 RMSE =  0.15213277424531094
Epoch 9600 RMSE =  0.15003770774872324
Epoch 9700 RMSE =  0.14801905712009716
Epoch 9800 RMSE =  0.14607323694983132
Epoch 9900 RMSE =  0.1441961205161043
Sample: [1. 0.] expected: [1.]
produced: [0.855173372778329]
Sample: [1. 1.] expected: [0.]
produced: [0.14218657969693413]
Sample: [0. 1.] expected: [1.]
produced: [0.8593056600868556]
Sample: [0. 0.] expected: [0.]
produced: [0.1417966043807155]
Epoch 10000 RMSE =  0.14238411928285225
Epoch 10100 RMSE =  0.14063367936679116
Epoch 10200 RMSE =  0.1389420557306878
Epoch 10300 RMSE =  0.13730596691726132
Epoch 10400 RMSE =  0.13572260144576884
Epoch 10500 RMSE =  0.13418955516018533
Epoch 10600 RMSE =  0.13270420684887446
Epoch 10700 RMSE =  0.13126430632611621
Epoch 10800 RMSE =  0.12986797923434873
Epoch 10900 RMSE =  0.12851295572099808
Sample: [0. 0.] expected: [0.]
produced: [0.13211126931866418]
Sample: [1. 0.] expected: [1.]
produced: [0.8715648109084146]
Sample: [0. 1.] expected: [1.]
produced: [0.8746859101045187]
Sample: [1. 1.] expected: [0.]
produced: [0.12273585964319238]
Epoch 11000 RMSE =  0.12719738366675049
Epoch 11100 RMSE =  0.12591949680227396
Epoch 11200 RMSE =  0.12467768210861743
Epoch 11300 RMSE =  0.12347030930197032
Epoch 11400 RMSE =  0.12229591799759774
Epoch 11500 RMSE =  0.12115303275487753
Epoch 11600 RMSE =  0.12004059797239446
Epoch 11700 RMSE =  0.11895706720968055
Epoch 11800 RMSE =  0.11790135337842483
Epoch 11900 RMSE =  0.11687242520931439
Sample: [1. 1.] expected: [0.]
produced: [0.10823729220188434]
Sample: [0. 1.] expected: [1.]
produced: [0.8860596450568784]
Sample: [0. 0.] expected: [0.]
produced: [0.12443776633834945]
Sample: [1. 0.] expected: [1.]
produced: [0.8837240587812696]
Epoch 12000 RMSE =  0.11586907277757505
Epoch 12100 RMSE =  0.11489036205346169
Epoch 12200 RMSE =  0.11393537374010888
Epoch 12300 RMSE =  0.1130031767610133
Epoch 12400 RMSE =  0.11209293019174532
Epoch 12500 RMSE =  0.11120381978194922
Epoch 12600 RMSE =  0.11033498307323956
Epoch 12700 RMSE =  0.10948586398491607
Epoch 12800 RMSE =  0.10865561899955747
Epoch 12900 RMSE =  0.10784371576844541
Sample: [0. 1.] expected: [1.]
produced: [0.895149550588691]
Sample: [1. 1.] expected: [0.]
produced: [0.0973431259939637]
Sample: [0. 0.] expected: [0.]
produced: [0.11815458926410455]
Sample: [1. 0.] expected: [1.]
produced: [0.8931897108363701]
Epoch 13000 RMSE =  0.10704934114161756
Epoch 13100 RMSE =  0.10627196995125587
Epoch 13200 RMSE =  0.10551104171732409
Epoch 13300 RMSE =  0.10476597879094204
Epoch 13400 RMSE =  0.10403624153020176
Epoch 13500 RMSE =  0.10332140049678927
Epoch 13600 RMSE =  0.10262092497322418
Epoch 13700 RMSE =  0.10193433813297416
Epoch 13800 RMSE =  0.10126121964470383
Epoch 13900 RMSE =  0.10060110430544644
Sample: [1. 1.] expected: [0.]
produced: [0.0885202804816245]
Sample: [0. 1.] expected: [1.]
produced: [0.9023167964990566]
Sample: [1. 0.] expected: [1.]
produced: [0.9008095312621544]
Sample: [0. 0.] expected: [0.]
produced: [0.11289981467824581]
Epoch 14000 RMSE =  0.09995369621175594
Epoch 14100 RMSE =  0.09931851662323395
Epoch 14200 RMSE =  0.09869518977667364
Epoch 14300 RMSE =  0.09808339175303639
Epoch 14400 RMSE =  0.09748280850803345
Epoch 14500 RMSE =  0.09689304677669432
Epoch 14600 RMSE =  0.09631381744911806
Epoch 14700 RMSE =  0.09574486974996248
Epoch 14800 RMSE =  0.09518584502372722
Epoch 14900 RMSE =  0.09463645493493353
Sample: [0. 0.] expected: [0.]
produced: [0.10830612122913745]
Sample: [1. 1.] expected: [0.]
produced: [0.0813449209761838]
Sample: [0. 1.] expected: [1.]
produced: [0.9082611401529437]
Sample: [1. 0.] expected: [1.]
produced: [0.9069765097204351]
Epoch 15000 RMSE =  0.09409649331158994
Epoch 15100 RMSE =  0.0935656559073716
Epoch 15200 RMSE =  0.09304371991103352
Epoch 15300 RMSE =  0.09253044994854781
Epoch 15400 RMSE =  0.09202556805232183
Epoch 15500 RMSE =  0.09152887083070238
Epoch 15600 RMSE =  0.09104016464663182
Epoch 15700 RMSE =  0.09055923447287413
Epoch 15800 RMSE =  0.09008586787183061
Epoch 15900 RMSE =  0.0896198807568987
Sample: [1. 1.] expected: [0.]
produced: [0.07546770635796438]
Sample: [1. 0.] expected: [1.]
produced: [0.9121993134014049]
Sample: [0. 1.] expected: [1.]
produced: [0.9133728428319097]
Sample: [0. 0.] expected: [0.]
produced: [0.10435612206507823]
Epoch 16000 RMSE =  0.08916108994623606
Epoch 16100 RMSE =  0.0887093131327808
Epoch 16200 RMSE =  0.08826435361584697
Epoch 16300 RMSE =  0.08782610598668578
Epoch 16400 RMSE =  0.08739434275343108
Epoch 16500 RMSE =  0.08696894094270034
Epoch 16600 RMSE =  0.08654973088723195
Epoch 16700 RMSE =  0.08613657548702876
Epoch 16800 RMSE =  0.08572933052108145
Epoch 16900 RMSE =  0.08532786734819832
Sample: [1. 0.] expected: [1.]
produced: [0.916689839956371]
Sample: [0. 0.] expected: [0.]
produced: [0.10084006495329084]
Sample: [0. 1.] expected: [1.]
produced: [0.9176846726293291]
Sample: [1. 1.] expected: [0.]
produced: [0.07048892378674271]
Epoch 17000 RMSE =  0.08493203600973095
Epoch 17100 RMSE =  0.08454172507928996
Epoch 17200 RMSE =  0.0841567795582271
Epoch 17300 RMSE =  0.08377712570099921
Epoch 17400 RMSE =  0.0834026169580528
Epoch 17500 RMSE =  0.08303314326210873
Epoch 17600 RMSE =  0.0826685861655246
Epoch 17700 RMSE =  0.08230883156547708
Epoch 17800 RMSE =  0.0819538087894816
Epoch 17900 RMSE =  0.08160341196996322
Sample: [1. 0.] expected: [1.]
produced: [0.9205368536391269]
Sample: [1. 1.] expected: [0.]
produced: [0.06616912474102923]
Sample: [0. 1.] expected: [1.]
produced: [0.921424609532093]
Sample: [0. 0.] expected: [0.]
produced: [0.09769491354120222]
Epoch 18000 RMSE =  0.08125751167979864
Epoch 18100 RMSE =  0.08091603625554003
Epoch 18200 RMSE =  0.08057888600389561
Epoch 18300 RMSE =  0.08024596044840855
Epoch 18400 RMSE =  0.07991720384224019
Epoch 18500 RMSE =  0.07959251140751203
Epoch 18600 RMSE =  0.0792717949405661
Epoch 18700 RMSE =  0.0789549835684302
Epoch 18800 RMSE =  0.07864200321614226
Epoch 18900 RMSE =  0.0783327671067579
Sample: [1. 0.] expected: [1.]
produced: [0.9239048335359885]
Sample: [0. 0.] expected: [0.]
produced: [0.09485997236169166]
Sample: [0. 1.] expected: [1.]
produced: [0.9247026440773083]
Sample: [1. 1.] expected: [0.]
produced: [0.06240514137782196]
Epoch 19000 RMSE =  0.07802721030989924
Epoch 19100 RMSE =  0.07772526160561365
Epoch 19200 RMSE =  0.07742683386709147
Epoch 19300 RMSE =  0.07713189458036826
Epoch 19400 RMSE =  0.07684034535823833
Epoch 19500 RMSE =  0.07655213175867774
Epoch 19600 RMSE =  0.07626719556194961
Epoch 19700 RMSE =  0.07598546200434138
Epoch 19800 RMSE =  0.07570689248708408
Epoch 19900 RMSE =  0.07543142494625849
Sample: [1. 1.] expected: [0.]
produced: [0.059076972412370574]
Sample: [0. 1.] expected: [1.]
produced: [0.9275967750363233]
Sample: [0. 0.] expected: [0.]
produced: [0.09226509302400676]
Sample: [1. 0.] expected: [1.]
produced: [0.9268540371218842]
Epoch 20000 RMSE =  0.07515898969958298
Final Epoch RMSE =  0.07515898969958298
Sample: [0. 0.] expected: [0.]
produced: [0.09226208583519398]
Sample: [1. 0.] expected: [1.]
produced: [0.09226208583519398, 0.926889619404805]
Sample: [0. 1.] expected: [1.]
produced: [0.09226208583519398, 0.926889619404805, 0.927622084663257]
Sample: [1. 1.] expected: [0.]
produced: [0.09226208583519398, 0.926889619404805, 0.927622084663257, 0.059073910080746]
RMSE =  0.07514271374217825


"""