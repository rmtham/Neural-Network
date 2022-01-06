""" This program trains and tests a neural network. """
from collections import deque
import random
from enum import Enum
import numpy as np
import math
from abc import ABC, abstractmethod
import json


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


def main():

    class MultiTypeEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, NNData):
                return {"__NNData__": o.__dict__}
            elif isinstance(o, np.ndarray):
                return {"__NDarray__": o.tolist()}
            elif isinstance(o, deque):
                return {"__deque__": list(o)}
            else:
                super().default(o)

    def multi_type_decoder(o):
        if "__NNData__" in o:
            item = o["__NNData__"]
            ret_obj = NNData()
            ret_obj._features = item["_features"]
            ret_obj._labels = item["_labels"]
            ret_obj._train_indices = item["_train_indices"]
            ret_obj._test_indices = item["_test_indices"]
            ret_obj._train_pool = item["_train_pool"]
            ret_obj._test_pool = item["_test_pool"]
            ret_obj._train_factor = item["_train_factor"]
            return ret_obj
        elif "__NDarray__" in o:
            return np.array(o["__NDarray__"])
        elif "__deque__" in o:
            return deque(o["__deque__"])
        else:
            return o

    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    xor_data = NNData(features, labels, 1)

    xor_data_encoded = json.dumps(xor_data, cls=MultiTypeEncoder)
    xor_data_decoded = json.loads(xor_data_encoded,
                                  object_hook=multi_type_decoder)

    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    network.train(xor_data_decoded, 10001, order=NNData.Order.RANDOM)

    with open("sin_data.txt", "r") as f:
        sin_decoded = json.load(f, object_hook=multi_type_decoder)

    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    network.train(sin_decoded, 10001, order=NNData.Order.RANDOM)


if __name__ == "__main__":
    main()

"""
Sample: [0. 0.] expected: [0.]
produced: [0.5811074419752063]
Sample: [1. 0.] expected: [1.]
produced: [0.7947060837444648]
Sample: [1. 1.] expected: [0.]
produced: [0.8986035037409195]
Sample: [0. 1.] expected: [1.]
produced: [0.7590301028071453]
Epoch 0 RMSE =  0.5579843634590251
Epoch 100 RMSE =  0.5278451412321308
Epoch 200 RMSE =  0.5188813861850804
Epoch 300 RMSE =  0.5170457916687692
Epoch 400 RMSE =  0.5166970017507122
Epoch 500 RMSE =  0.516615557650086
Epoch 600 RMSE =  0.5165804533874662
Epoch 700 RMSE =  0.5165586448880325
Epoch 800 RMSE =  0.5165380790676007
Epoch 900 RMSE =  0.5165163202433963
Sample: [0. 0.] expected: [0.]
produced: [0.3492883748402068]
Sample: [1. 1.] expected: [0.]
produced: [0.6986467288232285]
Sample: [1. 0.] expected: [1.]
produced: [0.562989433576984]
Sample: [0. 1.] expected: [1.]
produced: [0.48426252547064214]
Epoch 1000 RMSE =  0.5164961273555678
Epoch 1100 RMSE =  0.5164746530913479
Epoch 1200 RMSE =  0.5164580663712314
Epoch 1300 RMSE =  0.5164385098181158
Epoch 1400 RMSE =  0.5164223509295208
Epoch 1500 RMSE =  0.5164042643957992
Epoch 1600 RMSE =  0.5163841989250081
Epoch 1700 RMSE =  0.5163695553091323
Epoch 1800 RMSE =  0.516347510707992
Epoch 1900 RMSE =  0.5163327871526879
Sample: [1. 0.] expected: [1.]
produced: [0.5625309045885778]
Sample: [1. 1.] expected: [0.]
produced: [0.6987247744791383]
Sample: [0. 1.] expected: [1.]
produced: [0.48641863671104213]
Sample: [0. 0.] expected: [0.]
produced: [0.3506683029444659]
Epoch 2000 RMSE =  0.5163161810651267
Epoch 2100 RMSE =  0.51629395217778
Epoch 2200 RMSE =  0.5162806455143683
Epoch 2300 RMSE =  0.5162606416115262
Epoch 2400 RMSE =  0.5162388520531407
Epoch 2500 RMSE =  0.5162197598072069
Epoch 2600 RMSE =  0.5161985808743599
Epoch 2700 RMSE =  0.5161738169773442
Epoch 2800 RMSE =  0.5161569826800926
Epoch 2900 RMSE =  0.5161280755466586
Sample: [0. 1.] expected: [1.]
produced: [0.48929390430749864]
Sample: [1. 0.] expected: [1.]
produced: [0.5602747147754241]
Sample: [1. 1.] expected: [0.]
produced: [0.6987926360862027]
Sample: [0. 0.] expected: [0.]
produced: [0.35066860716832693]
Epoch 3000 RMSE =  0.5161052855147367
Epoch 3100 RMSE =  0.5160816802321713
Epoch 3200 RMSE =  0.5160579223603482
Epoch 3300 RMSE =  0.5160289242105212
Epoch 3400 RMSE =  0.5159989966275219
Epoch 3500 RMSE =  0.5159708601887961
Epoch 3600 RMSE =  0.5159392034480346
Epoch 3700 RMSE =  0.5159008066323473
Epoch 3800 RMSE =  0.5158683093558865
Epoch 3900 RMSE =  0.5158316562360397
Sample: [0. 1.] expected: [1.]
produced: [0.4925290132342772]
Sample: [1. 1.] expected: [0.]
produced: [0.6958110716636366]
Sample: [1. 0.] expected: [1.]
produced: [0.5533404899638125]
Sample: [0. 0.] expected: [0.]
produced: [0.350698778515897]
Epoch 4000 RMSE =  0.5157940967617273
Epoch 4100 RMSE =  0.5157525700255341
Epoch 4200 RMSE =  0.5157110349953514
Epoch 4300 RMSE =  0.5156662452216334
Epoch 4400 RMSE =  0.5156196155762821
Epoch 4500 RMSE =  0.5155666467269585
Epoch 4600 RMSE =  0.5155124297872018
Epoch 4700 RMSE =  0.5154579411387425
Epoch 4800 RMSE =  0.5154000822192117
Epoch 4900 RMSE =  0.5153362343542987
Sample: [0. 0.] expected: [0.]
produced: [0.3495507183250922]
Sample: [1. 1.] expected: [0.]
produced: [0.6903728023834467]
Sample: [1. 0.] expected: [1.]
produced: [0.5440997356905933]
Sample: [0. 1.] expected: [1.]
produced: [0.49465443825584315]
Epoch 5000 RMSE =  0.5152716513462258
Epoch 5100 RMSE =  0.5151988454222548
Epoch 5200 RMSE =  0.5151249412583122
Epoch 5300 RMSE =  0.5150455085001263
Epoch 5400 RMSE =  0.5149623470428822
Epoch 5500 RMSE =  0.5148726684677187
Epoch 5600 RMSE =  0.5147737790251792
Epoch 5700 RMSE =  0.5146723606919099
Epoch 5800 RMSE =  0.5145653667819087
Epoch 5900 RMSE =  0.5144494274270611
Sample: [0. 0.] expected: [0.]
produced: [0.348982200686335]
Sample: [0. 1.] expected: [1.]
produced: [0.5014356131961555]
Sample: [1. 0.] expected: [1.]
produced: [0.538970224068761]
Sample: [1. 1.] expected: [0.]
produced: [0.6893576638147172]
Epoch 6000 RMSE =  0.5143241845228221
Epoch 6100 RMSE =  0.5141914243698722
Epoch 6200 RMSE =  0.5140462240622267
Epoch 6300 RMSE =  0.5138988424892781
Epoch 6400 RMSE =  0.5137338053929196
Epoch 6500 RMSE =  0.5135592558776727
Epoch 6600 RMSE =  0.5133716870855588
Epoch 6700 RMSE =  0.513163236132632
Epoch 6800 RMSE =  0.5129469353111404
Epoch 6900 RMSE =  0.5127124186428663
Sample: [0. 0.] expected: [0.]
produced: [0.3483083403029601]
Sample: [1. 0.] expected: [1.]
produced: [0.5234160588884144]
Sample: [1. 1.] expected: [0.]
produced: [0.6781520049149075]
Sample: [0. 1.] expected: [1.]
produced: [0.5079573653532492]
Epoch 7000 RMSE =  0.5124565954620641
Epoch 7100 RMSE =  0.5121798006444637
Epoch 7200 RMSE =  0.5118783066159602
Epoch 7300 RMSE =  0.5115527273987376
Epoch 7400 RMSE =  0.5111999854888781
Epoch 7500 RMSE =  0.5108126172563935
Epoch 7600 RMSE =  0.5103949266441937
Epoch 7700 RMSE =  0.5099336261760431
Epoch 7800 RMSE =  0.5094365621280796
Epoch 7900 RMSE =  0.5088866620961982
Sample: [1. 0.] expected: [1.]
produced: [0.5043699734969221]
Sample: [0. 0.] expected: [0.]
produced: [0.3496017119490658]
Sample: [0. 1.] expected: [1.]
produced: [0.5186280269941564]
Sample: [1. 1.] expected: [0.]
produced: [0.6586669275752723]
Epoch 8000 RMSE =  0.5082891839411269
Epoch 8100 RMSE =  0.5076333577920914
Epoch 8200 RMSE =  0.5069159209309849
Epoch 8300 RMSE =  0.5061289455119433
Epoch 8400 RMSE =  0.5052649232236692
Epoch 8500 RMSE =  0.5043189514427928
Epoch 8600 RMSE =  0.5032834352619862
Epoch 8700 RMSE =  0.502156529238984
Epoch 8800 RMSE =  0.500928898503232
Epoch 8900 RMSE =  0.4995962421026829
Sample: [1. 0.] expected: [1.]
produced: [0.4813587547229561]
Sample: [0. 0.] expected: [0.]
produced: [0.3556855443430094]
Sample: [1. 1.] expected: [0.]
produced: [0.6071748408866736]
Sample: [0. 1.] expected: [1.]
produced: [0.5219804046622913]
Epoch 9000 RMSE =  0.49816286711518604
Epoch 9100 RMSE =  0.4966200154785286
Epoch 9200 RMSE =  0.494978899769902
Epoch 9300 RMSE =  0.4932366447932225
Epoch 9400 RMSE =  0.49140138829021657
Epoch 9500 RMSE =  0.4894812368940013
Epoch 9600 RMSE =  0.48747591367439047
Epoch 9700 RMSE =  0.4853962867268094
Epoch 9800 RMSE =  0.4832477375049893
Epoch 9900 RMSE =  0.4810251464101365
Sample: [0. 0.] expected: [0.]
produced: [0.3758501586402267]
Sample: [1. 1.] expected: [0.]
produced: [0.5236370110480194]
Sample: [0. 1.] expected: [1.]
produced: [0.5237753447580809]
Sample: [1. 0.] expected: [1.]
produced: [0.47606399988935344]
Epoch 10000 RMSE =  0.47873737987580817
Final Epoch RMSE =  0.47873737987580817
Sample: [0.01] expected: [0.00999983]
produced: [0.6593584725397581]
Sample: [0.49] expected: [0.47062589]
produced: [0.7402974039818679]
Sample: [0.66] expected: [0.61311685]
produced: [0.7654616415201219]
Sample: [0.87] expected: [0.76432894]
produced: [0.7942304675386062]
Sample: [0.41] expected: [0.39860933]
produced: [0.726715252329388]
Sample: [0.8] expected: [0.71735609]
produced: [0.7842323534625045]
Sample: [1.46] expected: [0.99386836]
produced: [0.8597198383360009]
Sample: [0.08] expected: [0.07991469]
produced: [0.6695117317708736]
Sample: [0.56] expected: [0.5311862]
produced: [0.7486022773260517]
Sample: [1.1] expected: [0.89120736]
produced: [0.8207720933309534]
Sample: [1.45] expected: [0.99271299]
produced: [0.8578392767948089]
Sample: [0.83] expected: [0.73793137]
produced: [0.7872174146412898]
Sample: [0.24] expected: [0.23770263]
produced: [0.6964381541372375]
Sample: [0.61] expected: [0.57286746]
produced: [0.7550860529528468]
Sample: [1.48] expected: [0.99588084]
produced: [0.8598052904492165]
Sample: [0.48] expected: [0.46177918]
produced: [0.7350026994501797]
Sample: [0.9] expected: [0.78332691]
produced: [0.7949260649239662]
Sample: [0.15] expected: [0.14943813]
produced: [0.6791271560406498]
Sample: [0.39] expected: [0.38018842]
produced: [0.7189777979598274]
Sample: [0.82] expected: [0.73114583]
produced: [0.7826924414557797]
Sample: [0.69] expected: [0.63653718]
produced: [0.7644032719038891]
Sample: [0.21] expected: [0.2084599]
produced: [0.6876106497569123]
Sample: [0.53] expected: [0.50553334]
produced: [0.7393099421195709]
Sample: [0.97] expected: [0.82488571]
produced: [0.8007085893243903]
Sample: [0.47] expected: [0.45288629]
produced: [0.7294983421894515]
Sample: [0.44] expected: [0.42593947]
produced: [0.7241697485250833]
Sample: [0.17] expected: [0.16918235]
produced: [0.6781203511633116]
Sample: [0.54] expected: [0.51413599]
produced: [0.7383226746365782]
Sample: [1.36] expected: [0.9778646]
produced: [0.843025841291482]
Sample: [1.2] expected: [0.93203909]
produced: [0.8260446509271178]
Sample: [0.34] expected: [0.33348709]
produced: [0.7061755181506957]
Epoch 0 RMSE =  0.29774728783352267
Epoch 100 RMSE =  0.22288824758737621
Epoch 200 RMSE =  0.2189676789050478
Epoch 300 RMSE =  0.21077135364871807
Epoch 400 RMSE =  0.19102842668555217
Epoch 500 RMSE =  0.16134388416610426
Epoch 600 RMSE =  0.13298407642801544
Epoch 700 RMSE =  0.1100628885114791
Epoch 800 RMSE =  0.09207382143950832
Epoch 900 RMSE =  0.07799568391840898
Sample: [0.39] expected: [0.38018842]
produced: [0.4121993171679851]
Sample: [0.61] expected: [0.57286746]
produced: [0.5770148585938439]
Sample: [0.17] expected: [0.16918235]
produced: [0.2487547025446703]
Sample: [0.44] expected: [0.42593947]
produced: [0.45112113261625264]
Sample: [0.49] expected: [0.47062589]
produced: [0.4896299418202804]
Sample: [0.83] expected: [0.73793137]
produced: [0.7054857503874163]
Sample: [0.01] expected: [0.00999983]
produced: [0.15690416581105993]
Sample: [1.45] expected: [0.99271299]
produced: [0.8797928962646013]
Sample: [1.36] expected: [0.9778646]
produced: [0.8656067156018712]
Sample: [0.56] expected: [0.5311862]
produced: [0.5418195089557408]
Sample: [0.69] expected: [0.63653718]
produced: [0.6288579579035058]
Sample: [0.82] expected: [0.73114583]
produced: [0.7008143924755377]
Sample: [0.15] expected: [0.14943813]
produced: [0.23565788114012323]
Sample: [0.47] expected: [0.45288629]
produced: [0.47438052994011354]
Sample: [0.24] expected: [0.23770263]
produced: [0.29721939120855473]
Sample: [1.48] expected: [0.99588084]
produced: [0.8841030438412881]
Sample: [0.54] expected: [0.51413599]
produced: [0.5271219046572164]
Sample: [1.1] expected: [0.89120736]
produced: [0.8076077697353645]
Sample: [0.53] expected: [0.50553334]
produced: [0.5198698941428518]
Sample: [0.8] expected: [0.71735609]
produced: [0.6907470616508496]
Sample: [1.46] expected: [0.99386836]
produced: [0.8814326593549782]
Sample: [0.41] expected: [0.39860933]
produced: [0.42786936696026184]
Sample: [0.66] expected: [0.61311685]
produced: [0.6101805579834122]
Sample: [0.97] expected: [0.82488571]
produced: [0.7654281860309176]
Sample: [0.87] expected: [0.76432894]
produced: [0.7246409632679262]
Sample: [0.08] expected: [0.07991469]
produced: [0.19350606365826134]
Sample: [0.9] expected: [0.78332691]
produced: [0.7377148832170066]
Sample: [0.34] expected: [0.33348709]
produced: [0.37309363843328386]
Sample: [1.2] expected: [0.93203909]
produced: [0.8337247441327886]
Sample: [0.21] expected: [0.2084599]
produced: [0.27609834691488094]
Sample: [0.48] expected: [0.46177918]
produced: [0.48234091979947735]
Epoch 1000 RMSE =  0.06702991589854342
Epoch 1100 RMSE =  0.0585353924729236
Epoch 1200 RMSE =  0.051987880666224964
Epoch 1300 RMSE =  0.04695554820462298
Epoch 1400 RMSE =  0.04309367111376517
Epoch 1500 RMSE =  0.040130966486969954
Epoch 1600 RMSE =  0.037855590102747944
Epoch 1700 RMSE =  0.0361043586836479
Epoch 1800 RMSE =  0.03475118356785193
Epoch 1900 RMSE =  0.033700027114861135
Sample: [1.36] expected: [0.9778646]
produced: [0.9105430422772619]
Sample: [0.9] expected: [0.78332691]
produced: [0.7848626879692819]
Sample: [0.49] expected: [0.47062589]
produced: [0.473395846643264]
Sample: [0.61] expected: [0.57286746]
produced: [0.5884598596211855]
Sample: [0.87] expected: [0.76432894]
produced: [0.7700947047146713]
Sample: [1.46] expected: [0.99386836]
produced: [0.9233174217006299]
Sample: [1.48] expected: [0.99588084]
produced: [0.9255589373539787]
Sample: [0.54] expected: [0.51413599]
produced: [0.5231065808712556]
Sample: [0.53] expected: [0.50553334]
produced: [0.5133022463462116]
Sample: [0.48] expected: [0.46177918]
produced: [0.46326492990802226]
Sample: [0.44] expected: [0.42593947]
produced: [0.4224238076630432]
Sample: [0.24] expected: [0.23770263]
produced: [0.23076092972151827]
Sample: [0.39] expected: [0.38018842]
produced: [0.3714193008306292]
Sample: [0.8] expected: [0.71735609]
produced: [0.7311487119358004]
Sample: [1.2] expected: [0.93203909]
produced: [0.8824859684189132]
Sample: [0.21] expected: [0.2084599]
produced: [0.20668118169114755]
Sample: [0.83] expected: [0.73793137]
produced: [0.7487382519006454]
Sample: [0.47] expected: [0.45288629]
produced: [0.4531451725946288]
Sample: [0.34] expected: [0.33348709]
produced: [0.32167443520429306]
Sample: [1.45] expected: [0.99271299]
produced: [0.9222170418304604]
Sample: [0.82] expected: [0.73114583]
produced: [0.743082682449992]
Sample: [0.69] expected: [0.63653718]
produced: [0.6553133548651416]
Sample: [0.56] expected: [0.5311862]
produced: [0.5423657443704853]
Sample: [0.15] expected: [0.14943813]
produced: [0.16355171571965527]
Sample: [0.01] expected: [0.00999983]
produced: [0.08984693928594094]
Sample: [0.97] expected: [0.82488571]
produced: [0.8150027910595886]
Sample: [0.66] expected: [0.61311685]
produced: [0.6312541820326643]
Sample: [0.17] expected: [0.16918235]
produced: [0.17707610744119076]
Sample: [0.08] expected: [0.07991469]
produced: [0.12211448550525072]
Sample: [1.1] expected: [0.89120736]
produced: [0.8580462117588581]
Sample: [0.41] expected: [0.39860933]
produced: [0.3916447546249297]
Epoch 2000 RMSE =  0.03287776272732267
Epoch 2100 RMSE =  0.03222914344657969
Epoch 2200 RMSE =  0.03171227246643713
Epoch 2300 RMSE =  0.031295685641919126
Epoch 2400 RMSE =  0.030955635909250778
Epoch 2500 RMSE =  0.030674103070798573
Epoch 2600 RMSE =  0.03043766417797646
Epoch 2700 RMSE =  0.030236251717121663
Epoch 2800 RMSE =  0.030061907626405326
Epoch 2900 RMSE =  0.02990896896652442
Sample: [1.46] expected: [0.99386836]
produced: [0.9326657223916053]
Sample: [0.61] expected: [0.57286746]
produced: [0.589493389155111]
Sample: [0.82] expected: [0.73114583]
produced: [0.7514144074203682]
Sample: [0.41] expected: [0.39860933]
produced: [0.3834110546493313]
Sample: [0.47] expected: [0.45288629]
produced: [0.447305857131169]
Sample: [0.9] expected: [0.78332691]
produced: [0.7947506952787745]
Sample: [0.39] expected: [0.38018842]
produced: [0.36240842544418694]
Sample: [1.48] expected: [0.99588084]
produced: [0.934764541098125]
Sample: [0.49] expected: [0.47062589]
produced: [0.4686174571836563]
Sample: [0.15] expected: [0.14943813]
produced: [0.15264727985803483]
Sample: [0.21] expected: [0.2084599]
produced: [0.1951899357126948]
Sample: [0.56] expected: [0.5311862]
produced: [0.5409755768051577]
Sample: [0.83] expected: [0.73793137]
produced: [0.7574017647351424]
Sample: [0.24] expected: [0.23770263]
produced: [0.21922078072801957]
Sample: [0.97] expected: [0.82488571]
produced: [0.8256664440889583]
Sample: [0.34] expected: [0.33348709]
produced: [0.3112949046238383]
Sample: [0.69] expected: [0.63653718]
produced: [0.6597498849310723]
Sample: [0.48] expected: [0.46177918]
produced: [0.45798299006573556]
Sample: [1.36] expected: [0.9778646]
produced: [0.920536742054642]
Sample: [0.53] expected: [0.50553334]
produced: [0.5104648887201231]
Sample: [0.8] expected: [0.71735609]
produced: [0.7391592547215595]
Sample: [0.87] expected: [0.76432894]
produced: [0.7795898057382157]
Sample: [0.66] expected: [0.61311685]
produced: [0.6344885102087968]
Sample: [1.1] expected: [0.89120736]
produced: [0.8690653162390474]
Sample: [1.45] expected: [0.99271299]
produced: [0.9315706561643688]
Sample: [0.01] expected: [0.00999983]
produced: [0.08154106170922358]
Sample: [0.17] expected: [0.16918235]
produced: [0.16593757506718543]
Sample: [1.2] expected: [0.93203909]
produced: [0.8932444616853922]
Sample: [0.08] expected: [0.07991469]
produced: [0.1124093538210095]
Sample: [0.44] expected: [0.42593947]
produced: [0.41532612661728446]
Sample: [0.54] expected: [0.51413599]
produced: [0.5206476272124945]
Epoch 3000 RMSE =  0.029772991391137905
Epoch 3100 RMSE =  0.029650504649934212
Epoch 3200 RMSE =  0.029538696946996363
Epoch 3300 RMSE =  0.029436313891579944
Epoch 3400 RMSE =  0.02934106876810191
Epoch 3500 RMSE =  0.029252130447826943
Epoch 3600 RMSE =  0.029168432678327923
Epoch 3700 RMSE =  0.02908943221938349
Epoch 3800 RMSE =  0.02901437900532786
Epoch 3900 RMSE =  0.028942631867972107
Sample: [1.48] expected: [0.99588084]
produced: [0.9389111895384658]
Sample: [0.17] expected: [0.16918235]
produced: [0.16483096391850768]
Sample: [0.44] expected: [0.42593947]
produced: [0.41303508028784436]
Sample: [0.53] expected: [0.50553334]
produced: [0.5086845896834121]
Sample: [0.97] expected: [0.82488571]
produced: [0.8291538419657403]
Sample: [0.34] expected: [0.33348709]
produced: [0.3090171083400992]
Sample: [0.15] expected: [0.14943813]
produced: [0.15167523041871145]
Sample: [0.87] expected: [0.76432894]
produced: [0.7822869588232692]
Sample: [0.39] expected: [0.38018842]
produced: [0.3600940634213772]
Sample: [0.8] expected: [0.71735609]
produced: [0.7410242885895038]
Sample: [0.41] expected: [0.39860933]
produced: [0.381128662776822]
Sample: [0.69] expected: [0.63653718]
produced: [0.6600750209139722]
Sample: [1.1] expected: [0.89120736]
produced: [0.8732276722538732]
Sample: [0.9] expected: [0.78332691]
produced: [0.7977101805792172]
Sample: [0.66] expected: [0.61311685]
produced: [0.6344172016028815]
Sample: [1.2] expected: [0.93203909]
produced: [0.8975433458325278]
Sample: [1.46] expected: [0.99386836]
produced: [0.9368544970463047]
Sample: [1.36] expected: [0.9778646]
produced: [0.9248465309901857]
Sample: [0.49] expected: [0.47062589]
produced: [0.46655942501769126]
Sample: [0.56] expected: [0.5311862]
produced: [0.5395554418350045]
Sample: [0.47] expected: [0.45288629]
produced: [0.4451975766292151]
Sample: [0.01] expected: [0.00999983]
produced: [0.0813458368294759]
Sample: [1.45] expected: [0.99271299]
produced: [0.9358040735268892]
Sample: [0.61] expected: [0.57286746]
produced: [0.5887472742096596]
Sample: [0.54] expected: [0.51413599]
produced: [0.5190373658981442]
Sample: [0.48] expected: [0.46177918]
produced: [0.45583896060617957]
Sample: [0.82] expected: [0.73114583]
produced: [0.7535456969916428]
Sample: [0.08] expected: [0.07991469]
produced: [0.11186832389664801]
Sample: [0.83] expected: [0.73793137]
produced: [0.759521562058482]
Sample: [0.24] expected: [0.23770263]
produced: [0.21748927087263586]
Sample: [0.21] expected: [0.2084599]
produced: [0.19368676726538572]
Epoch 4000 RMSE =  0.02887464524738975
Epoch 4100 RMSE =  0.028809331747245853
Epoch 4200 RMSE =  0.02874667400147936
Epoch 4300 RMSE =  0.02868640218490066
Epoch 4400 RMSE =  0.028628451553362442
Epoch 4500 RMSE =  0.028572569644618035
Epoch 4600 RMSE =  0.028518799272450334
Epoch 4700 RMSE =  0.028466813014096877
Epoch 4800 RMSE =  0.02841672564458689
Epoch 4900 RMSE =  0.028368344010263877
Sample: [0.44] expected: [0.42593947]
produced: [0.41194758973350315]
Sample: [0.69] expected: [0.63653718]
produced: [0.6594717224225274]
Sample: [0.01] expected: [0.00999983]
produced: [0.08255323478089463]
Sample: [0.48] expected: [0.46177918]
produced: [0.4545210930999578]
Sample: [0.54] expected: [0.51413599]
produced: [0.5176815096594117]
Sample: [1.45] expected: [0.99271299]
produced: [0.9383977005388704]
Sample: [0.47] expected: [0.45288629]
produced: [0.44391700545918666]
Sample: [0.97] expected: [0.82488571]
produced: [0.8307064703700744]
Sample: [1.36] expected: [0.9778646]
produced: [0.9274577530913911]
Sample: [0.66] expected: [0.61311685]
produced: [0.6336874356047375]
Sample: [0.21] expected: [0.2084599]
produced: [0.19430349790196835]
Sample: [1.46] expected: [0.99386836]
produced: [0.9394813978692832]
Sample: [0.41] expected: [0.39860933]
produced: [0.3802064495580882]
Sample: [0.15] expected: [0.14943813]
produced: [0.15260712284052666]
Sample: [0.8] expected: [0.71735609]
produced: [0.7413514266101402]
Sample: [0.56] expected: [0.5311862]
produced: [0.5382538990500066]
Sample: [0.82] expected: [0.73114583]
produced: [0.7540052205487233]
Sample: [1.48] expected: [0.99588084]
produced: [0.9415218297341074]
Sample: [0.39] expected: [0.38018842]
produced: [0.35923894889282126]
Sample: [1.1] expected: [0.89120736]
produced: [0.8754214648163929]
Sample: [0.17] expected: [0.16918235]
produced: [0.16569479378122173]
Sample: [0.34] expected: [0.33348709]
produced: [0.3086287290385771]
Sample: [0.53] expected: [0.50553334]
produced: [0.5074993731468133]
Sample: [0.9] expected: [0.78332691]
produced: [0.7988883613421126]
Sample: [0.24] expected: [0.23770263]
produced: [0.21799411904694355]
Sample: [1.2] expected: [0.93203909]
produced: [0.9000670267554165]
Sample: [0.83] expected: [0.73793137]
produced: [0.7602699842845423]
Sample: [0.61] expected: [0.57286746]
produced: [0.587705268468984]
Sample: [0.87] expected: [0.76432894]
produced: [0.7831503853193441]
Sample: [0.49] expected: [0.47062589]
produced: [0.46524455180387586]
Sample: [0.08] expected: [0.07991469]
produced: [0.11305654734059371]
Epoch 5000 RMSE =  0.028321500232299134
Epoch 5100 RMSE =  0.028276080433187845
Epoch 5200 RMSE =  0.02823223969905579
Epoch 5300 RMSE =  0.028189837721967873
Epoch 5400 RMSE =  0.028148725286584405
Epoch 5500 RMSE =  0.028108828482262666
Epoch 5600 RMSE =  0.028070212318449635
Epoch 5700 RMSE =  0.028032747052469553
Epoch 5800 RMSE =  0.027996451088755734
Epoch 5900 RMSE =  0.02796115966133531
Sample: [0.34] expected: [0.33348709]
produced: [0.3085485853610134]
Sample: [0.15] expected: [0.14943813]
produced: [0.15370508371158614]
Sample: [0.49] expected: [0.47062589]
produced: [0.46440446396675183]
Sample: [1.48] expected: [0.99588084]
produced: [0.9434823675389143]
Sample: [0.39] expected: [0.38018842]
produced: [0.3589859047182859]
Sample: [0.54] expected: [0.51413599]
produced: [0.5169345464723965]
Sample: [0.47] expected: [0.45288629]
produced: [0.44329136944669184]
Sample: [0.66] expected: [0.61311685]
produced: [0.6330743708553037]
Sample: [0.44] expected: [0.42593947]
produced: [0.41139806577223703]
Sample: [0.83] expected: [0.73793137]
produced: [0.7604625394382935]
Sample: [1.36] expected: [0.9778646]
produced: [0.9293973032201227]
Sample: [0.87] expected: [0.76432894]
produced: [0.7836601492513531]
Sample: [0.08] expected: [0.07991469]
produced: [0.1143394767803489]
Sample: [0.82] expected: [0.73114583]
produced: [0.7541926642948811]
Sample: [0.24] expected: [0.23770263]
produced: [0.21862521413572764]
Sample: [0.97] expected: [0.82488571]
produced: [0.8316994300558912]
Sample: [0.17] expected: [0.16918235]
produced: [0.16672278901530407]
Sample: [0.56] expected: [0.5311862]
produced: [0.537275830094816]
Sample: [0.8] expected: [0.71735609]
produced: [0.7412785561804567]
Sample: [0.41] expected: [0.39860933]
produced: [0.3796237295002189]
Sample: [1.46] expected: [0.99386836]
produced: [0.9414117850478818]
Sample: [1.1] expected: [0.89120736]
produced: [0.8769046774907017]
Sample: [1.2] expected: [0.93203909]
produced: [0.9017706482027842]
Sample: [0.21] expected: [0.2084599]
produced: [0.1952284960372859]
Sample: [0.48] expected: [0.46177918]
produced: [0.453879934828039]
Sample: [1.45] expected: [0.99271299]
produced: [0.9403828034979403]
Sample: [0.69] expected: [0.63653718]
produced: [0.6589941738109771]
Sample: [0.61] expected: [0.57286746]
produced: [0.586794268166592]
Sample: [0.01] expected: [0.00999983]
produced: [0.08385167422904986]
Sample: [0.9] expected: [0.78332691]
produced: [0.7994224949738032]
Sample: [0.53] expected: [0.50553334]
produced: [0.5063757095332023]
Epoch 6000 RMSE =  0.027926766030019277
Epoch 6100 RMSE =  0.02789356849426547
Epoch 6200 RMSE =  0.027860942200628683
Epoch 6300 RMSE =  0.027829829079566083
Epoch 6400 RMSE =  0.02779921881746816
Epoch 6500 RMSE =  0.02776950590923658
Epoch 6600 RMSE =  0.027740513742880703
Epoch 6700 RMSE =  0.027712447421360798
Epoch 6800 RMSE =  0.027685035082651393
Epoch 6900 RMSE =  0.027658234235723136
Sample: [0.66] expected: [0.61311685]
produced: [0.6322437482886668]
Sample: [0.17] expected: [0.16918235]
produced: [0.16755706735170517]
Sample: [0.47] expected: [0.45288629]
produced: [0.44244775970726]
Sample: [0.97] expected: [0.82488571]
produced: [0.832397787731]
Sample: [0.69] expected: [0.63653718]
produced: [0.6582233582798589]
Sample: [0.83] expected: [0.73793137]
produced: [0.7603289805027507]
Sample: [0.34] expected: [0.33348709]
produced: [0.3084565562422149]
Sample: [0.8] expected: [0.71735609]
produced: [0.7411259897015596]
Sample: [1.2] expected: [0.93203909]
produced: [0.9030326740290483]
Sample: [0.82] expected: [0.73114583]
produced: [0.7540594354589695]
Sample: [0.53] expected: [0.50553334]
produced: [0.505402861228289]
Sample: [0.24] expected: [0.23770263]
produced: [0.21910296148850322]
Sample: [0.01] expected: [0.00999983]
produced: [0.08487286903062613]
Sample: [0.87] expected: [0.76432894]
produced: [0.7836659329000877]
Sample: [0.39] expected: [0.38018842]
produced: [0.35839947058067]
Sample: [1.36] expected: [0.9778646]
produced: [0.9308227899182643]
Sample: [1.48] expected: [0.99588084]
produced: [0.9449686145002961]
Sample: [0.9] expected: [0.78332691]
produced: [0.7997491641659327]
Sample: [1.1] expected: [0.89120736]
produced: [0.8779647386726432]
Sample: [0.61] expected: [0.57286746]
produced: [0.5857621743823868]
Sample: [0.54] expected: [0.51413599]
produced: [0.5157754035621273]
Sample: [1.45] expected: [0.99271299]
produced: [0.9418384571820282]
Sample: [0.08] expected: [0.07991469]
produced: [0.11532284320871664]
Sample: [0.48] expected: [0.46177918]
produced: [0.4529048512012931]
Sample: [0.56] expected: [0.5311862]
produced: [0.5362836769026046]
Sample: [0.49] expected: [0.47062589]
produced: [0.46348972029448715]
Sample: [0.21] expected: [0.2084599]
produced: [0.19582343341376832]
Sample: [0.44] expected: [0.42593947]
produced: [0.41064835528016047]
Sample: [0.15] expected: [0.14943813]
produced: [0.15458940408930263]
Sample: [1.46] expected: [0.99386836]
produced: [0.9429334494046713]
Sample: [0.41] expected: [0.39860933]
produced: [0.37926834123347225]
Epoch 7000 RMSE =  0.027632069279577468
Epoch 7100 RMSE =  0.027606313936920007
Epoch 7200 RMSE =  0.027582242645739205
Epoch 7300 RMSE =  0.02755809960081069
Epoch 7400 RMSE =  0.027534662938714514
Epoch 7500 RMSE =  0.02751175323096318
Epoch 7600 RMSE =  0.027489269718071314
Epoch 7700 RMSE =  0.027467428994340837
Epoch 7800 RMSE =  0.027446096635673618
Epoch 7900 RMSE =  0.027425226127271274
Sample: [0.61] expected: [0.57286746]
produced: [0.5852992358297509]
Sample: [0.54] expected: [0.51413599]
produced: [0.5152942835479535]
Sample: [0.82] expected: [0.73114583]
produced: [0.7541913111499468]
Sample: [0.8] expected: [0.71735609]
produced: [0.7410899942417014]
Sample: [1.2] expected: [0.93203909]
produced: [0.9041592895629242]
Sample: [0.34] expected: [0.33348709]
produced: [0.3085061573879965]
Sample: [0.87] expected: [0.76432894]
produced: [0.7840192921411473]
Sample: [0.97] expected: [0.82488571]
produced: [0.8329317536228538]
Sample: [1.45] expected: [0.99271299]
produced: [0.9431173009784161]
Sample: [0.24] expected: [0.23770263]
produced: [0.21963474448350964]
Sample: [0.44] expected: [0.42593947]
produced: [0.41029162410584075]
Sample: [1.36] expected: [0.9778646]
produced: [0.9321267831242261]
Sample: [0.49] expected: [0.47062589]
produced: [0.4631300614818257]
Sample: [0.83] expected: [0.73793137]
produced: [0.7605446165722343]
Sample: [0.39] expected: [0.38018842]
produced: [0.3584160072332664]
Sample: [0.69] expected: [0.63653718]
produced: [0.6578631102865033]
Sample: [0.21] expected: [0.2084599]
produced: [0.196490210850765]
Sample: [1.46] expected: [0.99386836]
produced: [0.9442179331286888]
Sample: [0.47] expected: [0.45288629]
produced: [0.44203317143372545]
Sample: [0.56] expected: [0.5311862]
produced: [0.5358731084687085]
Sample: [0.15] expected: [0.14943813]
produced: [0.15541081430262804]
Sample: [0.48] expected: [0.46177918]
produced: [0.4526055885215304]
Sample: [1.1] expected: [0.89120736]
produced: [0.879037367244364]
Sample: [0.01] expected: [0.00999983]
produced: [0.08578698918245939]
Sample: [0.08] expected: [0.07991469]
produced: [0.11623473290189569]
Sample: [0.41] expected: [0.39860933]
produced: [0.37902839162112983]
Sample: [1.48] expected: [0.99588084]
produced: [0.94629292214141]
Sample: [0.53] expected: [0.50553334]
produced: [0.5051113248192192]
Sample: [0.17] expected: [0.16918235]
produced: [0.16833905120813605]
Sample: [0.66] expected: [0.61311685]
produced: [0.6318325938536613]
Sample: [0.9] expected: [0.78332691]
produced: [0.8002412165518724]
Epoch 8000 RMSE =  0.027404987058469398
Epoch 8100 RMSE =  0.027385081137660215
Epoch 8200 RMSE =  0.027365647211421814
Epoch 8300 RMSE =  0.027346697087247282
Epoch 8400 RMSE =  0.027328135526116914
Epoch 8500 RMSE =  0.027309560880415744
Epoch 8600 RMSE =  0.02729211336309946
Epoch 8700 RMSE =  0.02727472497586398
Epoch 8800 RMSE =  0.02725772251478287
Epoch 8900 RMSE =  0.027240951842437286
Sample: [0.21] expected: [0.2084599]
produced: [0.19699683393997203]
Sample: [0.9] expected: [0.78332691]
produced: [0.8005614864983587]
Sample: [0.82] expected: [0.73114583]
produced: [0.7542883903455408]
Sample: [0.48] expected: [0.46177918]
produced: [0.4520671852236366]
Sample: [0.01] expected: [0.00999983]
produced: [0.08647283842523663]
Sample: [1.36] expected: [0.9778646]
produced: [0.9331782279769596]
Sample: [0.66] expected: [0.61311685]
produced: [0.6312602710409077]
Sample: [0.34] expected: [0.33348709]
produced: [0.30856604958557815]
Sample: [0.97] expected: [0.82488571]
produced: [0.8335310564051506]
Sample: [0.53] expected: [0.50553334]
produced: [0.5044335476900954]
Sample: [0.24] expected: [0.23770263]
produced: [0.22006233798814095]
Sample: [0.47] expected: [0.45288629]
produced: [0.44157759844898237]
Sample: [0.41] expected: [0.39860933]
produced: [0.37880832036595685]
Sample: [0.39] expected: [0.38018842]
produced: [0.35834405997446345]
Sample: [0.08] expected: [0.07991469]
produced: [0.11697404740199578]
Sample: [1.1] expected: [0.89120736]
produced: [0.8798844007503954]
Sample: [0.15] expected: [0.14943813]
produced: [0.15605588885127983]
Sample: [1.46] expected: [0.99386836]
produced: [0.945328740033629]
Sample: [0.83] expected: [0.73793137]
produced: [0.7607447762824658]
Sample: [0.49] expected: [0.47062589]
produced: [0.4627613116492721]
Sample: [0.61] expected: [0.57286746]
produced: [0.5849328565647697]
Sample: [0.44] expected: [0.42593947]
produced: [0.4101129835459687]
Sample: [0.8] expected: [0.71735609]
produced: [0.7413001355827223]
Sample: [0.56] expected: [0.5311862]
produced: [0.5353471001164942]
Sample: [0.87] expected: [0.76432894]
produced: [0.7843454741056667]
Sample: [1.45] expected: [0.99271299]
produced: [0.9442283579198167]
Sample: [0.54] expected: [0.51413599]
produced: [0.5148564766203249]
Sample: [1.48] expected: [0.99588084]
produced: [0.9473574590035391]
Sample: [0.69] expected: [0.63653718]
produced: [0.6575388600653285]
Sample: [0.17] expected: [0.16918235]
produced: [0.1688803296381798]
Sample: [1.2] expected: [0.93203909]
produced: [0.9051912445789412]
Epoch 9000 RMSE =  0.027224721133265198
Epoch 9100 RMSE =  0.027208631927570514
Epoch 9200 RMSE =  0.02719284864052197
Epoch 9300 RMSE =  0.02717747401621953
Epoch 9400 RMSE =  0.027162437749874382
Epoch 9500 RMSE =  0.027147936933095827
Epoch 9600 RMSE =  0.027133490706171586
Epoch 9700 RMSE =  0.027119354832755765
Epoch 9800 RMSE =  0.027105471313806712
Epoch 9900 RMSE =  0.027091627986314877
Sample: [0.21] expected: [0.2084599]
produced: [0.19738933228449732]
Sample: [0.9] expected: [0.78332691]
produced: [0.8008133511392982]
Sample: [0.66] expected: [0.61311685]
produced: [0.63090564213319]
Sample: [0.48] expected: [0.46177918]
produced: [0.4516691178435758]
Sample: [0.56] expected: [0.5311862]
produced: [0.5347746070205398]
Sample: [1.36] expected: [0.9778646]
produced: [0.934091368980586]
Sample: [0.69] expected: [0.63653718]
produced: [0.6571446654066158]
Sample: [0.97] expected: [0.82488571]
produced: [0.833937260727229]
Sample: [0.24] expected: [0.23770263]
produced: [0.2203597426868369]
Sample: [0.08] expected: [0.07991469]
produced: [0.1175064197483801]
Sample: [0.34] expected: [0.33348709]
produced: [0.30857282361491234]
Sample: [0.17] expected: [0.16918235]
produced: [0.1693553350339596]
Sample: [0.41] expected: [0.39860933]
produced: [0.37853476168230177]
Sample: [1.45] expected: [0.99271299]
produced: [0.9451552670398048]
Sample: [0.01] expected: [0.00999983]
produced: [0.08708858748146964]
Sample: [0.15] expected: [0.14943813]
produced: [0.15652233420588293]
Sample: [1.48] expected: [0.99588084]
produced: [0.9482731324367843]
Sample: [0.61] expected: [0.57286746]
produced: [0.5843980403870903]
Sample: [1.46] expected: [0.99386836]
produced: [0.9462207385065992]
Sample: [0.47] expected: [0.45288629]
produced: [0.4412575040605156]
Sample: [0.49] expected: [0.47062589]
produced: [0.462332408549617]
Sample: [0.83] expected: [0.73793137]
produced: [0.760758610675391]
Sample: [1.1] expected: [0.89120736]
produced: [0.8805448334174512]
Sample: [0.8] expected: [0.71735609]
produced: [0.7412083099360517]
Sample: [0.44] expected: [0.42593947]
produced: [0.4097144324084298]
Sample: [0.39] expected: [0.38018842]
produced: [0.35813735089340343]
Sample: [0.54] expected: [0.51413599]
produced: [0.5144752590674712]
Sample: [0.53] expected: [0.50553334]
produced: [0.5041449721464688]
Sample: [0.87] expected: [0.76432894]
produced: [0.7845752519692257]
Sample: [1.2] expected: [0.93203909]
produced: [0.9060334468259054]
Sample: [0.82] expected: [0.73114583]
produced: [0.7543954709331987]
Epoch 10000 RMSE =  0.027078477496806082
Final Epoch RMSE =  0.027078477496806082
"""