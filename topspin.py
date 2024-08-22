class TopSpinState:
    """
    Class to represent a state of the TopSpin puzzle
    """
    def __init__(self, state, k=4):
        """
        Constructor
        :param state: state of the puzzle
        :param k: number of elements to flip
        """
        self.state_list = state
        self._k = k
        self._n = len(state)

    def get_n(self):
        """
        Get the length of the state
        :return: length of the state
        """
        return self._n


    def get_k(self):
        """
        Get the number of elements to flip
        :return: number of elements to flip
        """
        return self._k

    def is_goal(self):
        """
        Check if the state is the goal state
        :return: True if the state is the goal state, False otherwise
        """
        for i in range(0, self._n):
            if self.state_list[i] != i+1:
                return False
        return True

    def get_state_as_list(self):
        """
        Get the state as a list
        :return: state as a list
        """
        return self.state_list

    def clockwise(self):
        """
        Rotate the state clockwise
        :return: new state after rotating the current state clockwise
        """
        return TopSpinState(self.state_list[-1:] + self.state_list[:-1], self._k)

    def counter_clockwise(self):
        """
        Rotate the state counter-clockwise
        :return:  new state after rotating the current state counter-clockwise
        """
        return TopSpinState(self.state_list[1:] + self.state_list[:1], self._k)

    def flip(self):
        """
        Flip the first k elements of the state
        :return: new state after flipping the first k elements of the current state
        """
        return TopSpinState(self.state_list[:self._k][::-1] + self.state_list[self._k:], self._k)

    def get_neighbors(self):
        """
        Get the neighbors of the current state
        :return: list of neighbors of the current state
        """
        return [(self.clockwise(), 1), (self.counter_clockwise(), 1), (self.flip(), 1)]


    def __eq__(self, other):
        """
        Check if two states are equal
        :param other: state to compare with
        :return: True if the states are equal, False otherwise
        """
        return self.state_list == other.state_list


    def __hash__(self):
        """
        Hash function
        :return: hash value of the state
        """
        state_as_tuple = tuple(self.state_list)
        return hash(state_as_tuple)




