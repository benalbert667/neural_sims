import numpy as np
import random


class Neuron:
    def __init__(self, firing_rates, wiggle=0):
        """
        A single Neuron is defined by a list of firing rates (indexed by state number).
        A 'wiggle' can also be defined, allowing for a random delay or early start to a state transition.
        :param firing_rates: list of firing rates or lists of firing rates
        :param wiggle: max offset in steps from true state change step
        """
        self.__frs = firing_rates

        self.wiggle = wiggle

    def get_p(self, state_num):
        if type(self.__frs[state_num]) == float:
            return self.__frs[state_num]
        return random.choice(self.__frs[state_num])


class FakeData:
    def __init__(self, neurons, wiggle=None):
        """
        Define fake data by sets of fake neurons, which each have their own firing probabilities
        for each state (0 ... n), or multiple, which it will take on one at random.
        :param neurons: a list of predefined Neurons
        :param wiggle: max offset in steps from true state change step
        """
        self.neu = tuple(neurons)
        self.saved = None

        if wiggle is not None:
            for neuron in self.neu:
                if not neuron.wiggle:  # ignore neurons with custom wiggles
                    neuron.wiggle = wiggle

    def generate(self, steps: int, state_times: list, categorical=False, save=True):
        """
        Generates 'steps' time steps of data.  'state_times' (indexed by state number)
        defines when each state STARTS, meaning at least one of the values must be zero,
        and all values must be less than 'steps'.  The list [0] signifies only a single state
        throughout.
        :param steps: number of time steps to generate
        :param state_times: start times for each state
        :param categorical: if true, each time step will only show a single neuron firing (randomly select a neuron
        for each step)
        :param save: if True, saves the returned array in the FakeData object
        :return: a numpy array (num_neurons X steps) of generated data
        """
        assert all(st < steps for st in state_times) and any(st == 0 for st in state_times), \
            "Incorrect transition times or number of steps."

        data = np.random.rand(len(self.neu), steps)
        states = np.argsort(state_times)  # states in order of occurrence
        stops = sorted(state_times)[1:] + [steps]  # take off leading 0, add trailing 'steps'

        for n, neuron in enumerate(self.neu):
            start = 0
            for state, stop in zip(states, stops):
                p = neuron.get_p(state)
                if stop != steps:  # not last stop
                    while True:
                        offset = int(np.random.normal(0, .3) * neuron.wiggle)
                        if 0 <= start < stop + offset < steps:
                            stop += offset
                            break
                data[n, start:stop] = np.where(data[n, start:stop] < p, 1, 0)
                start = stop

        # Move through each time step and change to categorical
        if categorical:
            for s in range(steps):
                fires = np.where(data[:, s] == 1)[0]
                if fires.size > 0:
                    # choose a random neuron to fire at this time step
                    fires = np.delete(fires, random.randint(0, fires.size - 1))
                    data[fires, s] = 0  # make all other neurons not fire

        if save:
            self.saved = data.copy()
        return data


def random_fake_data(
        max_num_neurons=40,
        max_num_states=6,
        max_firing_rate=0.01,
        timesteps=1000,
        min_state_duration=None,
        wiggle=20,
        categorical=False):
    """
    Wrapper for FakeData, generates data with a random number of neurons each with
    random firing rates for a random number of states that start at random timesteps.
    :param max_num_neurons: Maximum number of neurons
    :param max_num_states: Maximum number of states
    :param max_firing_rate: Maximum probability for a neuron to fire in any state (per time step)
    :param timesteps: Number of time steps of data to be generated
    :param min_state_duration: Minimum number of timesteps that a single state can occupy
    :param wiggle: Wiggle value for the data (see FakeData)
    :param categorical: When true, generated data will be categorical
    :return: The generated data (a numpy array of shape (num_neurons X timesteps))
    """
    # If minimum state duration is not specified, set to number of timesteps /20
    if min_state_duration is None:
        min_state_duration = timesteps // 20
    # Define hyper-parameters
    num_neurons = random.randint(2, max_num_neurons)
    num_states = random.randint(1, max_num_states)
    # Create Neuron objects
    neurons = []
    for _ in range(num_neurons):
        firing_rates = [random.random() * max_firing_rate for _ in range(num_states)]
        neuron = Neuron(firing_rates)
        neurons.append(neuron)
    # Generate and return data
    gen = FakeData(neurons, wiggle=wiggle)
    state_times = sorted([0] +
                         [random.randint(min_state_duration, timesteps - min_state_duration)
                          for _ in range(num_states - 1)])
    while True:
        for i in range(num_states - 1):
            if state_times[i+1] - state_times[i] < min_state_duration:
                state_times = sorted([0] +
                                     [random.randint(min_state_duration, timesteps - min_state_duration)
                                      for _ in range(num_states - 1)])
                break
        else:
            break
    return gen.generate(timesteps, state_times, categorical=categorical, save=False), state_times


if __name__ == '__main__':
    # Define neurons
    fd = FakeData([
        Neuron([1/3, 1/10, 0.995]),
        Neuron([1/5, 1/5, 1/5]),
        Neuron([0.8, [0.6, 0.3], 0.8]),
        Neuron([[1/2, 1/4], [1/3, 2/3], [1/10, 1/11]]),
        Neuron([0.11, 0.12, 0.13]),
    ], wiggle=20)
    # Generate normal data
    dat = fd.generate(300, [0, 100, 250])
    for i, row in enumerate(dat):
        print('Neuron ' + str(i + 1) + ':', end=' ')
        for r in row:
            print('X' if r else '-', end='')
        print()
    print('\n\n')
    # Generate categorical data
    dat = fd.generate(300, [0, 50, 230], categorical=True)
    for i, row in enumerate(dat):
        print('Neuron ' + str(i + 1) + ':', end=' ')
        for r in row:
            print('X' if r else '-', end='')
        print()
    print('\n\n')
    # Generate random data
    dat = random_fake_data()
    for i, row in enumerate(dat):
        print('Neuron ' + str(i + 1) + ':', end=' ')
        for r in row:
            print('X' if r else '-', end='')
        print()
