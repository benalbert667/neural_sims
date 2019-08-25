from fake_data import random_fake_data
import numpy as np
import matplotlib.pyplot as plt


fd, state_times = random_fake_data(max_num_neurons=10, timesteps=10000)

print(fd.shape)
print(state_times)

for nfiring in fd:
    ave = np.average(nfiring)
    print(ave)
    p = []
    s = 0
    for d in nfiring:
        s += d - ave
        # s += d
        p.append(s)

    plt.plot(p, color="blue")
    for st in state_times:
        plt.axvline(x=st, color="red")
    plt.show()
