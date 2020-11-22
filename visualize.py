#! /usr/bin/python3
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys

fig, (graph_1) = plt.subplots(1,1)

# We want 2 minutes of data, at 20MS/s -> 120 * 20e6 = 2400e6
# The metadata says we have 352455250 samples
with open(sys.argv[1], "rb") as f:
    print("Reading file")
    mag = np.absolute(np.fromfile(f, np.complex64, -1))

print("Decimating")
mag = signal.decimate(mag, 10)
mag = signal.decimate(mag, 10)
mag = signal.decimate(mag, 10)
mag = signal.decimate(mag, 10)

print(len(mag))


fig.set_figheight(7)
fig.set_figwidth(25)


x = range(len(mag))

print("Graphing")
graph_1.plot(x, mag)
plt.show()

#The metadata says cf32 so I assume that's 32 bit floats for I and 32bfloat for Q

