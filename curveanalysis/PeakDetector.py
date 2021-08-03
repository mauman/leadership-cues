import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class PeakDetector:

    def __init__(self, signal, windowlength, threshold):
        self.signal = signal
        self.output = np.zeros(len(signal))
        self.windowlength = windowlength
        self.threshold = threshold

        

    def FindPeaks(self):
        peaks, _ = find_peaks(self.signal, distance=self.windowlength, height=self.threshold)
        self.output[peaks] = 1
        return self.output
        # plt.plot(self.signal)
        # plt.figure()
        # plt.plot(self.output)
        # plt.show()


