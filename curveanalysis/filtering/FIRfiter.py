import scipy.signal as signal



class FIRfilter:
    filter_n = 100

    def __init__(self, n, cutoff_hz, sample_rate):
        self.filter_n = n
        nyq_rate = sample_rate / 2.0
        cutoff_hz = 0.01
        self.firfilter = signal.firwin(n, cutoff = cutoff_hz/nyq_rate, window = "bartlett")

    def filter(self, rawsignal):
        return signal.lfilter(self.firfilter, 1.0, rawsignal)