import scipy.signal as signal

class Acceleration3D:

    def __init__(self):
        self.pX = None
        self.pY = None
        self.pZ = None
        self.vX = None
        self.vY = None
        self.vZ = None
        self.KE = None
        

    def __init__(self, pX, pY, pZ):
        self.SetSignal(pX, pY, pZ)
        self.Acc = None
        

    def SetSignal(self, pX, pY, pZ):
        self.pX = pX
        self.pY = pY
        self.pZ = pZ
        if len(pX) > 51 and len(pY) > 51 and len(pY) > 51:
            self.aX = signal.savgol_filter(self.pX, 51, 2, 2)
            self.aY = signal.savgol_filter(self.pY, 51, 2, 2)
            self.aZ = signal.savgol_filter(self.pZ, 51, 2, 2)
        else:
            self.aX = None
            self.aY = None
            self.aZ = None

    def GetAcc(self):
        if self.aX is not None and self.aY is not None and self.aZ is not None:
            self.Acc = self.aX ** 2 + self.aY ** 2 + self.aZ ** 2
        return self.Acc
        