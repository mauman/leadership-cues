import scipy.signal as signal

class KineticEnergy3D:

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
        self.KE = None
        

    def SetSignal(self, pX, pY, pZ):
        self.pX = pX
        self.pY = pY
        self.pZ = pZ
        if len(pX) > 51 and len(pY) > 51 and len(pY) > 51:
            self.vX = signal.savgol_filter(self.pX, 51, 2, 1)
            self.vY = signal.savgol_filter(self.pY, 51, 2, 1)
            self.vZ = signal.savgol_filter(self.pZ, 51, 2, 1)
        else:
            self.vX = None
            self.vY = None
            self.vZ = None

    def GetKE(self):
        if self.vX is not None and self.vY is not None and self.vZ is not None:
            self.KE = self.vX ** 2 + self.vY ** 2 + self.vZ ** 2
        return self.KE
        