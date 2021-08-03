from math import pi
from filemanagement.QTMTSVFileReader import QTMTSVFileReader
from filemanagement.CSVFileReader import CSVFileReader
from filemanagement.CSVFileWriter import CSVFileWriter
from curveanalysis.filtering.FIRfiter import FIRfilter
from curveanalysis.KineticEnergy3D import KineticEnergy3D
from curveanalysis.Acceleration3D import Acceleration3D
from curveanalysis.Velocity3D import Velocity3D
from curveanalysis.PeakDetector import PeakDetector
from curveanalysis.Anticipation import Anticipation
from basic.Distance import Distance
import scipy.signal as signal
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys

def countAboveThreshold(x, threshold):
    count = 0
    for item in x:
        if item > threshold:
            count = count + 1
    return count

def countAboveBelowThreshold(x, y, threshold1, threshold2):
    count = 0
    for i in range(len(x)):
        item1 = x[i]
        item2 = y[i]
        if item1 > threshold1 and item2 < threshold2:
            count = count + 1
    return count

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

# print(py_ang(np.array([-10, 0, 0]), np.array([-10, 0, 0])))

# sys.exit()

lowpassfilter = FIRfilter(100, 0.01, 100)

outputfile = CSVFileWriter("touch-output.csv", ",", "\"")

# segmentsfile = CSVFileReader("D:/lea_short_august2016/3rd time/new500all.csv", " ")
segmentsfile = CSVFileReader("D:/lea_short_august2016/3rd time/new250all.csv", " ")
windowlength = 250


segmentsfile.GetNext()
segmentsfile.GetNext()
segmentsfile.GetNext()

prev_trialNum = -1

segmentsCounter = 0

segment = 0

while segment != []:

    segment = segmentsfile.GetNext()

    if len(segment) == 0:
        break

    trialNum = int(segment[2])
    if trialNum != prev_trialNum:
        trialFileName = "D:/lea_short_august2016/trial_%03i.tsv" % trialNum
        qtmfile = QTMTSVFileReader(trialFileName)
        prev_trialNum = trialNum
    secondColumn = segment[1]
    person1Name = segment[6]
    person2Name = segment[7]
    leader = int(segment[8])
    framestart = int(segment[4])
    frameend = int(segment[5])
    print("segment: from frame %i to frame %i, person 1 is %s and person 2 is %s, leader is %i" % (framestart, frameend, person1Name, person2Name, leader))

    segmentsCounter = segmentsCounter + 1

    # C1 - Stability

    person1_W_Stab_Count = 0
    person2_W_Stab_Count = 0
    person1_A_Stab_Count = 0
    person2_A_Stab_Count = 0

    person1_WL_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_WRIST_LEFT X")
    person1_WL_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_WRIST_LEFT Y")
    person1_WL_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_WRIST_LEFT Z")
    person1_WR_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_WRIST_RIGHT X")
    person1_WR_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_WRIST_RIGHT Y")
    person1_WR_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_WRIST_RIGHT Z")

    person2_WL_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_WRIST_LEFT X")
    person2_WL_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_WRIST_LEFT Y")
    person2_WL_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_WRIST_LEFT Z")
    person2_WR_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_WRIST_RIGHT X")
    person2_WR_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_WRIST_RIGHT Y")
    person2_WR_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_WRIST_RIGHT Z")

    person1_AL_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_ANKLE_LEFT X")
    person1_AL_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_ANKLE_LEFT Y")
    person1_AL_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_ANKLE_LEFT Z")
    person1_AR_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_ANKLE_RIGHT X")
    person1_AR_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_ANKLE_RIGHT Y")
    person1_AR_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_ANKLE_RIGHT Z")

    person2_AL_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_ANKLE_LEFT X")
    person2_AL_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_ANKLE_LEFT Y")
    person2_AL_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_ANKLE_LEFT Z")
    person2_AR_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_ANKLE_RIGHT X")
    person2_AR_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_ANKLE_RIGHT Y")
    person2_AR_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_ANKLE_RIGHT Z")

    

    if len(person1_WL_pX) == 0 or len(person1_WL_pY) == 0 or len(person1_WL_pZ) == 0 or \
        len(person1_WR_pX) == 0 or len(person1_WR_pY) == 0 or len(person1_WR_pZ) == 0 or \
        len(person2_WL_pX) == 0 or len(person2_WL_pY) == 0 or len(person2_WL_pZ) == 0 or \
        len(person2_WR_pX) == 0 or len(person2_WR_pY) == 0 or len(person2_WR_pZ) == 0 or \
        len(person1_AL_pX) == 0 or len(person1_AL_pY) == 0 or len(person1_AL_pZ) == 0 or \
        len(person1_AR_pX) == 0 or len(person1_AR_pY) == 0 or len(person1_AR_pZ) == 0 or \
        len(person2_AL_pX) == 0 or len(person2_AL_pY) == 0 or len(person2_AL_pZ) == 0 or \
        len(person2_AR_pX) == 0 or len(person2_AR_pY) == 0 or len(person2_AR_pZ) == 0:        
        print("*** Warning: wrist or ankle position vector is empty - file %s" % trialFileName)
        continue
    else:

        # kinetic energy

        person1_WL_KE = KineticEnergy3D(person1_WL_pX, person1_WL_pY, person1_WL_pZ).GetKE()
        person1_WR_KE = KineticEnergy3D(person1_WR_pX, person1_WR_pY, person1_WR_pZ).GetKE()

        person2_WL_KE = KineticEnergy3D(person2_WL_pX, person2_WL_pY, person2_WL_pZ).GetKE()
        person2_WR_KE = KineticEnergy3D(person2_WR_pX, person2_WR_pY, person2_WR_pZ).GetKE()

        person1_AL_KE = KineticEnergy3D(person1_AL_pX, person1_AL_pY, person1_AL_pZ).GetKE()
        person1_AR_KE = KineticEnergy3D(person1_AR_pX, person1_AR_pY, person1_AR_pZ).GetKE()

        person2_AL_KE = KineticEnergy3D(person2_AL_pX, person2_AL_pY, person2_AL_pZ).GetKE()
        person2_AR_KE = KineticEnergy3D(person2_AR_pX, person2_AR_pY, person2_AR_pZ).GetKE()

        # acceleration

        person1_WL_Acc = Acceleration3D(person1_WL_pX, person1_WL_pY, person1_WL_pZ).GetAcc()
        person1_WR_Acc = Acceleration3D(person1_WR_pX, person1_WR_pY, person1_WR_pZ).GetAcc()

        person2_WL_Acc = Acceleration3D(person2_WL_pX, person2_WL_pY, person2_WL_pZ).GetAcc()
        person2_WR_Acc = Acceleration3D(person2_WR_pX, person2_WR_pY, person2_WR_pZ).GetAcc()

        person1_AL_Acc = Acceleration3D(person1_AL_pX, person1_AL_pY, person1_AL_pZ).GetAcc()
        person1_AR_Acc = Acceleration3D(person1_AR_pX, person1_AR_pY, person1_AR_pZ).GetAcc()

        person2_AL_Acc = Acceleration3D(person2_AL_pX, person2_AL_pY, person2_AL_pZ).GetAcc()
        person2_AR_Acc = Acceleration3D(person2_AR_pX, person2_AR_pY, person2_AR_pZ).GetAcc()


        if person1_WL_KE is not None and person1_WR_KE is not None and person2_WL_KE is not None and person2_WR_KE is not None\
        and person1_AL_KE is not None and person1_AR_KE is not None and person2_AL_KE is not None and person2_AR_KE is not None:

            person1_WL_KE_filtered = lowpassfilter.filter(person1_WL_KE)
            # plt.figure()
            # plt.plot(person1_WL_KE_filtered)
            person1_WR_KE_filtered = lowpassfilter.filter(person1_WR_KE)
        
            person2_WL_KE_filtered = lowpassfilter.filter(person2_WL_KE)
            person2_WR_KE_filtered = lowpassfilter.filter(person2_WR_KE)
            
            person1_AL_KE_filtered = lowpassfilter.filter(person1_AL_KE)
            person1_AR_KE_filtered = lowpassfilter.filter(person1_AR_KE)
        
            person2_AL_KE_filtered = lowpassfilter.filter(person2_AL_KE)
            person2_AR_KE_filtered = lowpassfilter.filter(person2_AR_KE)

            person1_W_Stab = abs(person1_WL_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                - person1_WL_KE[:-lowpassfilter.filter_n]) \
                    + abs(person1_WR_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                        - person1_WR_KE[:-lowpassfilter.filter_n])

            person2_W_Stab = abs(person2_WL_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                - person2_WL_KE[:-lowpassfilter.filter_n]) \
                    + abs(person2_WR_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                        - person2_WR_KE[:-lowpassfilter.filter_n])

            person1_A_Stab = abs(person1_AL_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                - person1_AL_KE[:-lowpassfilter.filter_n]) \
                    + abs(person1_AR_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                        - person1_AR_KE[:-lowpassfilter.filter_n])

            person2_A_Stab = abs(person2_AL_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                - person2_AL_KE[:-lowpassfilter.filter_n]) \
                    + abs(person2_AR_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2] \
                        - person2_AR_KE[:-lowpassfilter.filter_n])


            # plt.figure()
            # plt.plot(person1_WL_KE_filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2])
            # plt.plot(person1_WL_KE[:-lowpassfilter.filter_n])
            # plt.show()

            person1_W_Stab_Count = countAboveThreshold(person1_W_Stab, 0.5)
            person2_W_Stab_Count = countAboveThreshold(person2_W_Stab, 0.5)
            person1_A_Stab_Count = countAboveThreshold(person1_A_Stab, 0.5)
            person2_A_Stab_Count = countAboveThreshold(person2_A_Stab, 0.5)

        else:
            print("*** Warning: velocity vector is too short, file %s" % trialFileName)
            continue
    
    
    # C2 - Anticipation

    person1_HL_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HIP_LEFT X")
    person1_HL_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HIP_LEFT Y")
    person1_HL_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HIP_LEFT Z")
    person1_HR_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HIP_RIGHT X")
    person1_HR_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HIP_RIGHT Y")
    person1_HR_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HIP_RIGHT Z")

    person2_HL_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HIP_LEFT X")
    person2_HL_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HIP_LEFT Y")
    person2_HL_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HIP_LEFT Z")
    person2_HR_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HIP_RIGHT X")
    person2_HR_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HIP_RIGHT Y")
    person2_HR_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HIP_RIGHT Z")

    if len(person1_HL_pX) == 0 or len(person1_HL_pY) == 0 or len(person1_HL_pZ) == 0 or \
        len(person2_HL_pX) == 0 or len(person2_HL_pY) == 0 or len(person2_HL_pZ) == 0 or \
        len(person1_HR_pX) == 0 or len(person1_HR_pY) == 0 or len(person1_HR_pZ) == 0 or \
        len(person2_HR_pX) == 0 or len(person2_HR_pY) == 0 or len(person2_HR_pZ) == 0:
        print("*** Warning: hip position vector is empty - file %s" % trialFileName)
        continue
    
    else:
        facing = np.zeros(len(person1_HL_pX))
        for i in range(len(person1_HL_pX)):
            hip1 = np.array([   person1_HL_pX[i] - person1_HR_pX[i], \
                                person1_HL_pY[i] - person1_HR_pY[i], \
                                person1_HL_pZ[i] - person1_HR_pZ[i]])
            hip2 = np.array([   person2_HL_pX[i] - person2_HR_pX[i], \
                                person2_HL_pY[i] - person2_HR_pY[i], \
                                person2_HL_pZ[i] - person2_HR_pZ[i]])
            facing[i] = py_ang(hip1, hip2) / 3.141

        person1_WL_Acc_peaks = PeakDetector(person1_WL_Acc, 10, 0.002).FindPeaks()
        person1_WR_Acc_peaks = PeakDetector(person1_WR_Acc, 10, 0.002).FindPeaks()

        person2_WL_Acc_peaks = PeakDetector(person2_WL_Acc, 10, 0.002).FindPeaks()
        person2_WR_Acc_peaks = PeakDetector(person2_WR_Acc, 10, 0.002).FindPeaks()

        person1_AL_Acc_peaks = PeakDetector(person1_AL_Acc, 10, 0.002).FindPeaks()
        person1_AR_Acc_peaks = PeakDetector(person1_AR_Acc, 10, 0.002).FindPeaks()

        person2_AL_Acc_peaks = PeakDetector(person2_AL_Acc, 10, 0.002).FindPeaks()
        person2_AR_Acc_peaks = PeakDetector(person2_AR_Acc, 10, 0.002).FindPeaks()

        p1_WL_p2_WL_Ant = Anticipation(person1_WL_Acc_peaks, person2_WL_Acc_peaks, min(windowlength, len(person1_WL_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()
        p1_WL_p2_WR_Ant = Anticipation(person1_WL_Acc_peaks, person2_WR_Acc_peaks, min(windowlength, len(person1_WL_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()
        p1_WR_p2_WL_Ant = Anticipation(person1_WR_Acc_peaks, person2_WL_Acc_peaks, min(windowlength, len(person2_WL_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()
        p1_WR_p2_WR_Ant = Anticipation(person1_WR_Acc_peaks, person2_WR_Acc_peaks, min(windowlength, len(person2_WR_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()

        p1_AL_p2_AL_Ant = Anticipation(person1_AL_Acc_peaks, person2_AL_Acc_peaks, min(windowlength, len(person2_AL_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()
        p1_AL_p2_AR_Ant = Anticipation(person1_AL_Acc_peaks, person2_AR_Acc_peaks, min(windowlength, len(person2_AR_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()
        p1_AR_p2_AL_Ant = Anticipation(person1_AR_Acc_peaks, person2_AL_Acc_peaks, min(windowlength, len(person2_AL_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()
        p1_AR_p2_AR_Ant = Anticipation(person1_AR_Acc_peaks, person2_AR_Acc_peaks, min(windowlength, len(person2_AR_Acc_peaks)), int(windowlength - (windowlength * 0.10))).GetAnt()

        p1_p2_W_Ant =   (p1_WL_p2_WR_Ant[0] + p1_WR_p2_WL_Ant[0]) / 2 * facing + \
                        (p1_WR_p2_WR_Ant[0] + p1_WL_p2_WL_Ant[0]) / 2 * (1 - facing)
        
        p2_p1_W_Ant =   (p1_WL_p2_WR_Ant[1] + p1_WR_p2_WL_Ant[1]) / 2 * facing + \
                        (p1_WR_p2_WR_Ant[1] + p1_WL_p2_WL_Ant[1]) / 2 * (1 - facing)

        p1_p2_A_Ant =   (p1_AL_p2_AR_Ant[0] + p1_AR_p2_AL_Ant[0]) / 2 * facing + \
                        (p1_AR_p2_AR_Ant[0] + p1_AL_p2_AL_Ant[0]) / 2 * (1 - facing)
        
        p2_p1_A_Ant =   (p1_AL_p2_AR_Ant[1] + p1_AR_p2_AL_Ant[1]) / 2 * facing + \
                        (p1_AR_p2_AR_Ant[1] + p1_AL_p2_AL_Ant[1]) / 2 * (1 - facing)

        p1_p2_W_Ant_Count = countAboveThreshold(p1_p2_W_Ant, 0.2)
        p2_p1_W_Ant_Count = countAboveThreshold(p2_p1_W_Ant, 0.2)
        p1_p2_A_Ant_Count = countAboveThreshold(p1_p2_A_Ant, 0.2)
        p2_p1_A_Ant_Count = countAboveThreshold(p2_p1_A_Ant, 0.2)
    
    # C5 - Push
    
    person1_SB_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_STERNUM_BASE X")
    person1_SB_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_STERNUM_BASE Y")
    person1_SB_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_STERNUM_BASE Z")

    person2_SB_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_STERNUM_BASE X")
    person2_SB_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_STERNUM_BASE Y")
    person2_SB_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_STERNUM_BASE Z")

    if len(person1_SB_pX) == 0 or len(person1_SB_pY) == 0 or len(person1_SB_pZ) == 0 or \
        len(person2_SB_pX) == 0 or len(person2_SB_pY) == 0 or len(person2_SB_pZ) == 0:
        print("*** Warning: sternum position vector is empty - file %s" % trialFileName)
        continue
    else:
        p1_WR_p2_WR_Dist = Distance(person1_WR_pX, person1_WR_pY, person1_WR_pZ, person2_WR_pX, person2_WR_pY, person2_WR_pZ).GetDistance()
        p1_WL_p2_WR_Dist = Distance(person1_WL_pX, person1_WL_pY, person1_WL_pZ, person2_WR_pX, person2_WR_pY, person2_WR_pZ).GetDistance()
        p1_WR_p2_WL_Dist = Distance(person1_WR_pX, person1_WR_pY, person1_WR_pZ, person2_WL_pX, person2_WL_pY, person2_WL_pZ).GetDistance()
        p1_WL_p2_WL_Dist = Distance(person1_WL_pX, person1_WL_pY, person1_WL_pZ, person2_WL_pX, person2_WL_pY, person2_WL_pZ).GetDistance()

        p1_WR_p2_WR_pX = (person1_WR_pX + person2_WR_pX) / 2
        p1_WR_p2_WR_pY = (person1_WR_pY + person2_WR_pY) / 2
        p1_WR_p2_WR_pZ = (person1_WR_pZ + person2_WR_pZ) / 2

        p1_WL_p2_WR_pX = (person1_WL_pX + person2_WR_pX) / 2
        p1_WL_p2_WR_pY = (person1_WL_pY + person2_WR_pY) / 2
        p1_WL_p2_WR_pZ = (person1_WL_pZ + person2_WR_pZ) / 2

        p1_WR_p2_WL_pX = (person1_WR_pX + person2_WL_pX) / 2
        p1_WR_p2_WL_pY = (person1_WR_pY + person2_WL_pY) / 2
        p1_WR_p2_WL_pZ = (person1_WR_pZ + person2_WL_pZ) / 2

        p1_WL_p2_WL_pX = (person1_WL_pX + person2_WL_pX) / 2
        p1_WL_p2_WL_pY = (person1_WL_pY + person2_WL_pY) / 2
        p1_WL_p2_WL_pZ = (person1_WL_pZ + person2_WL_pZ) / 2

        p1_WR_p2_WR_p1_SB_Dist = Distance(p1_WR_p2_WR_pX, p1_WR_p2_WR_pY, p1_WR_p2_WR_pZ, person1_SB_pX, person1_SB_pY, person1_SB_pZ).GetDistance()
        p1_WR_p2_WR_p2_SB_Dist = Distance(p1_WR_p2_WR_pX, p1_WR_p2_WR_pY, p1_WR_p2_WR_pZ, person2_SB_pX, person2_SB_pY, person2_SB_pZ).GetDistance()
        p1_WR_p2_WR_p1_SB_Vel = Velocity3D(p1_WR_p2_WR_p1_SB_Dist, np.zeros(len(p1_WR_p2_WR_p1_SB_Dist)), np.zeros(len(p1_WR_p2_WR_p1_SB_Dist))).GetVel()
        p1_WR_p2_WR_p2_SB_Vel = Velocity3D(p1_WR_p2_WR_p2_SB_Dist, np.zeros(len(p1_WR_p2_WR_p2_SB_Dist)), np.zeros(len(p1_WR_p2_WR_p2_SB_Dist))).GetVel()


        p1_WL_p2_WR_p1_SB_Dist = Distance(p1_WL_p2_WR_pX, p1_WL_p2_WR_pY, p1_WL_p2_WR_pZ, person1_SB_pX, person1_SB_pY, person1_SB_pZ).GetDistance()
        p1_WL_p2_WR_p2_SB_Dist = Distance(p1_WL_p2_WR_pX, p1_WL_p2_WR_pY, p1_WL_p2_WR_pZ, person2_SB_pX, person2_SB_pY, person2_SB_pZ).GetDistance()
        p1_WL_p2_WR_p1_SB_Vel = Velocity3D(p1_WL_p2_WR_p1_SB_Dist, np.zeros(len(p1_WL_p2_WR_p1_SB_Dist)), np.zeros(len(p1_WL_p2_WR_p1_SB_Dist))).GetVel()
        p1_WL_p2_WR_p2_SB_Vel = Velocity3D(p1_WL_p2_WR_p2_SB_Dist, np.zeros(len(p1_WL_p2_WR_p2_SB_Dist)), np.zeros(len(p1_WL_p2_WR_p2_SB_Dist))).GetVel()

        p1_WR_p2_WL_p1_SB_Dist = Distance(p1_WR_p2_WL_pX, p1_WR_p2_WL_pY, p1_WR_p2_WL_pZ, person1_SB_pX, person1_SB_pY, person1_SB_pZ).GetDistance()
        p1_WR_p2_WL_p2_SB_Dist = Distance(p1_WR_p2_WL_pX, p1_WR_p2_WL_pY, p1_WR_p2_WL_pZ, person2_SB_pX, person2_SB_pY, person2_SB_pZ).GetDistance()
        p1_WR_p2_WL_p1_SB_Vel = Velocity3D(p1_WR_p2_WL_p1_SB_Dist, np.zeros(len(p1_WR_p2_WL_p1_SB_Dist)), np.zeros(len(p1_WR_p2_WL_p1_SB_Dist))).GetVel()
        p1_WR_p2_WL_p2_SB_Vel = Velocity3D(p1_WR_p2_WL_p2_SB_Dist, np.zeros(len(p1_WR_p2_WL_p2_SB_Dist)), np.zeros(len(p1_WR_p2_WL_p2_SB_Dist))).GetVel()

        p1_WL_p2_WL_p1_SB_Dist = Distance(p1_WL_p2_WL_pX, p1_WL_p2_WL_pY, p1_WL_p2_WL_pZ, person1_SB_pX, person1_SB_pY, person1_SB_pZ).GetDistance()
        p1_WL_p2_WL_p2_SB_Dist = Distance(p1_WL_p2_WL_pX, p1_WL_p2_WL_pY, p1_WL_p2_WL_pZ, person2_SB_pX, person2_SB_pY, person2_SB_pZ).GetDistance()
        p1_WL_p2_WL_p1_SB_Vel = Velocity3D(p1_WL_p2_WL_p1_SB_Dist, np.zeros(len(p1_WL_p2_WL_p1_SB_Dist)), np.zeros(len(p1_WL_p2_WL_p1_SB_Dist))).GetVel()
        p1_WL_p2_WL_p2_SB_Vel = Velocity3D(p1_WL_p2_WL_p2_SB_Dist, np.zeros(len(p1_WL_p2_WL_p2_SB_Dist)), np.zeros(len(p1_WL_p2_WL_p2_SB_Dist))).GetVel()

        hands_pair1 = np.zeros(len(p1_WR_p2_WR_Dist))
        for i in range(len(p1_WR_p2_WR_Dist)):
            if p1_WR_p2_WR_Dist[i] < 280:
                hands_pair1[i] = p1_WR_p2_WR_p1_SB_Vel[i] - p1_WR_p2_WR_p2_SB_Vel[i]

        hands_pair2 = np.zeros(len(p1_WL_p2_WR_Dist))
        for i in range(len(p1_WL_p2_WR_Dist)):
            if p1_WL_p2_WR_Dist[i] < 280:
                hands_pair2[i] = p1_WL_p2_WR_p1_SB_Vel[i] - p1_WL_p2_WR_p2_SB_Vel[i]

        hands_pair3 = np.zeros(len(p1_WR_p2_WL_Dist))
        for i in range(len(p1_WR_p2_WL_Dist)):
            if p1_WR_p2_WL_Dist[i] < 280:
                hands_pair3[i] = p1_WL_p2_WR_p1_SB_Vel[i] - p1_WR_p2_WL_p2_SB_Vel[i]

        hands_pair4 = np.zeros(len(p1_WL_p2_WL_Dist))
        for i in range(len(p1_WL_p2_WL_Dist)):
            if p1_WL_p2_WL_Dist[i] < 280:
                hands_pair4[i] = p1_WL_p2_WL_p1_SB_Vel[i] - p1_WL_p2_WL_p2_SB_Vel[i]

        push = hands_pair1 + hands_pair2 + hands_pair3 + hands_pair4

        push_toward_person1 = countAboveThreshold(-push, 0.3)
        push_toward_person2 = countAboveThreshold(push, 0.3)

    # C6 - Gaze

    
    person1_HL_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_LEFT X")
    person1_HL_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_LEFT Y")
    person1_HL_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_LEFT Z")

    person1_HR_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_RIGHT X")
    person1_HR_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_RIGHT Y")
    person1_HR_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_RIGHT Z")

    person1_HB_pX = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_BACK X")
    person1_HB_pY = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_BACK Y")
    person1_HB_pZ = qtmfile.GetValueFrames(framestart, frameend, person1Name + "_HEAD_BACK Z")

    person2_HL_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_LEFT X")
    person2_HL_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_LEFT Y")
    person2_HL_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_LEFT Z")

    person2_HR_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_RIGHT X")
    person2_HR_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_RIGHT Y")
    person2_HR_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_RIGHT Z")

    person2_HB_pX = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_BACK X")
    person2_HB_pY = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_BACK Y")
    person2_HB_pZ = qtmfile.GetValueFrames(framestart, frameend, person2Name + "_HEAD_BACK Z")



    if len(person1_HL_pX) == 0 or len(person1_HL_pY) == 0 or len(person1_HL_pZ) == 0 or \
        len(person1_HR_pX) == 0 or len(person1_HR_pY) == 0 or len(person1_HR_pZ) == 0 or \
        len(person1_HB_pX) == 0 or len(person1_HB_pY) == 0 or len(person1_HB_pZ) == 0 or \
        len(person2_HL_pX) == 0 or len(person2_HL_pY) == 0 or len(person2_HL_pZ) == 0 or \
        len(person2_HR_pX) == 0 or len(person2_HR_pY) == 0 or len(person2_HR_pZ) == 0 or \
        len(person2_HB_pX) == 0 or len(person2_HB_pY) == 0 or len(person2_HB_pZ) == 0:
        print("*** Warning: head position vector is empty - file %s" % trialFileName)
        continue
    else:

        person1_HC_pX = (person1_HL_pX + person1_HR_pX) / 2
        person1_HC_pY = (person1_HL_pY + person1_HR_pY) / 2
        person1_HC_pZ = (person1_HL_pZ + person1_HR_pZ) / 2

        person2_HC_pX = (person2_HL_pX + person2_HR_pX) / 2
        person2_HC_pY = (person2_HL_pY + person2_HR_pY) / 2
        person2_HC_pZ = (person2_HL_pZ + person2_HR_pZ) / 2

        p1_gazeat_p2 = np.zeros(len(person1_HC_pX))
        p2_gazeat_p1 = np.zeros(len(person1_HC_pX))

        for i in range(len(person1_HC_pX)):
            person1_gaze = np.array([person1_HC_pX[i] - person1_HB_pX[i], \
                                 person1_HC_pY[i] - person1_HB_pY[i], \
                                 person1_HC_pZ[i] - person1_HB_pZ[i]])

            person2_gaze = np.array([person2_HC_pX[i] - person2_HB_pX[i], \
                                 person2_HC_pY[i] - person2_HB_pY[i], \
                                 person2_HC_pZ[i] - person2_HB_pZ[i]])

            p1_p2_dir = np.array([person2_HB_pX[i] - person1_HB_pX[i], \
                              person2_HB_pY[i] - person1_HB_pY[i], \
                              person2_HB_pZ[i] - person1_HB_pZ[i]])

            p1_gazeat_p2[i] = py_ang(person1_gaze, p1_p2_dir)
            p2_gazeat_p1[i] = 3.141 - py_ang(person2_gaze, p1_p2_dir)

        gaze_toward_p1 = countAboveBelowThreshold(p1_gazeat_p2, p2_gazeat_p1, 0.95, 0.95)

        gaze_toward_p2 = countAboveBelowThreshold(p2_gazeat_p1, p1_gazeat_p2, 0.95, 0.95)


    # write all the features in the output file
    
    outputfile.Write([  secondColumn, framestart, frameend, leader, person1_W_Stab_Count, \
                        person2_W_Stab_Count, person1_A_Stab_Count, person2_A_Stab_Count, \
                        p1_p2_W_Ant_Count, p2_p1_W_Ant_Count, p1_p2_A_Ant_Count, p2_p1_A_Ant_Count, \
                        push_toward_person1, push_toward_person2, gaze_toward_p1, gaze_toward_p2])
    # filtered = lowpassfilter.filter(unfiltered)
    
    # person1_WL_vX = signal.savgol_filter(person1_WL_pX, 51, 2, 1)
    # signal_vY = signal.savgol_filter(signal_pY, 51, 2, 1)
    # signal_vZ = signal.savgol_filter(signal_pZ, 51, 2, 1)

    # signal_v = (signal_vX ** 2 + signal_vY ** 2 + signal_vZ ** 2) ** 0.5

    # signal_KE = signal_vX ** 2 + signal_vY ** 2 + signal_vZ ** 2

    # filtered2 = lowpassfilter.filter(filtered)

    # plt.plot(signal_KE)
    # plt.figure()
    # plt.plot(filtered)

    # plt.plot(filtered[lowpassfilter.filter_n//2:-lowpassfilter.filter_n//2])
    # plt.plot(filtered2[lowpassfilter.filter_n:])
    # plt.show()


segmentsfile.Close()
outputfile.Close()

