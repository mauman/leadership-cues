from .functs import eucldist
import numpy as np
import cv2

jointsNames = {"Nose":0, "Neck":1, "RShoulder":2, "RElbow":3, "RWrist":4, "LShoulder":5,
               "LElbow":6, "LWrist":7, "MidHip":8, "RHip":9, "RKnee":10, "RAnkle":11, "LHip":12, "LKnee":13,
               "LAnkle":14, "REye":15, "LEye":16, "REar":17, "LEar":18, "LBigToe":19, "LSmallToe":20,
               "LHeel":21, "RBigToe":22, "RSmallToe":23, "RHeel":24, "Background":25}

class Person:
    name = ""
    closest_d = -1
    closest_p = None
    coords = []
    flagged = False
    color = (255,255,255)
    BR = None
    cog = None

    def __init__(self,coords = None):
        if not coords is None:
            self.coords = coords
        else:
            self.coords = []
        self.prev_coords = []


    def __lt__(self,p):
        if self.closest_d == -1 and p.closest_d != -1:
            return False
        if p.closest_d == -1 and self.closest_d != -1:
            return True
        if self.closest_d <= p.closest_d:
            return True
        else:
            return False

    def distanceFrom(self,p2):
        self.distance = eucldist(self.getJoint("Neck"),p2.getJoint("Neck"))
        self.d_from = p2
        return self.distance

    def getJoint(self,name):
        joint = jointsNames[name]
        if joint != None:
            return (self.coords[joint][0],self.coords[joint][1])
        else:
            return (0,0)

    def getBoundingRectangle(self):

        minx = -1
        maxx = -1
        miny = -1
        maxy = -1

        for jointCoord in self.coords:
            if jointCoord[2] > 0:
                if jointCoord[0] < minx or minx == -1:
                    minx = jointCoord[0]
                if jointCoord[0] > maxx or maxx == -1:
                    maxx = jointCoord[0]
                if jointCoord[1] < miny or miny == -1:
                    miny = jointCoord[1]
                if jointCoord[1] > maxy or maxy == -1:
                    maxy = jointCoord[1]

        self.BR = [minx, miny, maxx, maxy]
        self.cog = [int((minx + maxx) / 2.0), int((miny + maxy) / 2.0)]

        return self.BR



    def findClosest(self,plist,threshold):
        closest_p = None
        closest_d = -1
        for person in plist.getList():
            if self.distanceFrom(person) < closest_d or closest_p == None:
                closest_d = self.distanceFrom(person)
                closest_p = person
        if closest_d < threshold:
            self.closest_d = closest_d
            self.closest_p = closest_p
    #if closest_p != None:
    #	closest_p.flagged = True

    def updateCoords(self,coords):
        self.prev_coords = self.coords
        self.coords = coords

    def getKineticEnergy(self):
        energy = 0
        if len(self.prev_coords) > 0:
            for i in range(1,2):
                energy += 0.5*eucldist(self.coords[i],self.prev_coords[i])**2
        return energy

    def fromString(self,s):
        try:
            tokens = s.replace(" ", "").split(",")
            self.name = tokens[0]
            coords = np.ndarray(shape=(25, 3), dtype=float, order='F')
            counter = 1
            for i in range(25):
                for j in range(2):
                    coords[i][j] = float(tokens[counter])
                    counter += 1
                if coords[i][0] != 0 and coords[i][1] != 0:
                    coords[i][2] = 1
                else:
                    coords[i][2] = 0
            self.coords = coords
        except:
            print("exception reading file\n")

    def draw(self,gc):
        if self.coords != []:
            for i in range(25):
                cv2.circle(gc, (int(self.coords[i][0]), int(self.coords[i][1])), 1, self.color, -2)
                self.drawBone(gc, "Neck", "RShoulder")
                self.drawBone(gc, "Neck", "REar")
                self.drawBone(gc, "Neck", "LEar")
                self.drawBone(gc, "Neck", "LShoulder")
                self.drawBone(gc, "RShoulder", "RElbow")
                self.drawBone(gc, "RElbow", "RWrist")
                self.drawBone(gc, "LShoulder", "LElbow")
                self.drawBone(gc, "LElbow", "LWrist")
                self.drawBone(gc, "Neck", "MidHip")
                self.drawBone(gc, "MidHip", "RHip")
                self.drawBone(gc, "MidHip", "LHip")
                self.drawBone(gc, "RHip", "RKnee")
                self.drawBone(gc, "RKnee", "RAnkle")
                self.drawBone(gc, "LHip", "LKnee")
                self.drawBone(gc, "LKnee", "LAnkle")


    def drawBone(self,gc,fromJoint,toJoint):
        condition = int(self.coords[jointsNames[fromJoint]][0]) != 0\
                    and int(self.coords[jointsNames[fromJoint]][1]) != 0\
                    and int(self.coords[jointsNames[toJoint]][0]) != 0 \
                    and int(self.coords[jointsNames[toJoint]][1]) != 0

        if condition:
            cv2.line(gc, (int(self.coords[jointsNames[fromJoint]][0]), int(self.coords[jointsNames[fromJoint]][1])),
                 (int(self.coords[jointsNames[toJoint]][0]), int(self.coords[jointsNames[toJoint]][1])),
                 self.color, 1)

    def setColor(self,color):
        self.color = color


# // Result for BODY_25 (25 body parts consisting of COCO + foot)
# // const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
# //     {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
# //     {8,  "MidHip"},
# //     {9,  "RHip"},
# //     {10, "RKnee"},
# //     {11, "RAnkle"},
# //     {12, "LHip"},
# //     {13, "LKnee"},
# //     {14, "LAnkle"},
# //     {15, "REye"},
# //     {16, "LEye"},
# //     {17, "REar"},
# //     {18, "LEar"},
# //     {19, "LBigToe"},
# //     {20, "LSmallToe"},
# //     {21, "LHeel"},
# //     {22, "RBigToe"},
# //     {23, "RSmallToe"},
# //     {24, "RHeel"},
# //     {25, "Background"}
# // };