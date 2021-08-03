import os
from filemanagement.CSVFileReader import CSVFileReader
import numpy as np

def binarysearch(arr, n):
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        if arr[mid] < n:
            low = mid + 1
        elif arr[mid] > n:
            high = mid - 1
        else: 
            return mid
    return -1

class QTMTSVFileReader:
    def __init__(self, filename):
        print("opening %s..." % filename)
        reader = CSVFileReader(filename, "\t")
        print("reading column names...")
        self.column_names = [""]
        while self.column_names[0] != "MARKER_NAMES":
            self.column_names = reader.GetNext()
        
        self.column_names = reader.GetNext()
        
        print("reading column values...")
        self.column_values = []
        row = reader.GetNext()
        while row != []:
            self.column_values.append([float(v) for v in row])
            row = reader.GetNext()
        print("done!")
        reader.Close()
        self.framenumbers = []
        for row in self.column_values:
            self.framenumbers.append(row[0])

    def GetFrame(self, N):
        if N in self.framenumbers:
            return self.column_values[binarysearch(self.framenumbers, N)]
        else:
            return None
    
    def GetValueFrame(self, N, columnname):
        frame = self.GetFrame(N)
        if frame is not None and columnname in self.column_names:
            return frame[self.column_names.index(columnname)]

    def GetValueFrames(self, start, end, columnname):
        result = []
        if columnname in self.column_names:
            i = start
            frame = self.GetFrame(i)
            while frame is not None and i < end:
                result.append(frame[self.column_names.index(columnname)])
                i = i + 1
                frame = self.GetFrame(i)
        return np.asarray(result)