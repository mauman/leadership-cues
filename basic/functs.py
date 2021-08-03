import math
import cv2

def eucldist(p1,p2):
	return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def drawLineDash(frame,p0,p1,color,thickness):
	dx = p1[0] - p0[0]
	dy = p1[1] - p0[1]
	l = int(math.sqrt(dx**2+dy**2))
	drawFlag = False
	for i in range(0,l-20,40):
		p2 = (int(p0[0] + dx * (i / l)),int(p0[1] + dy * (i / l)))
		p3 = (int(p0[0] + dx * ((i + 20) / l)),int(p0[1] + dy * ((i + 20) / l)))
		cv2.line(frame, (p2[0], p2[1]), (p3[0], p3[1]), color, thickness)

	p2 = (int(p0[0] + dx * ((l - 20) / l)), int(p0[1] + dy * ((l - 20) / l)))
	p3 = (int(p0[0] + dx), int(p0[1] + dy))

	cv2.line(frame, (p2[0], p2[1]), (p3[0], p3[1]), color, thickness)

	cv2.circle(frame,(p0[0],p0[1]),thickness + 2, color, -(thickness + 2))
	cv2.circle(frame,(p1[0], p1[1]), thickness + 2, color, -(thickness + 2))

def timeStrToSeconds(s):
	ms = int(s[0:s.find(":")]) * 3600
	ms += int(s[s.find(":") + 1:s.rfind(":")]) * 60
	ms += int(s[s.rfind(":") + 1:])
	return ms

def timeSecondsToStr(t):
	hours = int(t // 3600)
	minutes = int((t % 3600) // 60)
	seconds = int(t % 60)
	return "%02d:%02d:%02d" % (hours, minutes, seconds)