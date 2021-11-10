# coding: utf-8
 
import numpy as np
import cv2
import shutil
import os

def FindFile(start, name):
	#Find the files whose name string contains str(name), and the directory of files
	isFlag = 0
	lenname = len(name)
	if  '*' in name:
		name = name.strip('*')
		isFlag =1
		lenname = len(name)-1
	relpath =[]
	files_set = []
	dirs_set = []

	if isFlag == 0:
		for relpath, dirs, files in os.walk(start):
			for i in files:
				if name == i[-lenname:] and i[0]!="_":
					full_path = os.path.join(start, relpath, i)
					files_set.append(os.path.normpath(os.path.abspath(full_path)))
					dirs_set.append(os.path.join(start, relpath))
						
	elif isFlag == 1:
		for relpath, dirs, files in os.walk(start):
			for i in files:
				if os.path.splitext(i)[-1] == name:
					full_path = os.path.join(start, relpath, i)
					files_set.append(os.path.normpath(os.path.abspath(full_path)))
					dirs_set.append(os.path.join(start, relpath))
	
	dirs_set = list(set(dirs_set))
	return files_set,dirs_set   

def getFrame(dir,module ="DAB",save_path=r"./save_each_frames_front"):
	num = 0
	for file in dir:
		num += 1
		cap = cv2.VideoCapture()
		print (file)
		cap.open(file)
		if cap.isOpened() != True:
			os._exit(-1)
		
		#get the total numbers of frame
		totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		print ("the number of frames is {}".format(totalFrameNumber))
		
		#set the start frame to read the video
		frameToStart = 1
		cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart);
		
		#get the frame rate
		rate = cap.get(cv2.CAP_PROP_FPS)
		print ("the frame rate is {} fps".format(rate))
		
		# get each frames and save
		frame_num = 0
		while True:
			ret, frame = cap.read()
			if ret != True:
				break
			img_path = save_path+ "//" +module+ str(num)+str(frame_num)+".jpg"
			print (img_path)
			cv2.imwrite(img_path,frame)
			frame_num = frame_num + 1
		
			# wait 10 ms and if get 'q' from keyboard  break the circle
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
			
		cap.release()


if __name__ == '__main__':
	print ("Start")
	print ("the opencv version: {}".format(cv2.__version__))
	module = "DAB"
	DocDir = r"./" 
	save_path = "./save_each_frames_front"
	os.mkdir(save_path)
	avi_list = FindFile(DocDir,"avi")
	print (avi_list[0])
	getFrame(avi_list[0],module,save_path)
	

