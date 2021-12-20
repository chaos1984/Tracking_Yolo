import cv2
import os
import numpy as np
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import time
from multiprocessing.pool import ThreadPool
from collections import deque
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks,peak_prominences,medfilt
import pytesseract




def cv_show(name, img):
    '''
    Summary: image show
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        name[str]:window label
        img[np.arry]:img array
    Return:
    '''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,640, 640)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def OCR(img):
    '''
    Summary: Extract the vedio information
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        img[np.array]
    Return:
        fps
        offframe
    '''
    # cv_show('1',img)
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 210, 255, cv2.THRESH_BINARY_INV)[1]
    # cv_show('1',ref)
    text = pytesseract.image_to_string(ref, lang='eng')
    text = text.replace("—", "-")
    text = text.replace("\n", " ")
    text = text.replace("：", ":")
    text = text.split(":")
    text = [i.strip().split() for i in text]
    try:
        while text[0][0].isalpha():
            text[0].pop(0)
        return int(text[0][0]), abs(int(text[1][0]))

    except:
        return 4000, 40



def extractMxyContour(c):
    '''
    Summary: Extract center of box
    Author: Yujin Wang
    Date:  2021-12-19 
    Args: 
        c[torch.box]:box
    Return:
        (center_x,center_y)
    '''
    M = cv2.moments(c)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def distance(x, y):
    '''
    Summary: Calc distance
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        x[float]
        y[float]
    Return:
        dis[float]
    '''
    return np.sqrt(x*x+y*y)


def normalize(data):
    '''
    Summary: Normalization for data
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        data[np.array]
    Return:
        norm_dis[np.array]
    '''
    try:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    except:
        print(data)


def pyrDown(currentframe, pyrDownNum):
    '''
    Summary: Down samples
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        currentframe[np.array]
        pyrDownNum[int]:number
    Return:
        rentframe[np.array]
    '''
    for i in range(pyrDownNum):
        currentframe = cv2.pyrDown(currentframe)
    return currentframe


class CushionTracking():
    '''
    Summary: Tracking cushion by subbing background
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        Class
    Return:
        incidense = CushionTracking
    '''
    def __init__(self, filedir, target=False, mp=True):
        '''
        Summary: Initialization 
        Author: Yujin Wang
        Date:  2021-12-19 
        Args:
            filedir[str]:dirctory of file
            target[str]:testing res
            mp[bool]:multi-process
        Return:
        '''
        self.filedir = filedir
        self.plottarget = target
        self.threaded_mode = mp
        self.pyrDownNum = 2 # 2 down sample 
        self.scale = np.power(2, self.pyrDownNum)
        self.FlagRecord = True
        # self.offframe = int(abs(self.time_offset)/self.delta_t)

        self.t0 = 0

    def grayhist(self,img):
        '''
        Summary: Calchist from gray img.
        Author: Yujin Wang
        Date:  2021-12-19 
        Args: 
            img[file]:png
        Return:
            grayhist[np.arry]
        '''
        img_roi = img[int(self.fheight*0.4):int(self.fheight*0.6),int(self.fwidth*0.4):int(self.fwidth*0.6)] 
        img_roi = cv2.resize(img_roi,(640,640), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        
        # _, binary = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
        # cv_show('binary',binary)
        # cv_show('gray',gray)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        Normhist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX, -1)
        return np.array([i[0] for i in Normhist])

    def run(self):
        '''
        Summary: run main process
        Author: Yujin Wang
        Date:  2021-12-19 
        Args:
        Return:
        '''
        for file in self.filedir:
            self.filename = os.path.basename(file)
            print("\n*****%s start*****" %
                  (time.strftime("%X", time.localtime())))
            print(self.filename)
            t_start = time.time()
            # print (file)
            self.outVideo = file.replace(self.filename, "_"+self.filename)
            self.outImage = file.replace(
                self.filename, self.filename[:-4]+".jpg")

            # self.target = self.VideoDir + "\\"+file.split("\\")[-1].replace("avi","txt")
            if self.plottarget == True:
                self.getTarget
            self.area_list = []
            self.contour = []
            self.contour_id = []
            self.multiProccessing(file)
            self.end = self.time_offset + self.frames_num*self.delta_t
            self.frametime = np.linspace(
                self.time_offset, self.end, int(self.frames_num)+1)
            # print ("START:",self.time_offset,"END:",self.end,"NUM:",int(self.frames_num),"DELTA_T:",self.delta_t)
            # print ("tIME:",self.time_offset,"END:",self.end,"NUM:",int(self.frames_num),"DELT_T:",self.delta_t)
            self.timecost = round(time.time() - t_start, 1)
            # self.curveplot()
            print("*****%s END*****" % (time.strftime("%X", time.localtime())))
            print("*****%.3fs *****\n" % (self.timecost))
            # time.sleep(0.3)

    def getVideoInfo(self, frame):
        '''
        Summary: Get video information
        Author: Yujin Wang
        Date:  2021-12-19 
        Args:
            frame
        Return:
        '''
        
        fshape = frame.shape
        self.fheight = fshape[0]
        self.fwidth = fshape[1]
        self.bggrayhist = self.grayhist(frame)

        # self.text1 = OCR(frame[self.fwidth:self.fheight-int(1/2.*(self.fheight-self.fwidth+1)),0:self.fwidth])
        self.rate, self.offframe = OCR(
            frame[int(10*self.fheight/11):self.fheight, 0:self.fwidth])
        self.delta_t = 1./self.rate*1000
        self.time_offset = self.offframe*self.delta_t
        print("**Figure data**")
        print("Delta_t: %f Offset frame: %d" % (self.delta_t, self.offframe))

    def frame0(self, frame):
        '''
        Summary: Get video from frame0
        Author: Yujin Wang
        Date:  2021-12-19 
        Args:
            frame[]:frame0
        Return:
        '''
        self.getVideoInfo(frame)

        # self.videoWriter = cv2.VideoWriter(self.outVideo,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),30,(self.fwidth,self.fheight))
        self.frameArea = self.fheight*self.fwidth / \
            np.power(2, 2*self.pyrDownNum)
        self.bg = pyrDown(frame, self.pyrDownNum)
        self.bg = cv2.bilateralFilter(
            self.bg, d=0, sigmaColor=100, sigmaSpace=15)

    def multiProccessing(self, video_file):
        '''
        Summary: Multi process 
        Author: Yujin Wang
        Date:  2021-12-19 
        Args:
            video_file
        Return:
        '''
        self.discheck = np.array([])
        self.flagterminat = False
        self.cap = cv2.VideoCapture(video_file)
        self.frames_num = int(self.cap.get(7))-1
        print('Frame number of movie:', self.frames_num)
        ret, frame = self.cap.read()
        self.frame0(frame)

        # Multi_thread
        threadn = cv2.getNumberOfCPUs()
        pool = ThreadPool(processes=threadn)
        pending = deque()
        print("CPU", threadn)
        res_list = [0]

        frame_id = 0
        self.histcor = [1]
        while(True):
            frame_id += 1
            # if (frame_id > 10./self.delta_t+ abs(self.offframe)):
            #     break
            ret, frame = self.cap.read()
            if (frame_id < abs(self.offframe)+1):
                continue
            
            self.curgrayhist = self.grayhist(frame)
            self.corr = cv2.compareHist(self.curgrayhist, self.bggrayhist,method=cv2.HISTCMP_CORREL)
            # print (self.corr)
            self.histcor.append(self.corr)
            if (self.corr < 0.99):
                self.deployment_time = (frame_id-abs(self.offframe)) * self.delta_t
                break
        #     while len(pending) > 0 and pending[0].ready():
        #         res = pending.popleft().get()
        #         res_list.append(res)
        #     if len(pending) < threadn:

        #         if self.threaded_mode:
        #             try:
        #                 task = pool.apply_async(
        #                     self.frameprocess, (frame.copy(), frame_id))
        #             except:
        #                 break
        #             pending.append(task)
        #         else:
        #             # ret, frame = self.cap.read()
        #             res = self.frameprocess(frame.copy(), frame_id)

        #             # cv2.imshow('threaded video', res)
        #             res_list.append(res)
        #     if cv2.waitKey(150) & 0xff == 27:
        #         break
        # self.cap.release()
        # cv2.destroyAllWindows()

    def frameprocess(self, frame, frame_id):
        '''
        Summary: By substractting bg
        Author: Yujin Wang
        Date:  2021-12-19 
        Args:
        Return:
        '''
        frame_copy = pyrDown(frame, self.pyrDownNum)

        frame_blur = cv2.bilateralFilter(
            frame_copy, d=0, sigmaColor=100, sigmaSpace=15)
        score, gra, diff = ssim(
            self.bg, frame_blur, full=True, gradient=True, gaussian_weights=True, win_size=3)
        diff = (diff*255).astype('uint8')
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        index_list = []

        for index, c in enumerate(contours):
            # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
            # 找出图像矩
            try:
                cX, cY = extractMxyContour(c)

                contourArea = cv2.contourArea(contours[index])
                if contourArea > self.frameArea/4:
                    break
                # print (contourArea)
                if cY < self.fheight/5.:

                    # print (cY ,fheight/5)
                    areas.append(contourArea)
                    index_list.append(index)
            except:
                pass
            # 在图像上绘制轮廓及中心
        if len(areas) > 0:
            if self.FlagRecord:
                self.frame0id = frame_id

                self.FlagRecord = False
            self.deployment_time = (self.frame0id-self.offframe)*self.delta_t

            self.flagterminat = True

        else:
            time.sleep(0)

        return frame

    def drawContour(self, frame, c, ratio):
        epsilon = ratio*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(frame, [approx*self.scale], -1, (0, 255, 0), 1)

    def drawMinareBox(self, frame, c):
        # 得到最小矩形的坐标
        rect = cv2.minAreaRect(c)
        #
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box*self.scale], -1, (0, 255, 0), 1)
        return box

    def drawRect(self, frame, c):
        x, y, w, h = cv2.boundingRect(c*self.scale)

        cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h),
                      color=(255, 255, 255), thickness=3)

    def drawPoint(self, frame, x, y):
        cv2.circle(frame, (x*self.scale, y*self.scale), 10, (0, 0, 255), 0)


if __name__ == "__main__":
    VideoDir = [r"C:\Yoking\01_Study\Yolo\Testing\DAB_1020\Front\T-21340633_FRONT_DAB_-35.avi"]
    # VideoDir = [r"C:\Yoking\01_Study\Yolo\Testing\DAB_1020\Front\T-20408870_FRONT_DAB_85.avi"]
    # VideoDir = [r"C:\Yoking\01_Study\Yolo\Testing\DAB_1020\Front\T-20408870_FRONT_DAB_85.avi"]
    # VideoDir = [r"C:\Yoking\01_Study\Yolo\Testing\DAB_1020\Front\T-20408870_FRONT_DAB_85.avi"]
    # VideoDir = [r"C:\Yoking\01_Study\Yolo\Testing\DAB_1020\Front\T-20408870_FRONT_DAB_85.avi"]

    # VideoDir = [r"C:\Yoking\01_Study\Yolo\Testing\DAB_1020\Front\T-19157806_FRONT_DAB_85.avi"]
    for file in VideoDir:
        print(file)
        a = CushionTracking([file], target=False, mp=True)
        a.run()
        print("Deployment start time:", a.deployment_time)
        # print (a.deployment_time)
