import cv2
import os
from glob import glob
import numpy as np


def cropSquare(img,size=(640,640)):
    w,h,c = img.shape
    l = min(w,h)
    dst = img[0:l, 0:l]
    return cv2.resize(dst,size)

    
def videocombine(filelist):
    num_video = len(filelist)
    video = []
    w = []
    h = []
    for index,file in enumerate(filelist):
        video.append(cv2.VideoCapture(file))
        w.append(int(video[index].get(cv2.CAP_PROP_FRAME_WIDTH)))
        h.append(int(video[index].get(cv2.CAP_PROP_FRAME_HEIGHT)))


    w = max(w)
    h = max(h)
    print (w,h)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = video[0].get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('out.mp4', fourcc, fps, (num_video*w,h))


    while video[0].isOpened():
        framelist = []
        for i in range(num_video):
            ret, frame = video[i].read()  # 捕获一帧图像
            frame = cv2.resize(frame,(w,h))
            framelist.append(frame)
        if ret:
            frame = np.hstack(tuple(framelist))
            out.write(frame)
        else:
            break
    for i in range(num_video):
        video[i].release()
    
    out.release()
    cv2.destroyAllWindows()    

def getFrame(dir,video_name,save_path=r"/save_frames/"):

    video = dir+video_name
    save_path = dir+save_path
    try:
         os.mkdir(save_path)
    except:
        pass
   
    cap = cv2.VideoCapture()
    # video = r'C:\Yoking\01_Study\Yolo\Testing\DAB_1020\Front\save_clip\T-18499715_FRONT_DAB_85.avi'
    cap.open(video)
    # if cap.isOpened() != True:
    #     os._exit(-1)
    
    #get the total numbers of frame
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print ("The number of frames is {}".format(totalFrameNumber))
    
    #set the start frame to read the video
    frameToStart = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
    
    #get the frame rate
    rate = cap.get(cv2.CAP_PROP_FPS)
    print ("the frame rate is {} fps".format(rate))
    
    # get each frames and save
    frame_num = 0
    while True:
        ret, frame = cap.read()
        
        if ret != True:
            break
        frame = cropSquare(frame,size=(640,640))
        img_path = save_path+ "//" +video_name.replace(".avi","_")+str(frame_num)+".jpg"
        print (img_path)
        cv2.imwrite(img_path,frame)
        frame_num = frame_num + 1
    
        # wait 10 ms and if get 'q' from keyboard  break the circle
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release() 

def videoclip(video_path,video_name,frameStart,frameEnd,scale,save_path=r"/save_clip/"): 
    save_path = video_path+save_path
    try:
        os.mkdir(save_path)
    except:
        pass
    video = cv2.VideoCapture(video_path+video_name)
    # 需要明确视频保存的格式
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    out = cv2.VideoWriter(save_path+video_name.replace(".avi","_clip.avi"), fourcc, fps, (w,h))
    # 设置视频截取的开始时间
    video.set(cv2.CAP_PROP_POS_FRAMES, frameStart)
    Maxcount = frameEnd - frameStart
    count = 0 
    while video.isOpened():
        ret, frame = video.read()  # 捕获一帧图像
        if ret:
            frame = cv2.resize(frame,(w,h))
            out.write(frame)
            count += 1
            if Maxcount == count:
                break
        else:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:/Yoking/01_Study/Yolo/Testing/DAB_1020/Front/"
    files = glob(video_path + "*.avi")
    files_name = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    # print (files)

    for i,video_dir in enumerate(files):
        print ("Video:",video_dir)
        video_name = files_name[i].replace(".avi","_clip.avi")
        

        videoclip(video_path,files_name[i],frameStart=50,frameEnd=70,scale=1,save_path=r"save_clip/")
        getFrame(video_path+r"save_clip/",video_name,save_path=r"save_frames/")
    