# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
from enum import EnumMeta
import os
from re import X
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch._C import wait
import torch.backends.cudnn as cudnn
import time

import matplotlib.pyplot as plt
from glob import glob
import subbg
import scipy.signal as signals
from scipy.optimize import curve_fit

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync



def cov1d(data,kernel = [0,0,0,1,1,1,1],thr = 0.90,offside=2):
    '''
    Summary: Conv kernel for check curve valley
    Author: Yujin Wang
    Date:  2021-12-19 
    Args:
        data[np.array]:1-d signal
    Return:
        check[list]: Index list
        res[list]: Signal after conv
    '''
    size = len(kernel)
    padding = data.shape[0]%size
    res = []
    for i in range(size - padding):
        data = np.append(data,0)
    check = []
    for index,v in enumerate(data):
        if index+size > data.shape[0]:
            for i in range(size - 1):
                res.append(0)
            break
        temp = data[index:index+size]
        
        if np.corrcoef(kernel,temp)[0][1] > thr:
            check.append(index+offside)
            res.append(data[index+offside])
    return check,np.array(res)
        

def deployment(data,path,deltaT=0.25,deployment_time=0):
    '''
    Summary: Calc open time and full time based on curve
    Author: Yujin Wang
    Date:  2021-12-23 
    Args: 
        data[np.array]: Curve data of seam, cushion and hub from YoloV5
        path[str]: Curve figure save path
        delta_T[float]: Interval time between 2 frames.
        deployment_time[float]: Start time of deployment.
    Return:
        fulltime[float]: Cushion full time
    '''
    dis = []
    area = []
    w = []
    h = []
    far = []
    seam_dis =[]
    seam_vis_index = 0
    datalen = len(data['cushion'])
    time = [ i*deltaT for i in range(datalen)]
    cushion_start = 0
    for i in range(datalen): #Find hub 
        if data['hub'][i] !=[]:
            hub = data['hub'][i][0:2]
    
    for i in range(datalen): # Extract data from AI
        if data['cushion'][i] !=[]  and i > deployment_time/deltaT:
            dis.append(calcDis(data['cushion'][i][0:2],hub))
            area.append(data['cushion'][i][2]*data['cushion'][i][3] )
            w.append(data['cushion'][i][2])
            h.append(data['cushion'][i][3])
            far.append(data['cushion'][i][0]+data['cushion'][i][2]/2.) 
        else:
            cushion_start += 1
            dis.append(0)  
            area.append(0)
            w.append(0)
            h.append(0)
            far.append(0)
        if data['seam'][i] !=[] :
            seam_dis.append(calcDis(data['seam'][i][0:2],hub))
        else:
            seam_dis.append(0) 
    # Check time
    title = ''
    fig = plt.figure(figsize=(8,6)) #新建画布
    ax = plt.subplot(1,1,1) #子图初始化
    # plt.scatter(seamflag_time,np.array(seamflag_dis)/max(dis),marker="o",c='red')

    ax.scatter(deployment_time,0,marker="o",c='blue')
    title += "Open time:"+str(deployment_time) +"ms  "
    
    # cover_open_time = time[cushion_start-int(0.75/deltaT)] 
    # if cover_open_time < deployment_time :
    #     cover_open_time = time[int((deployment_time/deltaT + cushion_start)/2)]

    
    seam_dis = signals.medfilt(np.array(seam_dis)/max(seam_dis),3)
    plotCurve(ax,time,seam_dis,legend="Seam_dis",showflag = False)
    dis_filter = signals.medfilt(np.array(dis)/max(dis))

    plotCurve(ax,time,dis_filter,legend="Distance",showflag = False)

    # Check seam vsiable
    for i,v in enumerate(seam_dis):
        if v != 0 :
            seam_vis_index = i
            break
    if seam_vis_index == 0:
        full_checkflag = dis.index(max(dis))
        title += "No seam found!"
        full_time = 999
    else:
        full_checkflag = max(dis.index(max(dis)),seam_vis_index)
        # title += "T_d:"+str(cover_open_time)+"ms  "
        # ax.scatter(cover_open_time,0,marker="o",c='pink')
        # Check cushiong Min. displacement
        if seam_dis[full_checkflag] ==0:
            seam_dis[seam_dis==0]=2
            min_dis = dis_filter[seam_dis.argmin()]
        else:    
            min_dis = min(dis_filter[full_checkflag:])
  
        # Get min. value of dis
        for i,v in enumerate(dis_filter):
            if min_dis == v:
                full_time = time[i]
                break

       # check valley
        check,res  = cov1d(dis_filter,kernel = [0,0,1,1,1],thr = 0.90,offside=2)
        check1,_ = cov1d(res,kernel = [1,0,1],thr = 0.5,offside=1)


        # check_time = [time[i] for i in check];check_dis = [dis_filter[i] for i in check]
        # ax.scatter(check_time,check_dis,marker="o",c='black')
        check_time = [time[check[i]] for i in check1];check_dis = [dis_filter[check[i]] for i in check1]
        ax.scatter(check_time,check_dis,marker="s",c='black')
        
        
        title += "Full time:"+str(full_time)+"ms"
        ax.scatter(full_time,min_dis,marker="o",c='red')
    
    ax.set_title(title)

    print ("Curveplot fig is saved!",path.replace("avi","jpg"))
    plt.savefig(path.replace("avi","jpg"))
    plt.close()
    return full_time
    # plt.show()


def calcDis(p1,p2):
    '''
    Summary: Calc distance between points
    Author: Yujin Wang
    Date:  2021-12-23 
    Args:
        p1,p2[list]: Coordination of p1,p2
    Return:
        dist[float]: Distance between p1 and p2
    '''
    return np.sqrt(np.power(p1[0]-p2[0],2) + np.power(p1[1]-p2[1],2))



def plotCurve(ax,x,y,legend="Curve",xlabel="Time(ms)",ylabel="Value",showflag = True):
    '''
    Summary: Plot curve
    Author: Yujin Wang
    Date:  2021-12-23 
    Args:
    Return:
    '''
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(x,y,label=legend)
    ax.legend()
    ax.grid('on')
    if showflag == True:
        plt.show()

@torch.no_grad() # No grade calculation in Interfer
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        framestart = 40,
        frameend   = 300
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir  = Path(project)
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults



    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt,framestart=framestart,frameend=frameend)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    # Initial boxinfo
    boxinfo = {}
    for j in names:
        boxinfo[j] = []
        ii = 0
    for path, img, im0s, vid_cap in dataset:

        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)

        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions



        for i, det in enumerate(pred):  # per image
            for j in names:
                boxinfo[j].append([])
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    object_name = names[int(cls)]
                    boxinfo[object_name][-1] = ([xywh[0],xywh[1],xywh[2],xywh[3]])
                    # boxinfo(xywh,names,cls)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                            # Plot curve for dischange               
                            
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        ii += 1
        if ii == frameend:
            break 
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        
    return boxinfo,save_path

def parse_opt(video,video_path,framestart=40,frameend=300):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)') # 指定模型路径
    parser.add_argument('--source', type=str, default=video, help='file/dir/URL/glob, 0 for webcam') # 选择录像
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w') # 32 倍数，图像尺寸，默认无影响
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold') # 置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold') #iou 阈值
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image') #最大检测目标数量，默认
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # 选择GPU 还是 cpu
    parser.add_argument('--view-img', action='store_true', help='show results') # 默认
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') # 默认
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') # 默认
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes') # 默认
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos') # 默认
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3') ## 默认
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') # 默认
    parser.add_argument('--augment', action='store_true', help='augmented inference') # 默认
    parser.add_argument('--visualize', action='store_true', help='visualize features')# 默认
    parser.add_argument('--update', action='store_true', help='update all models')# 默认
    parser.add_argument('--project', default=video_path, help='save results to project/name')# 默认
    # parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')# 默认
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')# 默认
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')# 默认
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')# 默认
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')# 默认
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')# 默认
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')# 默认
    # Parameters for deployment video clip
    parser.add_argument('--framestart',  type=int, default=framestart, help='Crop frame start')# 视频起始帧
    parser.add_argument('--frameend',  type=int, default=frameend, help='Crop frame end')# 视频结束帧
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    return run(**vars(opt))
    

if __name__ == "__main__":
    t_start = time.time() # 统计时间
    print ("\n*****%s start*****" % (time.strftime("%X", time.localtime()))) # 打印开始时间
    
    # video_path = r"C:/Yoking/01_Study/Yolo/Testing/DigitalVideo/85_5/"
    # video_path = r"C:/Yoking/01_Study/Yolo/Testing/DigitalVideo/85_1/"
    # video_path = r"C:/Yoking/01_Study/Yolo/Testing/DigitalVideo/85_2/"
    # video_path = r"C:/Yoking/01_Study/Yolo/Testing/DigitalVideo/-35_1/"
    # video_path = r"C:/Yoking/01_Study/Yolo/Testing/DigitalVideo/85_9/"
    # video_path = r"C:/Yoking/01_Study/Yolo/Testing/DigitalVideo/23_4/"
    video_path = r"C:/Users/yujin.wang/Desktop/DAB_deployment_video/LT/"  # 录像路径
    predict = [] # 初始化预测结果，针对多个T号视频录像，用于程序初期的调试
    videolist  = glob(video_path+"/*") # 获得录像列表
    # videolist = [r'C:/Users/yujin.wang/Desktop/DAB_deployment_video/HT/85_14/']  # 如果只针对一个T号，可以输入一个录像的路径
    for videoPath in videolist:   #循环预测多个T号下的气袋

        print (videoPath)         #打印T录像路径
        videoPath += '/'          #补全T录像路径
        # files = glob(video_path + "*.avi")
        detect_path = videoPath + r'detect/'  #创建推测文件夹，如果存在则pass
        try:
            os.mkdir(detect_path)
        except:
            pass
        files = glob(videoPath + "*.avi") #获得T号下录像
        for file in files:                #循环获得录像文件
            if "FRONT" in file or "REAR" in file: #如果是正面DAB点爆录像，则采用帧差法进行目标追踪，获得
                front = subbg.CushionTracking([file],target=False,mp=True)
                front.run() 
                time1 = [i*0.25 for i in range(len(front.histcor))]
                plt.plot(time1,front.histcor)
                plt.savefig(file.replace("avi","png"))
                plt.close()
                # print ("DAB deployment time: %f s" %(a.deployment_time))
            else:
                pass
        for file in files:
            if "SIDE" in file:
                    opt = parse_opt(file,detect_path,framestart=front.offframe,frameend=300)
                    boxinfo,save_path = main(opt)
            else:
                pass
        t1 = deployment(boxinfo,save_path,deltaT=front.delta_t,deployment_time=front.deployment_time)
        filename = videoPath.replace("\\", "/").split("/")[-2]
        # for comparision with testing
        predict.append([filename,front.deployment_time,t1]) 
        
        timecost = round(time.time() - t_start,1)
        print ("*****%.3fs *****\n" % (timecost)) 
    predict = pd.DataFrame(predict)
    predict.to_csv(video_path+'predict.csv')
        