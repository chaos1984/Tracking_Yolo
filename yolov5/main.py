import time
import detect
import subbg
t_start = time.time()
print ("\n*****%s start*****" % (time.strftime("%X", time.localtime())))
video_path = r"C:/Yoking/01_Study/Yolo/Testing/DAB_1020/Front/"
files = glob(video_path + "*.avi")
files_name = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
for file in files:
    if "FRONT" in file:
        a = subbg.CushionTracking([file],target=False,mp=True,resolution=0.025)
        a.run()
    break   



opt = detect.parse_opt()
detect.main(opt)

timecost = round(time.time() - t_start,1)
print ("*****%.3fs *****\n" % (timecost)) 