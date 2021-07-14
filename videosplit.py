import cv2
vidcap = cv2.VideoCapture(r"C:\Users\laconicli\Documents\Tencent Files\1714469097\FileRecv\MobileFile\VID20210707163553.mp4",)
success, image = vidcap.read()
# image = cv2.resize(image, (360,240))
success=True
count=1
while success:
    image = cv2.resize(image, (360,240))
    cv2.imwrite("F:\download_code\FgSegNet\CDnet2014_dataset\mydata\mytable\input\in%06d.jpg" % count, image)
    count+=1
    success, image = vidcap.read()
