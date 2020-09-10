import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIAU3PP4REQKF4JSSE5",
                        aws_secret_access_key="2GfjTB3sFJfxWiC6HwBKPKOVVP5Pnb5ihRTxb6O/",
                        aws_session_token="FwoGZXIvYXdzEGUaDHeBIzz1hwbBGo0W7CLFAQSAo1ikIeU7DmT2n+LxHFJ3f+0Xu2hhsflZOVGGpqu36k0jcGVVkRgH0Db3h7XkRw+V2F08QB4IZiIVo0+tDrVCBNYgDaFTNmZ2h1C38HBaKhTm/jIBlle4IaHr83Gwp39LZTfMdtgLzfrh8QZnpRQcqhpsxvc67KnNLQ0ti2bhqcCS001eeQF6vq5vB68/cqhhvVduJQ3nKcEIrDs5afnKAmRksiJYNmhKuDhiS0ZBF63dXDi90F2V2kHoz/5cL4bQ/itZKIOq6PoFMi0HWRoWJ1CeiVK1EphwqPwZa/IDyEzLRmm7AGN0YAl16TScZpwkARFz10zJTaM=",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:333899925792:project/MaskDetection/version/MaskDetection.2020-09-09T12.25.02/1599634503830',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://w4obezivl2.execute-api.us-east-1.amazonaws.com/Main123?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
