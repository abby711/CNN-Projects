try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract
import cv2
import time
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def rectext(filename):
    text=pytesseract.image_to_string(Image.open(filename))
    return text

def captureimage():
    cam=cv2.VideoCapture(0)
    time.sleep(1)
    while True:
        _,img=cam.read()
        cv2.imshow("CAMERA-FEED",img)
        key=cv2.waitKey(1)&0xFF
        if key== ord("x"):
            cv2.imwrite("image.jpg",img)
            cam.release()
            cv2.destroyAllWindows()
            break
captureimage()
info=rectext('image.jpg')
print(info)
file=open("result.txt","w")
file.write(info)
file.close()
print("written successfully")
