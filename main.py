import cv2
import tkinter as tk
from PIL import Image, ImageTk

class ConfidenceTracker:
    def __init__(self, video_source=0):
        self.root = tk.Tk()
        self.root.title("Employee Confidence Tracker")

        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(self.root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.delay = 10  # milliseconds

        self.eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smileCascade =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        print(cv2.data.haarcascades)
        self.update()

        self.root.mainloop()

        

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face = self.faceCascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            #face
            for (x, y, w, h) in face:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Confidence: hello', (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            #eyes
            eyes = self.eyeCascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            smile = self.smileCascade.detectMultiScale(frame_gray, scaleFactor=3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in smile:
                cv2.rectangle(frame, (x, y), (x-w, y-h), (255, 0, 0), 2)
            
              
            

           
               
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.update)

if __name__ == "__main__":
    app = ConfidenceTracker()
