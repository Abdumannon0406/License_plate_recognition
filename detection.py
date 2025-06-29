from ultralytics import YOLO
import cv2,pytesseract

model=YOLO("pose.pt")
image_path="image_2025-06-26_13-11-47.png"
results=model.predict(image_path,save=True)
image=cv2.imread(image_path)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
results=results[0]
for i in results:
    xy=i.boxes.xyxy.cpu().numpy()[0]
    x1,y1,x2,y2=map(int,xy)
    cropped=image[y1:y2,x1:x2]
    cv2.imwrite("cropped.jpg",cropped)

    text=pytesseract.image_to_string(cropped)
    print(text)