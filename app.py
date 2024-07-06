import streamlit as st
import cv2
from easyocr import Reader
import numpy as np
from matplotlib import pyplot as plt

st.title('Handwritten Text Recognition')
st.header('Welcome to HTR model')

# --------------------------------------------------------------------------OCR
def recognize_handwritten_text(image_path):
    image_path = 'captured_frame.png'
    reader = Reader(['en'], gpu=True)
    result = reader.readtext(image_path)

    img = cv2.imread(image_path)
    spacer = 100
    font = cv2.FONT_HERSHEY_SIMPLEX

    for detection in result:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        text = detection[1]
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
        img = cv2.putText(img, text, (25, spacer), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        spacer += 20


    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, result

#----------------------------------------------------------------------------StreamlitUI
show_camera_feed = False
image_path = 'captured_frame.png'

if st.button("Get Image"):
    st.write("Instructions:")
    st.write("- Press 'S' to start/stop the camera feed.")
    st.write("- Press 'P' to capture the frame.")
    st.write("- Press 'O' to exit the camera.")
    
        #--------------------------------------------------------------------------------------------------------------------------CAMERA 
    

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Camera not found")
        exit()

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

   
    show_camera_feed = False

    while True:
        if show_camera_feed:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read a frame from the camera")
                break

            cv2.imshow("Camera Feed", frame)

        
        key = cv2.waitKey(1)

       
        if key == ord('o'):
            break

     
        if key == ord('s'):
            show_camera_feed = not show_camera_feed

        
        if key == ord('p'):
            if show_camera_feed:
        
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite("captured_frame.png", frame)
                    print("Frame saved as 'captured_frame.png'")
    cap.release()
    cv2.destroyAllWindows()
    #--------------------------------------------------------------------------------------------------------------------------CAMERA 


if st.button("Recognize"):
        st.image(image_path, caption="Captured Frame", use_column_width=True)
        st.write("Recognizing...")

        recognized_img, recognition_result = recognize_handwritten_text(image_path)
        st.image(recognized_img, caption="Recognized Text", use_column_width=True)
        st.write("Recognition Result:")
        for result in recognition_result:
            st.write(f"Text: {result[1]}, Confidence: {result[2]:.2f}")


