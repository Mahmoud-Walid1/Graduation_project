import streamlit as st
import cv2
import pickle
import numpy as np
import os

st.title("Add New Face Data")

st.write("""
This page allows you to add a new user's face to the system. 
Enter the name of the person and click 'Start Capture'. 
The camera will turn on and capture 100 images of the face. 
Please look directly at the camera and ensure good lighting.
""")

name = st.text_input("Enter Your Name:")

if st.button("Start Capture"):
    if not name:
        st.error("Please enter a name.")
    else:
        try:
            video = cv2.VideoCapture(0)
            if not video.isOpened():
                st.error("Could not open webcam. Please check if it is connected and not in use by another application.")
            else:
                facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
                faces_data = []
                i = 0
                
                st.info("Capturing images... Please wait.")
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()

                while len(faces_data) < 100:
                    ret, frame = video.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam.")
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = facedetect.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        crop_img = frame[y:y+h, x:x+w, :]
                        resized_img = cv2.resize(crop_img, (50, 50))
                        
                        # Append image every 10 frames to get some variation
                        if i % 10 == 0:
                            faces_data.append(resized_img)
                            
                        i += 1
                        
                        # Update progress
                        progress = len(faces_data)
                        progress_bar.progress(progress)
                        status_text.text(f"Captured: {progress}/100")

                        # Draw rectangle on the frame to show detection
                        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
                    
                    # Display the frame in the Streamlit app
                    frame_placeholder.image(frame, channels="BGR")

                    # Add a small delay to prevent crashing
                    cv2.waitKey(1)

                video.release()
                cv2.destroyAllWindows()

                if len(faces_data) == 100:
                    st.success("Image capture complete!")

                    faces_data_np = np.asarray(faces_data)
                    faces_data_np = faces_data_np.reshape(100, -1)

                    # Save names
                    if 'names.pkl' not in os.listdir('data/'):
                        names = [name] * 100
                        with open('data/names.pkl', 'wb') as f:
                            pickle.dump(names, f)
                    else:
                        with open('data/names.pkl', 'rb') as f:
                            names = pickle.load(f)
                        names = names + [name] * 100
                        with open('data/names.pkl', 'wb') as f:
                            pickle.dump(names, f)

                    # Save face data
                    if 'faces.pkl' not in os.listdir('data/'):
                        with open('data/faces.pkl', 'wb') as f:
                            pickle.dump(faces_data_np, f)
                    else:
                        with open('data/faces.pkl', 'rb') as f:
                            faces = pickle.load(f)
                        faces = np.append(faces, faces_data_np, axis=0)
                        with open('data/faces.pkl', 'wb') as f:
                            pickle.dump(faces, f)
                    
                    st.success("Face data saved successfully!")
                else:
                    st.error("Could not capture 100 face images. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
