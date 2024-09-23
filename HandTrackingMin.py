import cv2 #OpenCV library, used for real-time computer vision tasks
import mediapipe as mp #library for building machine learning pipelines for processing video, audio, and other streams of data
import time #library for time-related tasks

cap = cv2.VideoCapture(0) #Initializes the video capture object cap, using the camera at index 1. 
#index 1 - usually an external camera, index 0 - usually the built-in camera

mpHands = mp.solutions.hands # mediapipe hands
hands = mpHands.Hands() # default params
mpDraw = mp.solutions.drawing_utils

pTime = 0 # previous time
cTime = 0 # current time

try:
    while True: #loop runs indefinitely, constantly capturing frames from the camera
        success, img = cap.read() #captures a frame from the camera. It returns two values
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mediapipe uses RGB
        results = hands.process(imgRGB) # returns a dictionary
        # print(results.multi_hand_landmarks) # returns None if no hands detected

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # mpDraw.draw_landmarks(img, handLms) #Having just this only displays the landmarks
                for id, lm in enumerate(handLms.landmark): #id is the index and lm is the specific landmark
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    print(id, cx, cy)
                    if id == 0: # palm landmark
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draws the connections
        cTime = time.time()
        fps = 1/(cTime - pTime) # frames per second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img) #waits for 1 millisecond to allow the image to be displayed and checks if any key is pressed during that time. 
        #The 1 millisecond delay effectively makes the loop run as fast as possible, creating a real-time video stream
        cv2.waitKey(1) # 1ms delay

except KeyboardInterrupt:
    print("Exiting gracefully...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")