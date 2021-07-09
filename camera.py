#%% 
import cv2
#%%
def capture():
    camera = cv2.VideoCapture(0)
    # camera.set(3, width)
    # camera.set(4, height)
    while True:
        ret,frame = camera.read()
        cv2.imshow('press Y to save',frame)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            cv2.imwrite('face.jpg',frame)
            cv2.destroyAllWindows()
            break
    # print(type(frame))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera.release()
    return frame
