import cv2
import numpy as np
import math

def make_img(conf, title):
    size = conf["size"]
    img = np.full((size*120, size*120, 3), 255, dtype=np.uint8)
    
    #write title and so on
    episode = "episode"+str(conf["episode"])
    cv2.putText(img, episode, (size*50, size*5), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), lineType=cv2.LINE_AA)
    
    cv2.putText(img, "0", (size*5, size*115), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), lineType=cv2.LINE_AA)
    cv2.putText(img, "10", (size*3, size*11), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), lineType=cv2.LINE_AA)
    cv2.putText(img, "10", (size*108, size*115), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), lineType=cv2.LINE_AA)
    
    #make goal
    goal_color = [(255,0,0),
                  (0,255,0),
                  (0,0,255),
                  (255,0,255)]
    
    cv2.rectangle(img, (size*10*10, size*10*1), (size*10*11, size*10*2), goal_color[0], thickness=-1)
    cv2.rectangle(img, (size*10*1, size*10*10), (size*10*2, size*10*11), goal_color[1], thickness=-1)
    cv2.rectangle(img, (size*10*1, size*10*1), (size*10*2, size*10*2), goal_color[2], thickness=-1)
    cv2.rectangle(img, (size*10*10, size*10*10), (size*10*11, size*10*11), goal_color[3], thickness=-1)
    
    #write grid
    for i in range(1, 12):
        cv2.line(img, (size*10*i, size*10), (size*10*i, size*110), (0,0,0), thickness=2)
        cv2.line(img, (size*10, size*10*i), (size*110, size*10*i), (0,0,0), thickness=2)
    
    #put agent
    agent_num = 2
    for i in range(conf["agent_num"]):
        stri = str(i+1)
        x, y = conf["agent"+stri+"_x"], conf["agent"+stri+"_y"]
        s_x, s_y = int((x+1)*10*size), int((12-y-1)*10*size)
        angle = conf["agent"+stri+"_angle"]
        put_agent(img, x, y, angle, size)
        cv2.putText(img, stri, (s_x-16, s_y+16), cv2.FONT_HERSHEY_COMPLEX, size*0.2, (255, 255, 255), lineType=cv2.LINE_AA)
        
        
    #save img
    cv2.imwrite(title, img)
    
def put_agent(img, x, y, angle, size):
    s_x, s_y = int((x+1)*10*size), int((12-y-1)*10*size)
    cv2.circle(img, (s_x, s_y), size*5, (0, 0, 0), thickness=-1)
    arrowlength = 70
    angle = math.radians(angle + 90)
    arrx, arry = int(math.sin(angle)*50 + s_x), int(math.cos(angle)*50 + s_y)
    cv2.arrowedLine(img, (s_x, s_y), (arrx, arry), (0, 0, 0), thickness=3)