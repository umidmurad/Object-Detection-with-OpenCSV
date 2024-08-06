from __future__ import print_function
import pandas as pd #for CSV export
import numpy as np
import csv
import cv2 as cv
import video 
from common import anorm2, draw_str
from decimal import * #For recall and precision decimal value


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15), #size of the search window at each pyramid level.
                  maxLevel = 2, #size of the search window at each pyramid level. 
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) #parameter, specifying the termination criteria of the iterative search algorithm
 
# params for ShiTomasi corner detection 
feature_params = dict( maxCorners = 500,    #Specifying maximum number of corners as 500
                       qualityLevel = 0.3, # 0.3 is the minimum quality level below which the corners are rejected
                      minDistance = 7, # 7 is the minimum euclidean distance between two corners
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5 
        self.tracks = [] #this keeps the movement of the fish as an arrray
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
         
 
    def run(self):
        while True:
            _ret, frame = self.cam.read() #reading video
            if _ret:    
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #method is used to convert an video to Gray scaled video
                vis = frame.copy() #copy of the frame for later use
                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray 
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) #Calculates an optical flow using Lucas-Kanade method and
                    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params) #comparing to the next frame to find the change of pixel values to
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)  
                    print(d)                                      #verify indeed the change is good enough to be saved.
                    good = d < 1 #absolute value less than 1 is good data 
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue #this line breaks from the loop, which makes the circles dissapear.
                        tr.append((x, y))
                        if len(tr) > self.track_len:    #after it reaches 10th frame it deletes the first frame. 
                            del tr[0]                   #this deletion allows us to run the video smoothly without stopping.
                        new_tracks.append(tr)           #collects the movement data
                        cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)  #shows the tracked fish's head
    

                    pd.DataFrame(new_tracks).to_csv('MOVEMENT.csv') #CSV EXPORTING 
                    self.tracks = new_tracks
                    cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0)) #draws traces based on the movement stored in tr array
                    getcontext().prec = 5 #decimal count of Precision and Recall
                    draw_str(vis, (20, 20), 'Aprox. ZebraFish count: 713')
                    draw_str(vis, (20, 40), 'Live ZebraFish count: %d' % len(self.tracks))
                    #713 is aprx. amount of ZebraFish which is used to find Recall and Precision
                    draw_str(vis, (20, 60), 'Precision : ' +str(((Decimal(713+1)/Decimal(len(self.tracks)))*100)) +'%'  ) 
                    draw_str(vis, (20, 80), 'Recall : ' +str(((Decimal(len(self.tracks)+1)/Decimal(713+1))*100) ) +'%'  )
                    
                if self.frame_idx % self.detect_interval == 0:          # this if statement will trace the zebrafishes and
                    mask = np.zeros_like(frame_gray)                    # it will assign point (dots) for every fish to keep track of them 
                    mask[:] = 255                                       #assignment of the points happen based on the data we have collected.
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv.circle(mask, (x, y), 10, 0, -1) 
                    p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])
    
    
                self.frame_idx += 1
                self.prev_gray = frame_gray
                
                vis = cv.resize(vis, (960, 540)) #resizing the video output 
                cv.imshow('ZebraFish Counter', vis) #shows the video with visual changes
                
                ch = cv.waitKey(1) #ESC ends the video 
                if ch == 27:
                    break
            else:
                break # when video ends we are out of loop
                
 
def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()
    print('Done')
 
 
if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()