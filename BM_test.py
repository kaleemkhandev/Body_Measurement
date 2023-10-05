#import sys
#sys.path.insert(0, './pre-trained_model')

from tkinter import LEFT
from tkinter.tix import CheckList
from turtle import right
import cv2
import os
import torch
from PIL import Image, ImageDraw
import numpy as np

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import math


# Given a video capture object, read frames from the same and convert it to RGB
def grab_frame(img_path):
    frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return frame

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def ellipse_circ(a, b):
    #c = math.pi * (a+b) * ((3 * ((a-b)**2)) / ( (a+b)**2) )
    c = math.pi* (3*(a+b) - math.sqrt((3*a + b) * (a + 3*b))) 
    return c

class Inference_Handler():
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seg_model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
        self.seg_model.to(self.device).eval()
        print("here: ",self.device)

    def get_pred(self, img):
        # See if GPU is available and if yes, use it
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define the standard transforms that need to be done at inference time
        imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
        preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                        std  = imagenet_stats[1])])
        input_tensor = preprocess(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Make the predictions for labels across the image
        with torch.no_grad():
            output = self.seg_model(input_tensor)["out"][0]
            output = output.argmax(0)

        # Return the predictions
        return output.cpu().numpy()


    def infer(self, img_path):
        landmark_data = []
        frame_data = []
        # while cap.isOpened():
        #img_name = img_path.split(" ")[-1].split("(")[1].split(")")[0]
        #print("image:", img_name)
            
        frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        #print("image size: ", frame.shape)
        ratio = frame.shape[0]/frame.shape[1]
        
        #cv2.imshow('check',cv2.imread(img_path))
        #frame = cv2.resize()
        #frame = cv2.resize(frame,(1080,1920))

        detection_landmarks = self.pose.process(frame)

        traverse = detection_landmarks.pose_landmarks.landmark

        #mp_drawing.draw_landmarks(
        #    frame,
        #   detection_landmarks.pose_landmarks,
        #    mp_pose.POSE_CONNECTIONS,
        #    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        shoulder_left_x = 0
        shoulder_right_y = 0
        shoulder_left_y = 0
        shoulder_right_x = 0
        nose_x = 0
        nose_y = 0
        heel_x = 0
        heel_y = 0
        hip_left_x = 0
        hip_left_y = 0
        hip_right_x = 0
        hip_right_y = 0
        mouth_left_x = 0
        mouth_left_y = 0
        mouth_right_y = 0
        mouth_right_x = 0
        right_mid_neck = 0
        left_mid_neck = 0
        left_end_neck = 0
        right_end_neck = 0
        eye_left_y = 0
        eye_left_x = 0
        eye_right_y = 0
        eye_right_x = 0
        
        line_thickness = 2

        #Get landmarks
        #if traverse:
        for i,ld in enumerate(traverse):
            if i == 0:
                nose_x = int(ld.x *frame.shape[1])
                nose_y = int(ld.y *frame.shape[0])
                nose_z = ld.z
            elif i == 7:
                ear_left_y = int(ld.y *frame.shape[0])
                ear_left_x =  int(ld.x *frame.shape[1])
            elif i == 3:
                left_eye_x = int(ld.x *frame.shape[1])
                left_eye_y = int(ld.y *frame.shape[0])
            elif i == 29:
                heel_x = int(ld.x *frame.shape[1])
                heel_y = int(ld.y *frame.shape[0])
                heel_z = ld.z# *frame.shape[1])
            elif i == 30:
                r_heel_x = int(ld.x *frame.shape[1])
            elif i == 32:
                r_toes_x = int(ld.x *frame.shape[1])
            elif i == 11:
                shoulder_right_x = int(ld.x * frame.shape[1])
                shoulder_right_y = int(ld.y * frame.shape[0])
                shoulder_right_z = ld.z
            elif i == 12:
                shoulder_left_x = int(ld.x * frame.shape[1])
                shoulder_left_y = int(ld.y * frame.shape[0]) 
            elif i == 23:
                hip_right_x = int(ld.x * frame.shape[1])
                hip_right_y = int(ld.y * frame.shape[0])
            elif i == 24:
                hip_left_x = int(ld.x * frame.shape[1])
                hip_left_y = int(ld.y * frame.shape[0])
            elif i == 10:
                mouth_left_y = int(ld.y * frame.shape[0])
                mouth_left_x = int(ld.x * frame.shape[1])
            elif i == 15:
                wrist_left_y = int(ld.y *frame.shape[0])
                wrist_left_x = int(ld.x *frame.shape[1])
            elif i == 13:
                elbow_left_y = int(ld.y * frame.shape[0])
                elbow_left_x = int(ld.x * frame.shape[1])
            elif i == 9:
                mouth_right_x = int(ld.x * frame.shape[1])
                mouth_right_y = int(ld.y * frame.shape[0])
            elif i == 3:
                eye_left_x = int(ld.x * frame.shape[1])
                eye_left_y = int(ld.y * frame.shape[0])
            elif i == 4:
                eye_right_x = int(ld.x * frame.shape[1])
                eye_right_y = int(ld.y * frame.shape[0])


        
        
        angle = ""

        if (r_heel_x-r_toes_x) > 0:
            angle = "front"
        else:
            angle = "back"
    

        w = abs(shoulder_right_x - shoulder_left_x)
        h = abs(hip_left_y - shoulder_left_y)

        if (w/h) <0.25:
            angle = "side"

        print("Pose:", angle, "w/h:", w/h)


        if angle=="back":
            right_mid_navel = int(shoulder_left_y + abs(hip_left_y - shoulder_left_y)*0.6)
            left_mid_navel = int(shoulder_right_y + abs(hip_right_y - shoulder_right_y)*0.6)

            temp_left_mid_chest = (shoulder_right_y + right_mid_navel)//2
            temp_right_mid_chest = (shoulder_left_y + left_mid_navel)//2
            left_mid_chest = (shoulder_right_y + temp_right_mid_chest)//2
            right_mid_chest = (shoulder_left_y + temp_left_mid_chest)//2

            left_mid_neck = int( (mouth_left_y + abs(mouth_left_y - shoulder_left_y)*0.4) )
            right_mid_neck = int( (mouth_right_y + abs(mouth_right_y - shoulder_right_y)*0.4) )

        elif angle == "side":
            left_mid_navel = int(shoulder_left_y + abs(hip_left_y - shoulder_left_y)*0.6)
            right_mid_navel = int(shoulder_right_y + abs(hip_right_y - shoulder_right_y)*0.6)

            right_mid_chest = (shoulder_right_y + right_mid_navel)//2
            left_mid_chest = (shoulder_left_y + left_mid_navel)//2

            left_mid_neck = (mouth_left_y + shoulder_left_y)//2
            right_mid_neck = int(mouth_right_y +  abs((mouth_left_y  - shoulder_right_y) *0.3))

            lower_chest_mid_left = int ( left_mid_chest + abs(left_mid_chest - shoulder_left_y)*0.5 )
            lower_chest_mid_right = int ( right_mid_chest + abs(right_mid_chest - shoulder_right_y)*0.5 ) 




        else:
            left_mid_navel = int(shoulder_left_y + abs(hip_left_y - shoulder_left_y)*0.6)
            right_mid_navel = int(shoulder_right_y + abs(hip_right_y - shoulder_right_y)*0.6)

            temp_right_mid_chest = (shoulder_right_y + right_mid_navel)//2
            temp_left_mid_chest = (shoulder_left_y + left_mid_navel)//2

            right_mid_chest = int((shoulder_right_y + abs(shoulder_right_y - temp_right_mid_chest) *0.9 ))
            left_mid_chest = int((shoulder_left_y + abs(shoulder_left_y - temp_left_mid_chest) *0.9 ))

            left_mid_neck = int( (mouth_left_y + abs(mouth_left_y - shoulder_left_y)*0.4) )
            right_mid_neck = int( (mouth_right_y + abs(mouth_right_y - shoulder_right_y)*0.4) )
        


        
            

        #print(left_mid, right_mid)
        red = [0,0,255]
       

        width, height, channels = frame.shape
        small_frame = cv2.resize(frame,(1080,1920))
        labels = self.get_pred(small_frame)
        
        labels = np.array(labels, dtype='uint8')
        #print(type(labels))
        labels = cv2.resize(labels, (height, width))
        # Wherever there's empty space/no person, the label is zero 
        # Hence identify such areas and create a mask (replicate it across RGB channels)
        mask = labels == 0
        mask = np.repeat(mask[:, :, np.newaxis], channels, axis = 2)
        
        # Apply the Gaussian blur for background with the kernel size specified in constants
        blur_value = (51, 51)
        blur = cv2.GaussianBlur(frame, blur_value, 0)
        frame[mask] = blur[mask]

        #ax1.set_title("Blurred pic")

        # Set the data of the two images to frame and mask values respectively

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 8))
        ax2.set_title("Mask")

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Create two image objects to picture on top of the axes defined above
        im1 = ax1.imshow(frame)
        im2 = ax2.imshow(frame)
        im3 = ax3.imshow(frame)

        masked = mask*255

        #Overall height
        if angle == "side":
            top=0
            for i in reversed(range(0, ear_left_y)):
                if masked[i][ear_left_x][0] !=0:
                    top = i
                    break

            cv2.circle(frame, (ear_left_x,top), radius=0, color=(0, 0, 255), thickness=2)
            
            height = heel_y - top

        else:
            top=0
            for i in reversed(range(0,nose_y)):
                if masked[i][nose_x][0] !=0:
                    top = i
                    break
            cv2.circle(frame, (nose_x,top), radius=0, color=(0, 0, 255), thickness=2)
            cv2.circle(frame, (nose_x,heel_y), radius=0, color=(0, 0, 255), thickness=2)
            #cv2.imshow("height",frame)
            #cv2.waitKey(0)
            height = heel_y - top
            

        wrist_front = 0
        

        wrist_x_right = wrist_left_x

        #if img_name=='3':
        if wrist_left_y < left_mid_neck and angle!='back':
            while masked[wrist_left_y][wrist_x_right][0] == 0:
                wrist_x_right = wrist_x_right + 1

            wrist_x_left = wrist_left_x

            while masked[wrist_left_y][wrist_x_left][0] == 0:
                wrist_x_left = wrist_x_left -1

            cv2.line(frame, (wrist_x_left, wrist_left_y), (wrist_x_right, wrist_left_y), color=(0, 0, 255), thickness=2 )

            #wrist_front = cv2.norm((wrist_x_bottom, wrist_y_bottom), (wrist_x_top, wrist_y_top))
            wrist_front = abs(wrist_x_left - wrist_x_right)    


            chest_left = 0
            chest_right = 0
            
            shoulder_mid = (shoulder_left_x + shoulder_right_x) //2

            for i in reversed(range(0,shoulder_mid)):
                if masked[left_mid_chest][i][0]!=0:
                    chest_left = i
                    break

            for i in range(shoulder_mid, int(masked.shape[1])):
                if masked[right_mid_chest][i][0] != 0:
                    chest_right = i
                    break


            cv2.line(frame, (chest_left, left_mid_chest), (chest_right, right_mid_chest), (255, 255, 0), thickness=line_thickness)
            chest = abs(chest_right-chest_left)

            #lower chest:
            lower_chest_left = 0
            lower_chest_right = 0

            lower_chest_mid_left = int ( left_mid_chest + abs(left_mid_chest - shoulder_left_y)*0.3 )
            lower_chest_mid_right = int ( right_mid_chest + abs(right_mid_chest - shoulder_right_y)*0.3 )

            for i in reversed(range(0,shoulder_mid)):
                if masked[lower_chest_mid_left][i][0]!=0:
                    lower_chest_left = i
                    break

            for i in range(shoulder_mid, int(masked.shape[1])):
                if masked[lower_chest_mid_right][i][0] != 0:
                    lower_chest_right = i
                    break

            cv2.line(frame, (lower_chest_left, lower_chest_mid_left), (lower_chest_right, lower_chest_mid_right), (255, 0, 0), thickness=line_thickness)
            lower_chest = abs(lower_chest_right-lower_chest_left)


        wrist_side = 0
        hip_shoulder_mid_y = int ((shoulder_right_y + abs(shoulder_right_y - hip_right_y)*0.3))
        

        if wrist_left_y < hip_shoulder_mid_y and wrist_left_y > left_mid_neck and angle!='back':
        #if img_name == '4':
            wrist_side_top = wrist_left_y
            wrist_side_bottom = wrist_left_y
            
            print(wrist_side_bottom, wrist_side_top)

            while masked[wrist_side_top][wrist_left_x][0] == 0:
                wrist_side_top = wrist_side_top - 1
            
            while masked[wrist_side_bottom][wrist_left_x][0] == 0:
                wrist_side_bottom = wrist_side_bottom + 1
            

            chest_left = 0
            chest_right = 0

            shoulder_mid = (shoulder_left_x + shoulder_right_x) //2

            for i in reversed(range(0,shoulder_mid)):
                if masked[left_mid_chest][i][0]!=0:
                    chest_left = i
                    break

            for i in range(shoulder_mid, int(masked.shape[1])):
                if masked[right_mid_chest][i][0] != 0:
                    chest_right = i
                    break

            cv2.line(frame, (wrist_left_x, wrist_side_bottom), (wrist_left_x, wrist_side_top), color=(0, 0, 255), thickness=2 )
            wrist_side = abs(wrist_side_top - wrist_side_bottom)


            cv2.line(frame, (chest_left, left_mid_chest), (chest_right, right_mid_chest), (255, 255, 0), thickness=line_thickness)
            chest = abs(chest_right-chest_left)

            #lower chest:
            lower_chest_left = 0
            lower_chest_right = 0

            lower_chest_mid_left = int ( left_mid_chest + abs(left_mid_chest - shoulder_left_y)*0.3 )
            lower_chest_mid_right = int ( right_mid_chest + abs(right_mid_chest - shoulder_right_y)*0.3 )

            for i in reversed(range(0,shoulder_mid)):
                if masked[lower_chest_mid_left][i][0]!=0:
                    lower_chest_left = i
                    break

            for i in range(shoulder_mid, int(masked.shape[1])):
                if masked[lower_chest_mid_right][i][0] != 0:
                    lower_chest_right = i
                    break
            

            cv2.line(frame, (lower_chest_left, lower_chest_mid_left), (lower_chest_right, lower_chest_mid_right), (255, 0, 0), thickness=line_thickness)
            lower_chest = abs(lower_chest_right-lower_chest_left)




        left_end = 0
        right_end = 0

        left_end_neck = 0
        right_end_neck = 0

        left_end_chest = 0
        right_end_chest = 0

        if angle=="side":
            for i in range(0,int(masked.shape[1])):
                if masked[right_mid_navel][i][0] == 0:
                    #print(" right_end_navel: ",i)
                    left_end = i
                    break
            
            for i in reversed(range(0,masked.shape[1])):
                if masked[left_mid_navel][i][0] == 0:
                   #print(" left_end_navel: ",i)
                    right_end = i
                    break
            
            for i in reversed(range(0,int(masked.shape[1]))):
                if masked[right_mid_chest][i][0] == 0:
                    right_end_chest = i
                    break
            
            for i in range(0,right_end_chest):
                if masked[left_mid_chest][i][0] == 0:
                    left_end_chest = i
                    break

            for i in range(0,shoulder_left_x):
                if masked[lower_chest_mid_left][i][0]==0:
                    lower_chest_left = i
                    break

            for i in reversed(range(shoulder_right_x, int(masked.shape[1]))):
                if masked[lower_chest_mid_right][i][0] == 0:
                    lower_chest_right = i
                    break

            
            #neck
            for i in range(shoulder_right_x, int(masked.shape[1])):
                if masked[right_mid_neck][i][0] != 0:
                    right_end_neck = i
                    break
            
            for i in reversed(range(0,shoulder_left_x)):
                if masked[left_mid_neck][i][0] != 0:
                    left_end_neck = i
                    break
            

        elif angle == "back":
        #elif img_name == "0":
            for i in reversed(range(0,hip_right_x)):
                if masked[right_mid_navel][i][0] != 0:
                    left_end = i
                    break
            
            for i in range(hip_left_x, int(masked.shape[1])):
                if masked[left_mid_navel][i][0] != 0:
                    right_end = i
                    break
            
            #neck
            for i in range(shoulder_right_x, int(masked.shape[1])):
                if masked[left_mid_neck][i][0] == 0:
                    left_end_neck = i
                    break
            

            for i in reversed(range(0, shoulder_left_x)):
                if masked[left_mid_neck][i][0] == 0:
                    right_end_neck = i
                    break


            #suit
            back_right = shoulder_right_x + int(abs(shoulder_left_x-shoulder_right_x)*0.75)
            back_top = 0

            for i in reversed(range(0, shoulder_right_y)):
                if masked[i][back_right][0] != 0:
                    back_top = i
                    break

            #shoulder width
            wy = shoulder_right_y
            wx = shoulder_right_x
            while masked[wy-1][wx-1][0] == 0:
                  wy = wy-1
                  wx = wx-1


            wx_r = 0
            for i in range(wx, int(masked.shape[1])):
                if masked[wy][i][0] != 0:
                    wx_r = i
                    break
            
            
        
        else:
            for i in range(hip_right_x,int(masked.shape[1])):
                if masked[right_mid_navel][i][0] != 0:
                    right_end = i
                    break
    
            
            for i in reversed(range(0,hip_left_x)):
                if masked[left_mid_navel][i][0] != 0:
                    left_end = i
                    break
            

            #neck
            for i in range(mouth_right_x, int(masked.shape[1])):
                if masked[right_mid_neck][i][0] != 0:
                    right_end_neck = i
                    break
            
            for i in reversed(range(0, mouth_left_x)):
                if masked[left_mid_neck][i][0] != 0:
                    left_end_neck = i
                    break

            hip_mid_point = (hip_left_x + hip_right_x)//2
            hip_mid_end = 0

            for i in range(hip_left_y, int(masked.shape[0])):
                if masked[i][hip_mid_point][0] != 0:
                    hip_mid_end = i
                    break
            
            


            
        if angle=="side":
            waist_level_left = int(left_mid_navel + (hip_left_y - left_mid_navel) * 0.7)
            waist_level_right = int(right_mid_navel+ (hip_right_y - right_mid_navel) * 0.7)
        else:
            waist_level_left = (left_mid_navel + hip_left_y)//2
            waist_level_right = (right_mid_navel + hip_right_y)//2

        
        #cv2.circle(frame, (shoulder_left_y, shoulder_left_x), radius=0, color=(0, 0, 255), thickness=2)





        waist_right_end = 0
        waist_left_end = 0
        for i in range(int(hip_left_x), int(masked.shape[0])):
            if masked[waist_level_left][i][0] != 0:
                waist_right_end = i
                break
        
        for i in reversed(range(0,int(hip_right_x))):
            if masked[waist_level_right][i][0] != 0:
                waist_left_end = i
                break

        line_thickness=2


        cv2.line(frame, (left_end, left_mid_navel), (right_end, right_mid_navel), (0, 255, 0), thickness=line_thickness)
        cv2.line(frame, (waist_left_end, waist_level_left), (waist_right_end, waist_level_right), (255, 255, 0), thickness=line_thickness)
        
        if angle == "back":
            suit_left_y = int(hip_left_y + abs(waist_level_left - hip_left_y)*0.5) 
            cv2.line(frame, (back_right, back_top), (back_right, suit_left_y), (255, 0, 255), thickness=line_thickness)
            suit = abs(back_top - suit_left_y)

            cv2.line(frame, (wx, wy), (wx_r, wy) , color=(0, 0, 255), thickness=2)
            shoulder_width = abs(wx-wx_r)




        
        if elbow_left_y > nose_y and angle != "back":
            cv2.line(frame, (left_end_neck, left_mid_neck), (right_end_neck, right_mid_neck), (0, 255, 0), thickness=line_thickness)


        waist = abs(waist_right_end-waist_left_end)
        navel = abs(right_end - left_end)

        if angle == "side":
            cv2.line(frame, (left_end_chest, left_mid_chest), (right_end_chest, right_mid_chest), (255, 0, 0), thickness = line_thickness)
            chest = abs(right_end_chest-left_end_chest)


            cv2.line(frame, (lower_chest_left, lower_chest_mid_left), (lower_chest_right, lower_chest_mid_right), (255, 0, 0), thickness=2)
            lower_chest = abs(lower_chest_right-lower_chest_left)


            #outer leg
            cv2.line(frame, (hip_left_x, waist_level_left), (hip_left_x, heel_y), (0, 0, 255), thickness=line_thickness)
            outer_leg = abs(waist_level_left-heel_y)
           

        #elif img_name == '5':
        '''elif elbow_left_y < nose_y:
            chest_left = 0
            chest_right = 0

            for i in range(0,shoulder_left_x):
                if masked[left_mid_chest][i][0]==0:
                    chest_left = i
                    break

            for i in reversed(range(shoulder_right_x, int(masked.shape[1]))):
                if masked[right_mid_chest][i][0] == 0:
                    chest_right = i
                    break


            cv2.line(frame, (chest_left, left_mid_chest), (chest_right, right_mid_chest), (255, 0, 0), thickness=line_thickness)
            chest = abs(chest_right-chest_left)

            #lower chest:
            lower_chest_left = 0
            lower_chest_right = 0

            lower_chest_mid_left = int ( left_mid_chest + abs(left_mid_chest - shoulder_left_y)*0.67 )
            lower_chest_mid_right = int ( right_mid_chest + abs(right_mid_chest - shoulder_right_y)*0.67 )

            for i in range(0,shoulder_left_x):
                if masked[lower_chest_mid_left][i][0]==0:
                    lower_chest_left = i
                    break

            for i in reversed(range(shoulder_right_x, int(masked.shape[1]))):
                if masked[lower_chest_mid_right][i][0] == 0:
                    lower_chest_right = i
                    break

            cv2.line(frame, (lower_chest_left, lower_chest_mid_left), (lower_chest_right, lower_chest_mid_right), (255, 0, 0), thickness=line_thickness)
            lower_chest = abs(lower_chest_right-lower_chest_left)'''




       
        if angle!="side":

            if angle == 'front':
                cv2.line(frame, (hip_mid_point, waist_level_left), (hip_mid_point, hip_mid_end), (0, 255, 255), thickness=line_thickness)

             #inner leg:
            inner_leg_top = 0
            mid_leg = (hip_left_x + hip_right_x)//2
            for i in range(hip_left_y, frame.shape[0]):
                if masked[i][mid_leg][0] != 0:
                    inner_leg_top = i
                    break
            
            cv2.line(frame, (mid_leg, inner_leg_top), (heel_x, heel_y), (255, 0, 255), thickness=line_thickness)
            inner_leg = abs(inner_leg_top-heel_y)

            #front length
            if angle=="front" and wrist_left_y > nose_y:# and img_name !="0":
                front_right = shoulder_left_x + int(abs(shoulder_left_x-shoulder_right_x)*0.80)
                front_top = 0
                for i in reversed(range(0, shoulder_right_y)):
                    if masked[i][front_right][0] != 0:
                        front_top = i
                        break
                cv2.line(frame, (front_right, front_top), (front_right, waist_level_right), (255, 0, 255), thickness=line_thickness)
                front_length = abs(front_top - waist_level_right)


            #sleeve
            #if img_name!="3" and img_name!="5":
            if wrist_left_y > left_mid_neck: # and elbow_left_y > nose_y:
                cv2.line(frame, (shoulder_right_x, shoulder_right_y), (wrist_left_x, wrist_left_y), (255, 0, 0), thickness=line_thickness)
            sleeve = math.sqrt(abs(shoulder_right_y-wrist_left_y)**2 + abs(shoulder_right_x-wrist_left_x)**2) 


            #bicep:
            bicep = (shoulder_right_x + elbow_left_x)//2 

            bicep_top = 0
            bicep_bottom = 0

            for i in reversed(range(0, elbow_left_y)):
                if masked[i][bicep][0] != 0:
                    bicep_top = i
                    break
            
            for i in range(elbow_left_y, masked.shape[0]):
                if masked[i][bicep][0] !=0:
                    bicep_bottom = i
                    break
            

            #if img_name == "3" or img_name == "4":
            if hip_shoulder_mid_y > wrist_left_y and elbow_left_y > nose_y:
                cv2.line(frame, (bicep, bicep_top), (bicep, bicep_bottom), (255, 0, 0), thickness=line_thickness)
            b_f = abs(bicep_bottom - bicep_top)
            bicep_side = b_f


            bicep_front_bottom = 0
            bicep_front_top = 0
            t = (shoulder_right_y+elbow_left_y)//2

            
            for i in reversed(range(0, elbow_left_x)):
                if masked[t][i][0] == 0:
                    bicep_front_top = i
                    break

            x = t
            y = bicep_front_top

            masked = masked.astype(np.uint8)
            cv2.circle(masked, (y,x), radius=0, color=(255, 0, 255), thickness=2)


            for i in range(y, masked.shape[0]):
                x = x+1
                y = y-1
                if masked[x][y][0] != 0:
                    break

            #if img_name == "6":
            if angle == 'front' and wrist_left_y > hip_shoulder_mid_y:
                cv2.line(frame, (y, x), (bicep_front_top, t), (0, 0, 255), thickness=line_thickness)
            b = math.sqrt((y-bicep_front_top)**2 +  (x-t)**2)
            bicep_front = b

            

        #lower hips:
        lower_hip_left = 0
        lower_hip_right = 0

        for i in reversed(range(0,hip_right_x)):
            if masked[hip_right_y][i][0] != 0:
                lower_hip_left = i
                break
            
        for i in range(hip_left_x, masked.shape[1]):
            if masked[hip_left_y][i][0] != 0:
                lower_hip_right = i
                break
        
        
        cv2.line(frame, (lower_hip_left, hip_left_y), (lower_hip_right, hip_right_y), (0, 0, 255), thickness=line_thickness)
        lower_hip = abs(lower_hip_left - lower_hip_right)


        
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        im1.set_data(frame)
        im2.set_data(masked)

        mp_drawing.draw_landmarks(
            frame,
           detection_landmarks.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        im3.set_data(frame)

        
        cm_height = 189

        print(f"Height in cm: {cm_height}")

        print("Height in pixels:", height)

        if angle!="side":

            if angle == 'front':
                drop = abs(waist_level_left - hip_mid_end)
                print("Drop:", (cm_height*drop)/height, "cm")

            print("sleeve: ", (cm_height*sleeve)/height, "cm")

            b = (cm_height*b)/height
            b_f = (cm_height*b_f)/height

            print("bicep front: ", b, "cm")

            #if img_name == "3" or img_name == "4":
            if hip_shoulder_mid_y > wrist_left_y and elbow_left_y > nose_y:
                print("bicep side: ", b_f,"cm")

            if angle == "front" and wrist_left_y > nose_y:# and img_name!="0":
                print("front length:", (cm_height*front_length)/height, "cm")

            print("Inner Leg:",(cm_height*inner_leg)/height, "cm" )


        if angle=="side":
            print("outer leg:", (cm_height*outer_leg)/height, "cm")
        
        #if img_name == '0':
        elif angle == 'back':
            print("shoulder width:", (cm_height*shoulder_width)/height, "cm")
            suit_cm =  (cm_height*suit)/height
            print("suit:",suit_cm)

        
       
        print("lower hip:", (cm_height*lower_hip)/height)
        print("neck:",(cm_height*(abs(right_end_neck-left_end_neck)))/height)


        cm_waist = (cm_height*waist)/height 
        cm_navel = (cm_height*navel)/height 

        #if img_name == "5" or img_name == '2':
        if angle == 'side':
            cm_chest = (cm_height*chest)/height 
            print(f"Chest: {cm_chest}")
            
            cm_lower_chest = (cm_height*lower_chest)/height
            print(f"lower chest: {cm_lower_chest}") 

        elif angle == 'front' and wrist_left_y < hip_shoulder_mid_y:
            cm_chest = (cm_height*chest)/height
            print(f"Chest: {cm_chest}")

            cm_lower_chest = (cm_height*lower_chest)/height
            print(f"lower chest: {cm_lower_chest}") 
        else:
            cm_chest = 0


        if angle == "side":
            print(f"Upper hips: {cm_waist} cm")
            print(f"Waist in cm: {cm_navel} cm")
        elif angle == "back":
            print(f"Upper hips: {cm_waist} cm")
            print(f"Waist: {cm_navel} cm")
            
        else:
            print(f"Upper hips: {cm_waist} cm")
            print(f"Waist: {cm_navel} cm")


        #if img_name == '3':
        if wrist_left_y < shoulder_left_y and elbow_left_y > nose_y:
            wrist_front_cm = (cm_height*wrist_front)/height
            print("wrist front:", wrist_front_cm, "cm")

        #if img_name == '4':
        if wrist_left_y < hip_shoulder_mid_y and wrist_left_y > nose_y:
            wrist_side_cm = (cm_height*wrist_side)/height
            print("wrist side:", wrist_side_cm, "cm")


        print("\n")

    
        plt.show()
        plt.pause(0.01)
       
        cv2.imwrite("results/Basit new results/neck_labelled_"+img_path.split("/")[-1], cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite("results/Basit new results/mask_of_" + img_path.split("/")[-1], masked)
        return [cm_chest, cm_navel, cm_waist, angle]
        
if __name__ == "__main__":


    front = []
    back = []
    side = []
    handler = Inference_Handler()

    pics = "Abdul basit new"
    for i in os.listdir(pics):
       path = pics+"/"+i
       meas = handler.infer(path)
       
       if meas[-1] == "side":
        side.append(meas[:3])
       else:
        front.append(meas[:3])

        

    #path = "Images for neck girth/05.jpg"
    #handler.infer(path)
    #handler.infer("Images for neck girth/06.jpg", "back")
    #handler.infer("Images/01.jpg") 