'''
@Author: haoMax
@Github: https://github.com/liuzehao
@Blog: https://blog.csdn.net/liu506039293
@Date: 2019-09-23 11:36:53
@LastEditTime: 2019-11-14 17:27:19
@LastEditors: haoMax
@Description: 
'''
import tensorflow as tf
import os
import numpy as np
import cv2

from numba import jit
IMAGE_SIZE = 448
out_pb_path = "./frozen_model.pb"
test_path='./logs'
bias=50###这个值注意一下是倾斜程度,如果有的图比较偏可以设置再高一点
def freeze_graph_test(img):
    img=cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
    valid_images = np.ndarray((1, IMAGE_SIZE, IMAGE_SIZE,3))
    valid_images[0,:]=img
    pred = sess.run(pred_annotation, feed_dict={image: valid_images,keep_probability: 1.0})
    #utils.save_image(pred[0].astype(np.uint8), FLAGS.logs_dir, name="pred")
    #cv2.imwrite("a.png",pred[0].astype(np.uint8)*50)
    # cv2.imshow("a",pred[0].astype(np.uint8)*50)
    # cv2.waitKey(0)
    return pred[0].astype(np.uint8)*50
def find_head(imga):
    for i in range(1,IMAGE_SIZE-1):
        for t in range(1,IMAGE_SIZE-1):
            mean=(int(imga[i+1][t])+int(imga[i][t+1])+int(imga[i-1][t])+int(imga[i][t-1])+int(imga[i-1][t-1])+int(imga[i+1][t+1])+int(imga[i-1][t+1])+int(imga[i+1][t-1])+int(imga[i][t]))/9
            if mean==100:
                print("找到了")
                return t,i
def findtwopoint(ps,point):
    ps=np.array(ps)
    point=np.array(point)
    c = [point[i] -ps  for i in range(len(point))]
    list1=[]
    for u in range(len(c)):
        op2=np.linalg.norm(np.array(c[u]))
        list1.append(op2)
    i=list1.index(min(list1))
    list1[i]=9999999
    t=list1.index(min(list1)) 
    return i,t

@jit
def findmin(image):
    for zy in range(IMAGE_SIZE):
        for zx in range(IMAGE_SIZE):
            if image[zy][zx]==100:
                image[zy][zx]=0
    return image
@jit
def findmin2(image):
    for zy in range(IMAGE_SIZE):
        for zx in range(IMAGE_SIZE):
            if image[zy][zx]==50:
                image[zy][zx]=0
    return image
@jit
def findmax(image):
    for zy in range(IMAGE_SIZE-1,-1,-1):
        for zx in range(IMAGE_SIZE-1,-1,-1):
            if image[zy][zx]==50:
                maxx=zx
                maxy=zy
                return maxx,maxy

if __name__ == '__main__':
#初始化
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(out_pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            keep_probability = sess.graph.get_tensor_by_name(name="keep_probabilty:0")
            image =sess.graph.get_tensor_by_name(name="input_image:0")
            pred_annotation = sess.graph.get_tensor_by_name("inference/prediction:0")
            sess = tf.Session()
            #遍历图像
            for i in os.listdir(test_path):
                path=os.path.join(test_path,i)
                print(i)
                img=cv2.imread(path)
                size=img.shape#0是h 1是w
                #print("h,w",size[0],size[1])
                #检查是不是网查件
                lower_blue = np.array([100,110,110])
                upper_blue = np.array([130,255,255])
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                res = cv2.bitwise_and(img, img, mask=mask)
                # cv2.imshow('Result', res)
                m=np.max(res)
                if m==0:#等于0就证明不是网查件
                    #1.进行fcn分割
                    imga=freeze_graph_test(img)
                    print("1::::",np.unique(imga))
                    image2=findmin(imga.copy())
                    
                    print("2::::",np.unique(image2))
                    # print("min:",minx,miny)
                    # maxx,maxy=findmax(imga)
                    # print("max:",maxx,maxy)
                    # image = sess.run([image]) 
                    image3=findmin2(imga.copy())
                    cv2.imshow("zz",imga)
                    cv2.imshow("zz2",image2)
                    cv2.imshow("zz23",image3)
                    cv2.waitKey(0)

                    #2.边缘检测
                    _,contours, hierarchy = cv2.findContours(image2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)#找边框

                    for t in range(len(contours)):
                        hull = cv2.convexHull(contours[t])
                        epsilon = 0.01*cv2.arcLength(hull, True)#0.36081735求边框周长，epsilon 是精度

                        point=cv2.approxPolyDP(hull, epsilon, True) 
                        # point[:,0,0]=point[:,0,0]*size[1]/IMAGE_SIZE
                        # point[:,0,1]=point[:,0,1]*size[0]/IMAGE_SIZE
                        if (len(point) == 4):#and (int(point[0][0][0:1])>700) and (int(point[0][0][1:2])>1000)这里找身份证特征，因为背景简单直接判断边框周长就行
                            length= cv2.arcLength(point, True)#计算轮廓边长
                            area=cv2.contourArea(point)
                            if(area>60):
                                head_x,head_y=find_head(imga)
                                # head_x=int(head_x*size[1]/IMAGE_SIZE)
                                # head_y=int(head_y*size[1]/IMAGE_SIZE)
                                # print(head_x,head_y)
                                p1=point[0][0].tolist()
                                p2=point[1][0].tolist()
                                p3=point[2][0].tolist()
                                p4=point[3][0].tolist()
                                point=[p1,p2,p3,p4]
                                num1,num2=findtwopoint([head_x,head_y],point)#紧邻的两点
                                near1=point[num1]
                                near2=point[num2]
                                list_point=[0,1,2,3]
                                list_point.remove(num1)
                                list_point.remove(num2)
                                other1=point[list_point[0]]
                                other2=point[list_point[1]]
                                #print(near1,near2,other1,other2)
                                dx=abs(near1[0]-near2[0])
                                dy=abs(near1[1]-near2[1])
                                pp1_x=0
                                pp1_y=0
                                pp2_x=0
                                pp2_y=0
                                pp3_x=0
                                pp3_y=0
                                pp4_x=0
                                pp4_y=0
                                #靠近y
                                print("head_y:",head_y)
                                head_x=int(head_x*size[1]/IMAGE_SIZE)
                                head_y=int(head_y*size[0]/IMAGE_SIZE)
                                
                                if dx<bias:    
                                    if near1[0]<IMAGE_SIZE/2:
                                         #区分靠近标题上下点
                                        if near1[1]<near2[1]:
                                            near1,near2=near2,near1
                                        pp1_x=near1[0]
                                        pp1_y=near1[1]
                                        pp2_x=near2[0]
                                        pp2_y=near2[1]
                                        #print(near1,near2,other1,other2)
                                        # print(near1[1],other1[1])
                                        if abs(near1[1]-other1[1])<20:
                                            pp4_x=other1[0]
                                            pp4_y=other1[1]
                                            #print("1")
                                        if abs(near1[1]-other2[1])<20:
                                            pp4_x=other2[0]
                                            pp4_y=other2[1]
                                           # print("2")
                                        if abs(near2[1]-other1[1])<20:
                                            pp3_x=other1[0]
                                            pp3_y=other1[1]
                                          #  print("3")
                                        if abs(near2[1]-other2[1])<20:
                                            pp3_x=other2[0]
                                            pp3_y=other2[1]
                                         #   print("4")
                                        #print((pp1_x,pp1_y),(pp2_x,pp2_y),(pp3_x,pp3_y),(pp4_x,pp4_y))
                                        pp1_x_r=int(pp1_x*size[1]/IMAGE_SIZE)
                                        pp1_y_r=int(pp1_y*size[0]/IMAGE_SIZE)
                                        pp2_x_r=int(pp2_x*size[1]/IMAGE_SIZE)
                                        pp2_y_r=int(pp2_y*size[0]/IMAGE_SIZE)
                                        pp3_x_r=int(pp3_x*size[1]/IMAGE_SIZE)
                                        pp3_y_r=int(pp3_y*size[0]/IMAGE_SIZE)
                                        pp4_x_r=int(pp4_x*size[1]/IMAGE_SIZE)
                                        pp4_y_r=int(pp4_y*size[0]/IMAGE_SIZE)
                                        # pts1=np.float32([(pp1_y_r,pp1_x_r),(pp2_y_r,pp2_x_r),(pp3_y_r,pp3_x_r),(pp4_y_r,pp4_x_r)])
                                        # pts2=np.float32([[0, 0],[0, pp3_x_r-pp2_x_r],[pp1_y_r-pp2_y_r,0],[pp1_y_r-pp2_y_r,pp3_x_r-pp2_x_r]])#透视变换目标坐标
                                        img_jie=img[pp2_y_r-30:pp1_y_r,head_x:pp4_x_r]
                                        imgjie_y=img_jie.shape[0]
                                        imgjie_x=img_jie.shape[1]
                                        # print(imgjie_x,imgjie_y)
                                        pts1=np.float32([(0,imgjie_y),(0,0),(imgjie_x,0),(imgjie_x,imgjie_y)])
                                        pts2=np.float32([[0, 0],[imgjie_y,0],[imgjie_y, imgjie_x],[0,imgjie_x]])#透视变换目标坐标
                                        # print(pts1)
                                        # print(pts2)
                                        M = cv2.getPerspectiveTransform(pts1, pts2)
                                        #img[pp2_y_r:pp1_y_r,pp1_x_r:pp4_x_r]
                                        dst = cv2.warpPerspective(img_jie, M, (imgjie_y,imgjie_x))
                                        #dst=cv2.resize(dst,(int(pp3_y-pp1_y),int(pp4_x-pp1_x)))
                                        # cv2.imshow("a",dst)
                                        # img=cv2.resize(img,(int(size[1]/2),int(size[0]/2)))
                                        # cv2.circle(img, pts1[0], 60, (0, 0, 255), 0)
                                        # cv2.circle(img, pts1[1], 60, (0, 0, 255), 0)
                                        # cv2.circle(img, pts1[2], 60, (0, 0, 255), 0)
                                        # cv2.circle(img, pts1[3], 60, (0, 0, 255), 0)

                                        # cv2.imshow("b",img)
                                        # cv2.waitKey(0)
                                        #设置不同边界情况下4个点的位置
                                        #上边界
                                        print("y左")
                                    else:
                                        if near1[1]>near2[1]:
                                            near1,near2=near2,near1
                                        pp1_x=near1[0]
                                        pp1_y=near1[1]
                                        pp2_x=near2[0]
                                        pp2_y=near2[1]
                                        #print(near1,near2,other1,other2)
                                        # print(near1[1],other1[1])
                                        if abs(near1[1]-other1[1])<20:
                                            pp4_x=other1[0]
                                            pp4_y=other1[1]
                                            #print("1")
                                        if abs(near1[1]-other2[1])<20:
                                            pp4_x=other2[0]
                                            pp4_y=other2[1]
                                           # print("2")
                                        if abs(near2[1]-other1[1])<20:
                                            pp3_x=other1[0]
                                            pp3_y=other1[1]
                                          #  print("3")
                                        if abs(near2[1]-other2[1])<20:
                                            pp3_x=other2[0]
                                            pp3_y=other2[1]
                                         #   print("4")
                                        # print((pp1_x,pp1_y),(pp2_x,pp2_y),(pp3_x,pp3_y),(pp4_x,pp4_y))

                                        pp1_x_r=int(pp1_x*size[1]/IMAGE_SIZE)
                                        pp1_y_r=int(pp1_y*size[0]/IMAGE_SIZE)
                                        pp2_x_r=int(pp2_x*size[1]/IMAGE_SIZE)
                                        pp2_y_r=int(pp2_y*size[0]/IMAGE_SIZE)
                                        pp3_x_r=int(pp3_x*size[1]/IMAGE_SIZE)
                                        pp3_y_r=int(pp3_y*size[0]/IMAGE_SIZE)
                                        pp4_x_r=int(pp4_x*size[1]/IMAGE_SIZE)
                                        pp4_y_r=int(pp4_y*size[0]/IMAGE_SIZE)
                                        # print(pp1_y_r-20,pp2_y_r+60,pp4_x_r-10,head_x+10)
                                        img_jie=img[pp1_y_r-150:pp2_y_r+150,pp4_x_r:head_x+100]
                                        imgjie_y=img_jie.shape[0]
                                        imgjie_x=img_jie.shape[1]

                                        print(imgjie_x,imgjie_y)
                                        pts1=np.float32([(imgjie_x,0),(imgjie_x,imgjie_y),(0,imgjie_y),(0,0)])
                                        pts2=np.float32([[0, 0],[imgjie_y,0],[imgjie_y, imgjie_x],[0,imgjie_x]])#透视变换目标坐标
                                        # print(pts1)
                                        # print(pts2)
                                        M = cv2.getPerspectiveTransform(pts1, pts2)
                                        #img[pp2_y_r:pp1_y_r,pp1_x_r:pp4_x_r]
                                        dst = cv2.warpPerspective(img_jie, M, (imgjie_y,imgjie_x))
                                        img=cv2.resize(img,(int(size[1]/4),int(size[0]/4)))
                                        img_jie=cv2.resize(img_jie,(int(img_jie.shape[1]/4),int(img_jie.shape[0]/4)))
                                        cv2.imshow("a",img_jie)
                                        cv2.imshow("c",img)
                                        # cv2.imshow("d",dst)
                                        cv2.waitKey(0)
                                        print("y右")
                                else:#靠近x
                                    if near1[1]<IMAGE_SIZE/2:
                                        if near1[0]>near2[0]:
                                            near1,near2=near2,near1
                                        pp1_x=near1[0]
                                        pp1_y=near1[1]
                                        pp2_x=near2[0]
                                        pp2_y=near2[1]
                                        #print(near1,near2,other1,other2)
                                        # print(near1[1],other1[1])
                                        if abs(near1[0]-other1[0])<20:
                                            pp4_x=other1[0]
                                            pp4_y=other1[1]
                                            #print("1")
                                        if abs(near1[0]-other2[0])<20:
                                            pp4_x=other2[0]
                                            pp4_y=other2[1]
                                           # print("2")
                                        if abs(near2[0]-other1[0])<20:
                                            pp3_x=other1[0]
                                            pp3_y=other1[1]
                                          #  print("3")
                                        if abs(near2[0]-other2[0])<20:
                                            pp3_x=other2[0]
                                            pp3_y=other2[1]
                                         #   print("4")
                                        # print((pp1_x,pp1_y),(pp2_x,pp2_y),(pp3_x,pp3_y),(pp4_x,pp4_y))

                                        pp1_x_r=int(pp1_x*size[1]/IMAGE_SIZE)
                                        pp1_y_r=int(pp1_y*size[0]/IMAGE_SIZE)
                                        pp2_x_r=int(pp2_x*size[1]/IMAGE_SIZE)
                                        pp2_y_r=int(pp2_y*size[0]/IMAGE_SIZE)
                                        pp3_x_r=int(pp3_x*size[1]/IMAGE_SIZE)
                                        pp3_y_r=int(pp3_y*size[0]/IMAGE_SIZE)
                                        pp4_x_r=int(pp4_x*size[1]/IMAGE_SIZE)
                                        pp4_y_r=int(pp4_y*size[0]/IMAGE_SIZE)
                                        img_jie=img[head_y:pp3_y_r,pp1_x_r:pp2_x_r+35]
                                        imgjie_y=img_jie.shape[0]
                                        imgjie_x=img_jie.shape[1]
                                        # print(imgjie_x,imgjie_y)
                                        # pts1=np.float32([[0, 0],[0,imgjie_x],[imgjie_y, imgjie_x],[imgjie_y,0]])
                                        pts1=np.float32([(0,0),(0,imgjie_y),(imgjie_x, imgjie_y),(imgjie_x,0)])
                                        pts2=np.float32([[0, 0],[0,imgjie_y],[imgjie_x, imgjie_y],[imgjie_x,0]])#透视变换目标坐标
                                        # print(pts1)
                                        # print(pts2)
                                        M = cv2.getPerspectiveTransform(pts1, pts2)
                                        #img[pp2_y_r:pp1_y_r,pp1_x_r:pp4_x_r]
                                        dst = cv2.warpPerspective(img_jie, M, (imgjie_x,imgjie_y))
                                        # cv2.imshow("c",img_jie)
                                        # cv2.imshow("d",dst)
                                        # cv2.waitKey(0)
                                        #上边界
                                        print("x上")
                                    else:
                                        if near1[0]<near2[0]:
                                            near1,near2=near2,near1
                                        pp1_x=near1[0]
                                        pp1_y=near1[1]
                                        pp2_x=near2[0]
                                        pp2_y=near2[1]
                                        #print(near1,near2,other1,other2)
                                        # print(near1[1],other1[1])
                                        if abs(near1[0]-other1[0])<20:
                                            pp4_x=other1[0]
                                            pp4_y=other1[1]
                                            #print("1")
                                        if abs(near1[0]-other2[0])<20:
                                            pp4_x=other2[0]
                                            pp4_y=other2[1]
                                           # print("2")
                                        if abs(near2[0]-other1[0])<20:
                                            pp3_x=other1[0]
                                            pp3_y=other1[1]
                                          #  print("3")
                                        if abs(near2[0]-other2[0])<20:
                                            pp3_x=other2[0]
                                            pp3_y=other2[1]
                                         #   print("4")
                                        print((pp1_x,pp1_y),(pp2_x,pp2_y),(pp3_x,pp3_y),(pp4_x,pp4_y))

                                        pp1_x_r=int(pp1_x*size[1]/IMAGE_SIZE)
                                        pp1_y_r=int(pp1_y*size[0]/IMAGE_SIZE)
                                        pp2_x_r=int(pp2_x*size[1]/IMAGE_SIZE)
                                        pp2_y_r=int(pp2_y*size[0]/IMAGE_SIZE)
                                        pp3_x_r=int(pp3_x*size[1]/IMAGE_SIZE)
                                        pp3_y_r=int(pp3_y*size[0]/IMAGE_SIZE)
                                        pp4_x_r=int(pp4_x*size[1]/IMAGE_SIZE)
                                        pp4_y_r=int(pp4_y*size[0]/IMAGE_SIZE)
                                        print(head_y)
                                        img_jie=img[pp3_y_r:head_y+int(pp2_y_r/14),pp2_x_r-30:pp1_x_r]
                                        imgjie_y=img_jie.shape[0]
                                        imgjie_x=img_jie.shape[1]
                                        # print(imgjie_x,imgjie_y)
                                        #pts1=np.float32([[imgjie_y, imgjie_x],[imgjie_y, 0],[0, 0],[0,imgjie_x]])
                                        pts1=np.float32([(imgjie_x,imgjie_y),(0,imgjie_y),(0,0),(imgjie_x,0)])
                                        pts2=np.float32([[0, 0],[imgjie_x,0],[imgjie_x, imgjie_y],[0,imgjie_y]])#透视变换目标坐标
                                        # print(pts1)
                                        # print(pts2)
                                        M = cv2.getPerspectiveTransform(pts1, pts2)
                                        #img[pp2_y_r:pp1_y_r,pp1_x_r:pp4_x_r]
                                        dst = cv2.warpPerspective(img_jie, M, (imgjie_x,imgjie_y))
                                        # cv2.imshow("c",img_jie)
                                        # cv2.imshow("d",dst)
                                        # cv2.waitKey(0)
                                        print("x下")

                                show_size=dst.shape
                                dst_show=cv2.resize(dst,(int(show_size[1]/2),int(show_size[0]/2)))
                                cv2.imshow("a",dst_show)
                                
                                show_size2=img.shape
                                img_show=cv2.resize(img,(int(show_size2[1]/4),int(show_size2[0]/4)))
                                cv2.imshow("c",img_show)
                                # cv2.imshow("b",imga)
                                cv2.waitKey(0)
                                # print(np.unique(imga))
                                
                                # print("area:",area)
                                # print("lebgth:",length)
                                # for i in range(len(point)-1):
                                #     # print(point[i][0])
                                #     cv2.line(img, (int(point[i][0][0]*size[1]/IMAGE_SIZE),int(point[i][0][1]*size[0]/IMAGE_SIZE)), (int(point[i+1][0][0]*size[1]/IMAGE_SIZE),int(point[i+1][0][1]*size[0]/IMAGE_SIZE)), (0,255,0), 3)
                                # # cv2.line(img, tuple(point[0][0]), tuple(point[3][0]), (0,255,0), 3)
                                # cv2.line(img, (int(point[0][0][0]*size[1]/IMAGE_SIZE),int(point[0][0][1]*size[0]/IMAGE_SIZE)), (int(point[3][0][0]*size[1]/IMAGE_SIZE),int(point[3][0][1]*size[0]/IMAGE_SIZE)), (0,255,0), 3)
                               
                                # M = cv2.getPerspectiveTransform(pts1, pts2)
                                # dst = cv2.warpPerspective(img, M, (int(size[1]),int(size[0])))
                                # dst=cv2.resize(dst,(int(size[1]/2),int(size[0]/2)))
                                # cv2.imshow("a",dst)
                                # img=cv2.resize(img,(int(size[1]/2),int(size[0]/2)))
                                # cv2.imshow("b",img)
                                # cv2.waitKey(0)
                        
                    #3.判断方向
                    #4.映射回原图
                
                    #5.透视变换
                else:
                    print("这个是网图")
            #2.透视变换
    #cv2.imshow("a",mask)
    # cv2.waitKey(0)
    ##2.fcn变换
    #freeze_graph_test(pb_path=out_pb_path,test_path=test_path)
    
