# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import tensorflow as tf
import math
import numpy as np
import os
import pickle
import sys
import time

from models import detect_face
from models import facenet
import random
from scipy import misc
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from utils import load_graph
from utils import get_batch
from utils import get_label_batch


def embedding(face_list, name_list):
    print("载入如固化模型......")
    graph = load_graph('./model_check_point/20170216-091149/freeze_model.pb') 
    with tf.Session(graph=graph) as sess:          
        images_placeholder = graph.get_tensor_by_name('prefix/input:0')
        embeddings = graph.get_tensor_by_name("prefix/embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("prefix/phase_train:0")            
        image_size = 160    # images_placeholder.get_shape()[1]
        embedding_size = embeddings.get_shape()[1] 

        print("对图像进行预处理(白化)......")
        face = []
        for img in face_list:
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            whiten_img = facenet.prewhiten(img)
            face.append(whiten_img)

        print("图像的高维度映射......")

        face_arr = np.array(face).reshape(-1,160,160,3)
        # name_arr = np.array(name_list)
        # print(face_arr.shape[0])
        # print(name_arr)
        
        emb = []
        name = []
        n_bacth = int(len(face_arr) / CONFIG.batch_size)
        # print(n_bacth)

        for i in range(n_bacth):
            batch = get_batch(face_arr, CONFIG.batch_size, i)
            label_batch = get_label_batch(name_list, CONFIG.batch_size, i)
            feed_dict = {images_placeholder:batch, phase_train_placeholder:False}
            # start = time.time()
            # emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            emb.append(sess.run(embeddings, feed_dict=feed_dict))
            name.append(label_batch)
            # print(label_batch)
            # end = time.time() 
            # print(end-start)

        emb = np.array(emb).reshape(-1,128)
        # name.flatten()
        name = np.array(name).flatten() #[y for x in name for y in x]
        print(len(name), emb.shape)
        emb_dict = dict(zip(name, emb))

        with open("./data/train/" + CONFIG.user + "/emb.pkl", "wb") as f:
           pickle.dump(emb_dict, f)
           print("pkl保存在: ./data/train/" + CONFIG.user + "文件夹下......") # save as pickle

def video_save():
    print('创建网络并载入参数......')
    # 经验参数
    minsize = 20   # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709  # scale factor
    name_list = []
    face_list = []   
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
            path = os.path.join("data","train",CONFIG.user)
            # print(path)
            assert os.path.exists(path) , "路径不存在"

            for item in os.listdir(path):
                if not item.startswith('.') and (item.split(".")[-1] == "mp4"):
                    print("正在保存{0:}的脸部头像".format(item))
                    cap = cv2.VideoCapture(os.path.join(path, item))

                    count = 0  
                    while(True):
                        ret, frame = cap.read()
                        count += 1  
                        print(count)
                        if not ret:   # 读取帧数不成功
                           break
                        if count > CONFIG.n_frame:
                            break
                        if count % CONFIG.skip_frame == 0:             
                            # start = time.time()
                            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                            # end = time.time()
                            # print(end-start)  
                            nrof_faces = bounding_boxes.shape[0]       
                
                            if nrof_faces == 0:
                                    crop = []
                            else:
                                crop_faces=[]
                                face_position = bounding_boxes[0]
                                face_position=face_position.astype(int)
                                crop=frame[face_position[1]:face_position[3],
                                            face_position[0]:face_position[2],]

                                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                                face_list.append(crop)
                                name_list.append(item.split('.')[0] + '_'+ str(count))
            return face_list, name_list

        cap.release()
        cv2.destroyAllWindows() 


def main():
    print("人脸检测......")
    face_crop, name_list = video_save()
    assert len(face_crop), "无人注册"
    # print(face_crop[0].shape)
    # print(name_list)
    print("人脸检测完成")
    print("人脸的向量映射......") 
    embedding(face_crop, name_list)
    print("人脸向量映射完成")

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Parameters for the script.")
    PARSER.add_argument("--user", type=str, default="dever", help="family name")
    PARSER.add_argument("--skip_frame", type=int, default=15, help="number of skip frame")
    PARSER.add_argument("--batch_size", type=int, default=7, help="number of batch size")
    PARSER.add_argument("--n_frame", type = int, default=100, help="number of n_frame")
    CONFIG = PARSER.parse_args()
    main()

    # Todo:
    # 参考官网的代码,提高速度和准确度