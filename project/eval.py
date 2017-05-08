# _*_ encoding:utf-8 _*_
import argparse
import cv2
import numpy as np
import pickle
import re
import tensorflow as tf
import time

from models import detect_face
from models import facenet

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from utils import load_graph
from utils import get_batch
from utils import get_label_batch

def read_video():
    path = "./data/test/video/" + CONFIG.user + "/" + CONFIG.video_file + ".mp4"
    cap = cv2.VideoCapture(path)
    count = 0
    frame_arr = []
    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break
        if count % CONFIG.skip_frame == 0:
            frame_arr.append(frame)
    return frame_arr

def crop_face(frame_arr):
    minsize = 100     # minimum size of face
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709    # scale factor
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
            
            crop_faces=[] 
            for frame in frame_arr:
                # time_start = time.time()

                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0] 
          
                for face_position in bounding_boxes:
                    face_position=face_position.astype(int)
                    crop=frame[face_position[1]:face_position[3],face_position[0]:face_position[2],]
                    crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
                    crop_faces.append(crop)
                # print(time.time() - time_start)

    return np.array(crop_faces).reshape(-1,160,160,3)  


def embedding(crop_Face, graph): 
    with tf.Session(graph=graph) as sess:          
        images_placeholder = graph.get_tensor_by_name('prefix/input:0')
        embeddings = graph.get_tensor_by_name("prefix/embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("prefix/phase_train:0")            
        image_size = 160    # images_placeholder.get_shape()[1]
        embedding_size = embeddings.get_shape()[1]

        print("对图像进行预处理(白化)......")
        face = []
        for img in crop_Face:
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            whiten_img = facenet.prewhiten(img)
            face.append(whiten_img) 

        print("图像的高维度映射......")
        face_arr = np.array(face).reshape(-1,160,160,3)
        
        emb = []
        n_bacth = int(len(face_arr) / CONFIG.batch_size)

        for i in range(n_bacth):
            # time_begin = time.time()
            batch = get_batch(face_arr, CONFIG.batch_size, i)
            feed_dict = {images_placeholder:batch, phase_train_placeholder:False}
            emb.append(sess.run(embeddings, feed_dict=feed_dict))
            # print("mapping time:", time.time() - time_begin)

        emb = np.array(emb).reshape(-1,128)    
        return emb

def l2distance(vec1, vec2):  # Euclidean distance 
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def decision(emb,cls):
        with open('./data/train/' +  CONFIG.user + '/emb.pkl', "rb") as f:
            emb_dict = pickle.load(f)
            em_dict = dict(emb_dict)

        # result = cls.predict(emb)
        result_count = {"others":0}
        for emb_item in emb:
            result = cls.predict(emb_item)

            li = [l2distance(emb_dict[item], emb_item) for item in emb_dict.keys() if item.startswith(result[0])]
            if np.mean(li) > 0.7:
                result_count["others"] += 1
            #     print("陌生人开门")
            else:
                if result[0] not in result_count.keys():
                    result_count[result[0]] = 1
                else:
                    result_count[result[0]] += 1

        print(result_count)

        total_nunm = sum(result_count.values())

        for k,v in result_count.iteritems():
            if k == CONFIG.video_file:
                print("预测成功的比例为:{0:}".format(float(v)/total_nunm))

        
        
def train():
    print("读取 pkl 文件并进行训练......")
    with open('./data/train/' +  CONFIG.user + '/emb.pkl', "rb") as f:
        emb_dict = pickle.load(f)
        em_dict = dict(emb_dict)

    y_train = np.array(emb_dict.keys())
    X_train = np.array(emb_dict.values())

    labels = [item.split('_')[0] for item in y_train]
    # Todo:
    # 多角度改进分类器的效果
    # (1) 使用多个分类器进行embedding SVM,DT,KNN,RF,Adaboost
    # (2) 测试不同分类器的效果
    # (3) 连续视频帧的效果,连续监测问题
    # (4) 加入若干负类 
    # cls = SVC(kernel='linear',C=1)
    cls = LinearSVC(C=1, multi_class='ovr')
    # cls = KNeighborsClassifier(n_neighbors = 3)
    cls.fit(X_train, labels) 
    return cls
         

def main():
    frame_arr = read_video()
    crop_Face = crop_face(frame_arr)
    cls = train()
    graph = load_graph('./model_check_point/20170216-091149/freeze_model.pb')
    emb = embedding(crop_Face, graph)
    result_count = decision(emb, cls)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Parameters for the script.")
    PARSER.add_argument('--video_file', type=str, 
        help='path of video file', default='yeyun')
    PARSER.add_argument("--user", type=str, default="dever", help = "family name")
    PARSER.add_argument("--skip_frame", type=int, default= 1, help = "number of skip frame")
    PARSER.add_argument("--batch_size", type=int, default = 1, help = "number of batch_size")
    CONFIG = PARSER.parse_args()
    main()

