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

# 输入：单个视频，标记单个人
# 输出：检测到的人脸数目
#     检测正确的人脸数目(漏检率) -> 有多少帧没有检测出这个人
#     检测错误的人脸数目(误检率) -> 其他人预测成这个人

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

        print("人脸识别结果:{0:}".format(result_count))

        # 统计在字典中的人的个数：
        y_train = np.array(emb_dict.keys())
        labels = set([item.split('_')[0] for item in y_train])       

        count_in_home = 0
        for k in result_count.keys():
            if k in labels:
                count_in_home += result_count[k]
        print("检测出人脸数:{0:}" .format(np.sum(result_count.values())))
        print("家庭人数:{0:} ". format(count_in_home))
        print("家庭人数所占比例: {0:}" .format(float(count_in_home)/np.sum(result_count.values())))
        # print(float(count_in_home)/np.sum(result_count.values()))

        num_people = len(result_count)
        total_frame = np.sum(result_count.values())
        alpha = 0.4
        # 打印最终结果：
        print("对视频的预测结果: ")
        for k,v in result_count.iteritems():
            if v  > (total_frame / num_people) * alpha and k not in  ["others"]:
                print k
        print("视频标签:{0:}". format(CONFIG.video_file))

        # return result_count 
            #     print(result, np.mean(li))
 
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
        help='path of video file', default='单人-徘徊')
    PARSER.add_argument("--user", type=str, default="dever", help = "family name")
    PARSER.add_argument("--skip_frame", type=int, default= 1, help = "number of skip frame")
    PARSER.add_argument("--batch_size", type=int, default = 1, help = "number of batch_size")
    CONFIG = PARSER.parse_args()
    main()

    # Todo:
    # 代码规范
    # 提前输出问题? (出于效率考虑，提前跳出的问题）

