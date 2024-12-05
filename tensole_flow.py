import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort import nn_matching
import mediapipe as mp
import os

# EfficientNetを特徴量抽出モデルとして構築
def build_feature_extractor():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)

model = build_feature_extractor()

# 特徴量抽出関数
def extract_features(img):
    img_array = preprocess_input(np.expand_dims(img, axis=0))
    features = model.predict(img_array)
    return features / np.linalg.norm(features, axis=1, keepdims=True)

# 画像の前処理関数
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB').resize(target_size)
    return np.array(img)

# ターゲット画像の平均特徴量を計算
def calculate_average_features(image_folder, image_files):
    all_features = []
    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        img_array = preprocess_image(img_path)
        features = extract_features(img_array)
        all_features.append(features)
    return np.mean(all_features, axis=0)

# MediapipeのPose初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# DeepSORTの初期化
def initialize_tracker(max_cosine_distance=0.2, nn_budget=None):
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    return Tracker(metric, max_age=10, n_init=1)

tracker = initialize_tracker()

# 類似度の閾値
SIMILARITY_THRESHOLD = 0.85

# HOGによる人物検出の初期化
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ターゲット画像情報
image_folder = "target_images/"
target_image_files = [f"target{i}.png" for i in range(1, 13)]
target_features = calculate_average_features(image_folder, target_image_files)

# 動画の入力
cap = cv2.VideoCapture("dancing.mp4")

def process_frame(frame):
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))
    detections = []

    for (x, y, w, h) in boxes:
        person_img = frame[y:y+h, x:x+w]
        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        person_img_resized = cv2.resize(person_img_rgb, (224, 224))

        features = extract_features(person_img_resized)
        similarity = np.dot(features, target_features.T).item()

        if similarity > SIMILARITY_THRESHOLD:
            detection = Detection([x, y, w, h], confidence=0.5, feature=features.squeeze())
            detections.append(detection)

            results = pose.process(person_img_rgb)
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    x_pos = int(landmark.x * w) + x
                    y_pos = int(landmark.y * h) + y
                    cv2.circle(frame, (x_pos, y_pos), 3, (0, 0, 255), -1)

    return detections

def update_and_draw_tracks(frame, detections):
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlwh()
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = process_frame(frame)
    update_and_draw_tracks(frame, detections)

    cv2.imshow('Target Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
