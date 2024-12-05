import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort import nn_matching
import mediapipe as mp
import torchreid
import torchvision.models as models
import os

# OSNetモデルの構築と事前学習済み重みのロード
# model = torchreid.models.build_model(
#     name='osnet_x0_25',
#     num_classes=1000,
#     loss='softmax',
#     pretrained=True
# )

model = models.mobilenet_v3_large(pretrained=True)

model.eval()

class ResizeHalf:
    def __call__(self, img):
        # 元の画像サイズを取得
        width, height = img.size  # PIL画像で (幅, 高さ)
        # 縦横比を維持しつつ、画像を半分のサイズにリサイズ
        new_size = (width // 4, height // 4)
        return img.resize(new_size)

# 画像前処理の定義
transform = transforms.Compose([
    ResizeHalf(),  # 画像を半分に縮小（縦横比保持）
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 特徴量の抽出関数
def get_features(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = F.normalize(model(img_tensor), p=2, dim=1)
    return features

# ターゲット画像フォルダと特徴量取得
image_folder = "target_images"
target_image_paths = [
    os.path.join(image_folder, f"target{i}.png") for i in range(1, 13)
]

def get_average_features(image_paths):
    all_features = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        features = get_features(img)
        all_features.append(features)
    avg_features = torch.mean(torch.stack(all_features), dim=0)
    return avg_features

target_features = get_average_features(target_image_paths)

# MediapipeのPoseモデルを初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# DeepSORTの初期化
max_cosine_distance = 0.2
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age=10, n_init=3)

# 類似度の閾値
SIMILARITY_THRESHOLD = 0.85

# 動画の入力
cap = cv2.VideoCapture("dancing.mp4")

# HOGによる人物検出設定
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# グローバルフラグ
target_found = False  # ターゲット発見フラグ

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレーム全体の解像度を下げる（縦横比を維持）
    target_width = 140  # 縮小後の幅（例: 640ピクセル）
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    # 縦横比を維持して高さを計算
    new_width = target_width
    new_height = int(new_width / aspect_ratio)

    # リサイズ
    frame = cv2.resize(frame, (new_width, new_height))

    detections = []  # 検出結果リスト

    # HOGで人物を検出（ターゲットが見つかる前でも検出は続ける）
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    for (x, y, w, h) in boxes:
        # 検出された人物の領域を取得（範囲外チェック付き）
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        person_img = frame[y:y+h, x:x+w]
        person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        person_img_pil = Image.fromarray(person_img_rgb)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 検出された人物の特徴量を取得
        features = get_features(person_img_pil)

        # ターゲットとの類似度を計算
        similarity = F.cosine_similarity(features, target_features, dim=1).item()

        # 類似度が閾値を超えた場合のみDeepSORTに追加
        if similarity > SIMILARITY_THRESHOLD and not target_found:
            target_found = True  # ターゲット発見フラグを更新
            detection = Detection([x, y, w, h], confidence=0.5, feature=features.squeeze().numpy())
            detections.append(detection)

            # ターゲットと判断された場合、関節を描画
            results = pose.process(person_img_rgb)
            if results.pose_landmarks:
                 for landmark in results.pose_landmarks.landmark:
                     x_pos = int(landmark.x * w) + x
                     y_pos = int(landmark.y * h) + y
                     cv2.circle(frame, (x_pos, y_pos), 3, (0, 0, 255), -1)

        detection = Detection([x, y, w, h], confidence=0.5, feature=features.squeeze().numpy())
        detections.append(detection)

    # DeepSORTでトラッキングを更新
    tracker.predict()
    tracker.update(detections)

    # トラックの描画
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlwh()
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 結果の表示
    cv2.imshow('Target Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
