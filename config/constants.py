YOLO_PATH = './models/yolov8n-face.pt'
CLASSIFIER_PATH = './models/best_model.pth'
TEST_NAME = 'test_hospital.png'

INPUT_PATH = './data/input/'
OUTPUT_PATH = './data/output/'

TRAIN_DATASET_PATH = './data/train'
VAL_DATASET_PATH = './data/val'
MODELS_PATH = './models'

FACE_ROI_SIZE = (128, 128)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]



NUM_STATUS = {
            0: "with_mask",
            1: "without_mask",
            2: "mask_incorrect"
        }

NUM_COLOR = {
    0: (0, 255, 0), # Green
    1: (255, 0, 0), # Red
    2: (150, 100, 0) # ~Orange
}