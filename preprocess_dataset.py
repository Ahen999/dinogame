import os, cv2
from tqdm import tqdm
from preprocess import preprocess_final

DATASET_LOCATION = './Sign-Language-Digits-Dataset/Edited Dataset/'
CLASSES = sorted(os.listdir(DATASET_LOCATION))
PREPROCESSED_DATASET_LOCATION = './Sign-Language-Digits-Dataset/Preprocessed-Dataset/'

def preprocess_dataset():
    if not os.path.exists(PREPROCESSED_DATASET_LOCATION):
        os.mkdir(PREPROCESSED_DATASET_LOCATION)
    for _class in CLASSES:
        class_dir = os.path.join(PREPROCESSED_DATASET_LOCATION, _class)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
    
    for _class in CLASSES:
        data_loc = os.path.join(DATASET_LOCATION, _class)
        image_files = os.listdir(data_loc)
        c = 1
        for _file in tqdm(image_files, desc=f"Processing {_class}"):
            img_path = os.path.join(data_loc, _file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            processed = preprocess_final(img)
            output_path = os.path.join(PREPROCESSED_DATASET_LOCATION, _class, f"{c}.jpg")
            cv2.imwrite(output_path, processed)
            c += 1

if __name__ == '__main__':
    preprocess_dataset()
