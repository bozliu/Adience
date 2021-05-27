import os
import shutil

import cv2
from mtcnn import MTCNN


def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == "__main__":
    source_path = "./Adience_Benchmark"
    target_path = "./mtcnn"
    detector = MTCNN()
    error_count = 0

    for root, dir_list, file_list in os.walk(source_path):
        # print(root, dir_list, file_list)
        for file in file_list:
            try:
                old_path = os.path.join(root, file)
                new_path = os.path.join(target_path, old_path)
                dir_path = os.path.join(target_path, root)
                check_path(dir_path)
                if os.path.splitext(file)[-1][1:] not in ["png", "jpg"]:
                    shutil.copyfile(old_path, new_path)
                else:
                    # <class 'numpy.ndarray'>
                    img = cv2.cvtColor(cv2.imread(old_path), cv2.COLOR_BGR2RGB)
                    mtcnn_res = detector.detect_faces(img)
                    if len(mtcnn_res) == 0:
                        print(f"Cannot find any box for {old_path}")
                        shutil.copyfile(old_path, new_path)
                    else:
                        idx, max_confidence = 0, 0
                        for i, j in enumerate(mtcnn_res):
                            idx = i if j["confidence"] > max_confidence else idx
                            max_confidence = max(max_confidence, j["confidence"])
                        x, y, w, h = mtcnn_res[idx]["box"]
                        crop_img = img[y:y + h, x:x + w]
                        cv2.imwrite(new_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            except:
                pass
