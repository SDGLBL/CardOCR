from pathlib import Path

from PIL import Image

# from application.bankcard import draw_img_by_box
# from text.yoyo import keras_detect as detect
from ocr.densenet.api import predict

# from text.yoyo.detector.detectors import TextDetector
# from utils.config.dconfig import show_densenetFlag_info
# from utils.image_tools.tools import get_boxes, sort_box

SAVE_PATH = './results/'


def readfile(path):
    floder = Path(path)
    file_paths = floder.glob('*.*')
    dic = {}
    for image_path in file_paths:
        image_label = image_path.stem
        # img = cv2.imread(str(image_path))
        # dic[image_label] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(image_path)
        dic[image_label] = img.convert('L')
    return dic


# def text_detect(img,
#                 MAX_HORIZONTAL_GAP=150,
#                 MIN_V_OVERLAPS=0.8,
#                 MIN_SIZE_SIM=0.4,
#                 TEXT_PROPOSALS_MIN_SCORE=0.05,
#                 TEXT_PROPOSALS_NMS_THRESH=0.05,
#                 TEXT_LINE_NMS_THRESH=0.1,
#                 ):
#     boxes, scores = detect.text_detect(np.array(img))
#     boxes = np.array(boxes, dtype=np.float32)
#     scores = np.array(scores, dtype=np.float32)
#     textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
#     shape = img.shape[:2]
#     boxes = textdetector.detect(boxes,
#                                 scores[:, np.newaxis],
#                                 shape,
#                                 TEXT_PROPOSALS_MIN_SCORE,
#                                 TEXT_PROPOSALS_NMS_THRESH,
#                                 TEXT_LINE_NMS_THRESH,
#                                 )
#
#     text_recs = get_boxes(boxes)
#     newBox = []
#     rx = 1
#     ry = 1
#     for box in text_recs:
#         x1, y1 = (box[0], box[1])
#         x2, y2 = (box[2], box[3])
#         x3, y3 = (box[6], box[7])
#         x4, y4 = (box[4], box[5])
#         newBox.append([x1 * rx - 15, y1 * ry, x2 * rx + 5, y2 * ry, x3 * rx + 5,
#                        y3 * ry, x4 * rx - 15, y4 * ry])
#     return sort_box(newBox)


if __name__ == '__main__':
    # show_densenetFlag_info()
    dic = readfile('./test_images')
    # for label, image in dic.items():
    #     text_box = text_detect(image)
    #     raw_img = draw_img_by_box(image, text_box)
    #     # result = predict(crop_img)
    #     raw_img.save(SAVE_PATH + label + '.jpeg')
    #     # crop_img.save(SAVE_PATH +label +' '+ result +'.jpeg')
    allnum = len(dic)
    rightnum = 0
    for i, j in dic.items():
        result = predict(j)
        flag = False
        remove_1 = str(i).replace('_', '')
        remove_2 = result.replace('_', '')
        print(remove_1)
        print(remove_2)
        if remove_1 == remove_2:
            flag = True
            rightnum += 1
        print(i + ':' + result + str(flag) + '\n')
    right_rate = rightnum / allnum * 100
    print('准确率:' + str(rightnum) + '/' + str(allnum) + '   ' + str(right_rate) + '%')
