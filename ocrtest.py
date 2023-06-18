import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from mmocr.apis import MMOCRInferencer
import cv2

ocr = MMOCRInferencer(
    det=r'E:\code\MMOCR_tutorials\mmocr\configs\textdet\dbnetpp\dbnetpp_resnet50-dcnv2_fpnc_1200e_ccpd.py',
    det_weights=r"E:\code\MMOCR_tutorials\weights\dnetpp_10.pth",
    rec=r'E:\code\MMOCR_tutorials\mmocr\configs\textrecog\aster\aster_resnet45_6e_st_ccpd.py',
    rec_weights=r"E:\code\MMOCR_tutorials\weights\aster_180.pth",
    device="cuda")


# result = ocr(r'E:\code\MMOCR_tutorials\test\test4.jpg')
# print(result['predictions'][0]['det_polygons'])

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "STSONG.TTF", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def ocrimg(img):
    result = ocr(img)
    for i in range(len(result['predictions'][0]['det_polygons'])):
        pts = np.array(result['predictions'][0]['det_polygons'][i], np.int64)
        pts = pts.reshape((-1, 1, 2))
        if result['predictions'][0]['det_scores'][i] > 0.9:
            if (result['predictions'][0]['det_polygons']) is not None:
                cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
            text = result['predictions'][0]['rec_texts'][i]+"    "+str(round(result['predictions'][0]['det_scores'][i]*100,3))
            begin_point_x = pts[0][0][0]
            begin_point_y = pts[0][0][1] - 30 if pts[0][0][0] - 10 > 0 else 0
            img = cv2ImgAddText(img, text, begin_point_x, begin_point_y)
        # cv2.imshow("A video", img)
        # cv2.putText(img, text, (begin_point_x, begin_point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #             (100, 255, 0), 2)
    return img


video_path = r"E:\code\MMOCR_tutorials\video\65.flv"
videooutpath = r'result_video.mp4'
i = 0
time_1 = 0
time_2 = 0
capture = cv2.VideoCapture(video_path)
cnt = capture.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(videooutpath, fourcc, 20.0, (width, height), True)
if capture.isOpened():
    while True:
        ret, img_src = capture.read()
        if not ret: break
        img_out = ocrimg(img_src)  # 自己写函数op_one_img()逐帧处理
        writer.write(img_out)
        i += 1
        time_1 = time.perf_counter()
        print("处理中：{}/{},处理时间：{}".format(i, cnt,time_1-time_2))
        time_2 = time.perf_counter()
else:
    print('视频打开失败！')
writer.release()
