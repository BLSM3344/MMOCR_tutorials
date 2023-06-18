# python裁剪图片并保存
from PIL import Image
import os
import cv2

srcPath = r"E:\dataset\CCPDdet\textdet_imgs\train"
dstPath = r"E:\dataset\CCPDrec\train"

# 读取图片
for fi in os.listdir(srcPath):
    if fi.endswith(".jpg"):
        file_name = fi.split("-", 6)
        plate_bbox = file_name[2].split("_", 1)
        x1 = int(plate_bbox[0].split("&", 1)[0])
        y1 = int(plate_bbox[0].split("&", 1)[1])
        x2 = int(plate_bbox[1].split("&", 1)[0])
        y2 = int(plate_bbox[1].split("&", 1)[1])
        pl_bb = (x1, y1, x2, y2)
        img = Image.open(os.path.join(srcPath, fi))
        img_rec = img.crop(pl_bb)
        savs_path = os.path.join(dstPath, fi)
        img_rec.save(savs_path)
