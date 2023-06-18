import json
import os


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

data_list = []
invalid = []
folder_paths = [[r"E:\dataset\CCPDdet\textdet_imgs\train", "textrecog_train.json"],
                [r"E:\dataset\CCPDdet\textdet_imgs\test","textrecog_test.json"]]
pic_num = 0

# print(folder_path.split('\\')[-2:])  测试返回的是不是：textdet_imgs\test，发现不是，返回的是textdet_imgs和test，是分开的
# 遍历整个文件夹
for folder_path,name in folder_paths:
    for fi in os.listdir(folder_path):
        if fi.endswith(".jpg"):  # 文件是以jpg结尾的
            text = ''
            # 图片名字的每个部分是用-分割的，6是最多分割次数，也就是最多会分割成7份
            file_name = fi.split("-", 6)
            # 有些图片命名不正确，筛选出来删掉
            if len(file_name) < 7:
                os.remove(os.path.join(folder_path, fi))
                invalid.append(fi)
                continue
            # 提取bbox
            plate_bbox = file_name[2].split("_", 1)
            x1 = int(plate_bbox[0].split("&", 1)[0])
            y1 = int(plate_bbox[0].split("&", 1)[1])
            x2 = int(plate_bbox[1].split("&", 1)[0])
            y2 = int(plate_bbox[1].split("&", 1)[1])
            pl_bb = [x1, y1, x2, y2]
            # 提取polygon
            plate_polygon = file_name[3].split("_", 3)
            px1 = int(plate_polygon[0].split("&", 1)[0])
            py1 = int(plate_polygon[0].split("&", 1)[1])
            px2 = int(plate_polygon[1].split("&", 1)[0])
            py2 = int(plate_polygon[1].split("&", 1)[1])
            px3 = int(plate_polygon[2].split("&", 1)[0])
            py3 = int(plate_polygon[2].split("&", 1)[1])
            px4 = int(plate_polygon[3].split("&", 1)[0])
            py4 = int(plate_polygon[3].split("&", 1)[1])
            pl_po = [px1, py1, px2, py2, px3, py3, px4, py4]
            palte_text = file_name[4].split("_", 7)
            text_len = len(palte_text)
            text += provinces[int(palte_text[0])]
            text += alphabets[int(palte_text[1])]
            text += ads[int(palte_text[2])] + ads[int(palte_text[3])] + ads[int(palte_text[4])] + ads[
                int(palte_text[5])] + ads[int(palte_text[6])]
            # 新能源的车牌是多一位的，需要多读取一位
            if text_len == 8:
                text += ads[int(palte_text[7])]
            new_instance = {
                # img_path是数据集根目录到图片之间的相对地址的，这里巨坑，手册也不说清楚，还好问题不大，看报错就知道了
                "img_path": os.path.join(folder_path.split('\\')[-1], fi),
                "height": abs(y1-y2),
                "width": abs(x1-x2),
                "instances":
                    [
                        {
                            "bbox": pl_bb,
                            "bbox_label": 0,
                            "polygon": pl_po,
                            "text": text,
                            "ignore": False
                        },
                    ],
            }
            data_list.append(new_instance)
    print(invalid)
    print(len(data_list))
    json_data = {
        "metainfo": {
            "dataset_type": "TextRecogDataset",
            "task_name": "textrecog",
            "category": [{"id": 0, "name": "text"}]
        },
        "data_list": data_list
    }
    # 一定要encoding="utf-8"，因为有中文，不这么弄会报个和utf-8有关的错误
    with open(name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    data_list = []

