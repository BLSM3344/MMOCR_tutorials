import shutil
import os
import glob
import random

folder_path = r"E:\dataset\CCPD2019\CCPD2019\ccpd_base"
test_path = r"E:\dataset\CCPDdet\textdet_imgs\test"
train_path = r"E:\dataset\CCPDdet\textdet_imgs\train"

# 用嵌套的方法进入每一个子文件夹
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    global test_path
    global train_path
    if os.path.isfile(dir):  # 如果文件路径是文件，这执行文件的判断，看是否是所需类型
        if ext is None:
            Filelist.append(dir)
        else:
            if dir.endswith(ext): # 检查文件后缀，看是否是满足要求的文件类型
                if (random.randint(1, 10)) < 8:  # 划分数据集和测试集
                    # 判断文件是否存在，CCPD数据集会有重复的图片放在不同的文件夹里
                    if not (os.path.exists(os.path.join(train_path,dir.split('\\')[-1]))):
                        shutil.move(dir, train_path)
                else:
                    if not (os.path.exists(os.path.join(test_path, dir.split('\\')[-1]))):
                        shutil.move(dir, test_path)
            Filelist.append(dir)

    elif os.path.isdir(dir):  # 如果文件路径是文件夹，则执行嵌套，继续访问子文件夹
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist


Filelist = getFileList(folder_path, [], ".jpg")
print(len(Filelist))
