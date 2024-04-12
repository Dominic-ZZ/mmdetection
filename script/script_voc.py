import os, cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageChops
import numpy as np


def get_classes(data_dir):
    """读取文件夹中的所有xml文件(VOC)格式，获取数据集的种类和数量"""

    # 存储所有类别的集合
    # classes = set()
    classes = {}

    # 遍历数据集文件夹中的xml文件，获取类别
    for xml_file in os.listdir(data_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(data_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                obj_name = obj.find("name").text
                # classes.add(obj_name)
                classes[obj_name] = classes.get(obj_name, 0) + 1

    # 输出所有类别
    i=0
    for key in classes.keys():
        i+=1
        print(i, " ", "|"+str(key)+"|", str(classes[key])+"张")


def replace_name(folder_path, old_str, new_str):
    """将文件夹下所有文件名中的某个字符串替换"""
    for filename in os.listdir(folder_path):
        if old_str in filename:
            new_filename = filename.replace(old_str, new_str)
            os.rename(
                os.path.join(folder_path, filename),
                os.path.join(folder_path, new_filename),
            )
            print("替换：", filename, "  =>  ", new_filename)


def remove_class(dataset_path, class_name):
    """
    用于读取VOC格式存储的AI数据集中的xml文件, 识别包含的类别并删除指定的类别的相关标注信息。
    如果某张图片中只包含指定的类别, 则会将该图片及其对应的xml文件删除。
    """
    for root_dir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    if name == class_name:
                        root.remove(obj)
                if len(root.findall("object")) == 0:
                    image_path = os.path.splitext(xml_path)[0] + ".jpg"
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        os.remove(xml_path)
                        print(f"Removed {image_path} and {xml_path}")
                else:
                    tree.write(xml_path)


def remove_del_class(dataset_path, class_name):
    """
    删除包含指定类的图和xml
    """
    for root_dir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    if name == class_name:
                        image_path = os.path.splitext(xml_path)[0] + ".jpg"
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            os.remove(xml_path)
                            print(f"Removed {image_path} and {xml_path}")


def del_no_target(dataset_path):
    """删除没有target的图片和xml"""
    for root_dir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if not root.findall("object"):
                    image_path = os.path.splitext(xml_path)[0] + ".jpg"
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        os.remove(xml_path)
                        print(f"Removed {image_path} and {xml_path}")


def modify_classname(dataset_dir, old_classname, new_classname):
    """
    修改VOC格式AI数据集中的类名

    :param xml_file_path: VOC格式AI数据集中的XML文件名
    :param old_classname: 需要被修改的类名
    :param new_classname: 修改后的类名
    """

    for root_dir, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(root_dir, file)
                tree = ET.parse(xml_file_path)
                root = tree.getroot()

                for obj in root.findall("object"):
                    name = obj.find("name").text
                    if name == old_classname:
                        obj.find("name").text = new_classname

                tree.write(xml_file_path)


def rename_img_and_xml(dataset_path, prefix):
    # 获取所有图片和XML文件的路径
    file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        idx = 0
        for file in files:
            if file.endswith(".jpg"):
                jpg_path = os.path.join(root, file)
                xml_path = os.path.join(root, file.replace(".jpg", ".xml"))
                if not os.path.isfile(xml_path):
                    print(xml_path, "not exists!")
                    break

                file_name = prefix + str(idx)

                os.rename(
                    jpg_path,
                    os.path.join(os.path.dirname(jpg_path), file_name + ".jpg"),
                )
                os.rename(
                    xml_path,
                    os.path.join(os.path.dirname(xml_path), file_name + ".xml"),
                )
                print(jpg_path, xml_path)
                idx += 1


def found_same_img(path):
    dir_list = []
    for img_name in os.listdir(path):
        if img_name.endswith(".jpg"):
            img_dir = os.path.join(path, img_name)
            dir_list.append(img_dir)
    print("共：", len(dir_list), "张")
    for No, img_dir in enumerate(dir_list):
        # start=time.time()
        img = Image.open(img_dir).resize((100, 100))
        i = 0
        for other_img_dir in dir_list[No + 1 :]:
            i += 1
            # other_img=cv2.imread(other_img_dir)
            other_img = Image.open(other_img_dir).resize((100, 100))
            # print(img_dir,other_img_dir)
            try:
                diff = ImageChops.difference(img, other_img)
                if diff.getbbox() is None:
                    print("Same:", img_dir, "----", other_img_dir)
            except:
                print("Error, 图片不匹配: ", img_dir, "----", other_img_dir)
                continue
        print("次数:", i)


def del_rotated_img(dataset_path, percentage):
    """计算图中纯黑所占比例，删除旋转增强后的图片"""
    rotated_num = 0
    total = 0
    for img_name in os.listdir(dataset_path):
        if img_name.endswith(".jpg"):
            total += 1
            img_dir = os.path.join(dataset_path, img_name)
            xml_dir = os.path.join(dataset_path, img_name.replace(".jpg", ".xml"))
            if os.path.exists(img_dir) and os.path.exists(xml_dir):
                # if os.path.exists(img_dir):
                img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                black_pixels = np.sum(img == 0)

                # 计算图片总像素点数量
                total_pixels = img.shape[0] * img.shape[1]

                # 计算纯黑色区域比例
                black_ratio = black_pixels / total_pixels
                if black_ratio > percentage:
                    rotated_num += 1
                    os.remove(img_dir)
                    os.remove(xml_dir)
    print("rotated/total: %d/%d, deleted: %d" % (rotated_num, total, rotated_num))

def rotate_images_in_folder(folder_path, angle, expand):
    """将所有图像全部旋转
    angle: 旋转度数
    expand: True=逆时针
    """
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        print("文件夹路径不存在！")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件是否是图片
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif')):
            try:
                # 打开图片并旋转90度
                with Image.open(file_path) as img:
                    rotated_img = img.rotate(angle, expand=expand) #True=顺时针
                    rotated_img.save(file_path)
                print(f"{filename} 已经被" + ("顺时针" if expand else "逆时针") + "旋转" + str(angle) + "度")
            except Exception as e:
                print(f"处理文件 {filename} 时出错：{e}")
        else:
            print(f"跳过文件 {filename}，因为不是图片")

import os

def remove_unmatched_data(image_folder, annotation_folder):
    '''删除：没有标注文件的图片和没有对应图片的标注文件'''
    # 获取所有图片和标注文件的文件名
    image_files = os.listdir(image_folder)
    annotation_files = os.listdir(annotation_folder)
    
    # 获取所有有标注文件的图片和所有有图片的标注文件
    annotated_images = set([filename.split('.')[0] for filename in annotation_files])
    annotated_files = set([filename.split('.')[0] for filename in image_files])
    
    # 找到需要删除的图片和标注文件
    images_to_remove = [filename for filename in image_files if filename.split('.')[0] not in annotated_images]
    annotations_to_remove = [filename for filename in annotation_files if filename.split('.')[0] not in annotated_files]
    
    # 删除没有标注的图片
    for filename in images_to_remove:
        os.remove(os.path.join(image_folder, filename))
        print(f"Removed {filename}")
    
    # 删除没有对应图片的标注文件
    for filename in annotations_to_remove:
        os.remove(os.path.join(annotation_folder, filename))
        print(f"Removed {filename}")


if __name__ == "__main__":
    dataset_path = r"/workspace/mmdetection/data/drug/annotations"
    # image_folder="/workspace/mmdetection/data/drug/img"
    # annotation_folder="/workspace/mmdetection/data/drug/annotations"

    get_classes(dataset_path)
    # replace_name(dataset_path, "_jpg_", "_")
    # remove_class(dataset_path, "1")
    # remove_del_class(dataset_path, "delete")
    # del_no_target(dataset_path)
    # modify_classname(dataset_path, "", "")
    # rename_img_and_xml(dataset_path, "WildAnimal_")
    # found_same_img(dataset_path)
    # del_rotated_img(dataset_path, 0.002)
    # rotate_images_in_folder(folder_path=dataset_path, angle=90, expand=True)
    # remove_unmatched_data(image_folder, annotation_folder)
