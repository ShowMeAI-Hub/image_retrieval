import os
import cv2

# 删除路径与文件
def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

# 把图片压缩成拇指盖大小的小图
def resize_images(full_folder, out_folder, suffix='thumb', height=100, del_former_thumb=False):
    if del_former_thumb:
        del_file(out_folder)
    for image_file in os.listdir(full_folder):
        image = cv2.imread(full_folder + image_file)
        height_src, width_src, _ = image.shape
        #print('width: {}, height: {}'.format(width_src, height_src))

        width = (height / height_src) * width_src
        # print(' Thumb width: {}, height: {}'.format(width, height))

        resized_image = cv2.resize(image, (int(width), int(height)))

        image_name, image_extension = os.path.splitext(image_file)
        cv2.imwrite(out_folder + image_name + suffix + image_extension, resized_image)
    print('批量调整图像大小完成.')


if __name__ == '__main__':
    create_thumb_images(full_folder='./save_pic/',
                        out_folder='./resized_images/',
                        suffix='',
                        height=200,
                        del_former_thumb=True,
                        )
