import cv2
import numpy as np
import os

# 指定文件夹路径
folder_path = r"./test"

# 遍历文件夹下的所有照片
for filename in os.listdir(folder_path):
    # 检查文件是否为.jpg文件
    if filename.endswith(".jpg"):
        # 加载图像
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 应用阈值处理以突出窗户（黑色小方块）
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # 找出窗户的轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 根据轮廓的大小过滤掉噪点，并去除面积大于500的轮廓
        min_area = 500  # 窗户轮廓的最小面积
        max_area = 50000 # 去除面积大于500的轮廓
        windows = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) <= max_area]

        # 可视化：在原始图像上绘制符合条件的轮廓
        output_image = image.copy()
        cv2.drawContours(output_image, windows, -1, (0, 255, 0), 2)

        # 计算剩余窗户的数量
        window_count = len(windows)

        # 输出窗户数量
        print(f"{filename} 窗户数量为: {window_count}")

        # 显示带有窗户标记的图片窗口（可选）
        resized_image = cv2.resize(output_image, (800, 800))
        cv2.imshow(f"{filename} - detect", resized_image)
        cv2.waitKey(0)  # 逐张显示，按任意键切换到下一张

        # 如果需要保存带有轮廓标记的图片
        output_path = os.path.join(folder_path, f"output_{filename}")
        cv2.imwrite(output_path, resized_image)

# 关闭所有窗口
cv2.destroyAllWindows()