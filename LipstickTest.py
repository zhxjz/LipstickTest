# encoding:utf-8
from collections import deque
import dlib
import numpy as np
import cv2

# 获得人脸矩形的坐标信息
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

# 将包含68个特征的的shape转换为numpy array格式
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# 将待检测的image进行resize
def resize(image, width=1200):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# 未加入代码部分
# 想借周围rgb值取均值来做边缘处理，但是失败了，明暗中的阴影部分会取极端值变小黑格。
def getbgr(image, xx, yy):
    bb = 0
    gg = 0
    rr = 0
    bgr = image[yy, xx]
    for ii in range(-1, 1):
        for jj in range(-1, 1):
            if ii == 0 and jj == 0:
                continue
            tx = xx + ii
            ty = yy + jj
            tbgr = image[ty, tx]
            bb += tbgr[0]
            gg += tbgr[1]
            rr += tbgr[2]

    bb = int(bb / 8 / 3 + bgr[0] / 3 * 2)
    gg = int(gg / 8 / 3 + bgr[1] / 3 * 2)
    rr = int(rr / 8 / 3 + bgr[2] / 3 * 2)
    print(bb, gg, rr)
    return bgr

# 未加入代码部分
# 想借bfs来拓展嘴唇边缘不自然部分。
def bfs(image, edge_list, convex):
    que = deque()
    dis = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    for node in edge_list:
        que.append((node[0], node[1]))
        dis[node[0], node[1]] = -10

    cc = 0
    while que:
        cc += 1
        now = que.popleft()
        if dis[now[0], now[1]] + 1 < 0:
            for xx in range(-1, 1):
                for yy in range(-1, 1):
                    ty = yy + now[1]
                    tx = xx + now[0]
                    if cv2.pointPolygonTest(convex, (tx, ty), False) < 0:
                        if dis[tx, ty] > dis[now[0], now[1]] + 1:
                            dis[tx, ty] = dis[now[0], now[1]] + 1
                            tbgr = image[ty, tx]
                            if tbgr[0] + 40 < 255:
                                image[ty, tx][0] = tbgr[0] + 40
                            if tbgr[1] - 10 < 255:
                                image[ty, tx][1] = tbgr[1] - 10
                            if tbgr[2] - 10 < 255:
                                image[ty, tx][2] = tbgr[2] - 10
                            if (tx, ty) not in que:
                                que.append((tx, ty))
    return image


def feature(image_file, lipstick_color):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    image = cv2.imread(image_file)
    image = resize(image, width=1200)
    print(image.shape)
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度图里定位人脸
    rects = detector(gray, 1)
    # shapes存储找到的人脸框，人脸框仅包含四个角数值如frontal_face_detector.png所示。
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        shape = shape[48:]
        shapes.append(shape)

    # 图片转为hsv形式，色调（H），饱和度（S），亮度（V）
    # H:  0 — 180
    # S:  0 — 255
    # V:  0 — 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for shape in shapes:
        sx, sy = image.shape[0], image.shape[1]
        # 外嘴唇凸包
        hull = cv2.convexHull(shape)
        # 内嘴唇凸包
        hull2 = cv2.convexHull(shape[12:])
        ## 圈出凸包区域
        # cv2.drawContours(image, [hull], -1, (255, 100, 168), -1)
        # cv2.drawContours(image, [hull2], -1, (168, 100, 168), -1)
        for xx in range(sx):
            for yy in range(sy):
                dist = cv2.pointPolygonTest(hull, (xx, yy), False)
                dist_inside = cv2.pointPolygonTest(hull2, (xx, yy), False)
                # 在外嘴唇凸包以内、在内嘴唇凸包以外部分为嘴唇
                if (dist >= 0 and dist_inside < 0):
                    image[yy, xx][0] = lipstick_color[0]
                    image[yy, xx][1] = lipstick_color[1]
                    image[yy, xx][2] += 10

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # 高斯滤波是bfs尝试失败来做边缘柔和处理的
    image = cv2.GaussianBlur(image, (7, 7), 0)
    return image

if __name__ == "__main__":
    input_image_path = "test.jpg"  # 输入图像.jpg
    lipstick_color = [175, 150, 0]  # 嘴唇颜色
    image_output = feature(input_image_path, lipstick_color)  # 处理图像
    cv2.imshow("Output", image_output)  # 显示
    cv2.imwrite("process2+" + input_image_path, image_output)  # 保存
    cv2.waitKey(0)