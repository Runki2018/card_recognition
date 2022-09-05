import cv2
import numpy as np
from imutils.perspective import four_point_transform


def img_show(img, window="image"):
    """快速观察图片"""
    cv2.namedWindow(window, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow(window, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def img_draw_bbox(img, tl, br, color=(0, 255, 0), thickness=2, show=True):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.rectangle(img, tl, br, color=color, thickness=thickness)
    if show:
        img_show(img)

    return img


def img_resize(img, w=1600, h=2400, interpolation=cv2.INTER_AREA):
    """
    缩放图像
        https://www.cnblogs.com/lfri/p/10596530.html

    :param img:  原图
    :param w: 缩放后图像的宽度
    :param h: 缩放后图像的高度
    :param interpolation: 插值方法
    :return: 缩放后的图像
    """
    img = cv2.resize(img, dsize=(w, h), interpolation=interpolation)
    return img


def find_contours_loc(img, dif_pixel=6):
    """
        获取轮廓定位, 返回每个矩形定位符的中心点和宽高
    :param img: 灰度图
    :param dif_pixel:  各轮廓的中心坐标相差至少dif_pixel个像素, 用于去除边框。也可以用IOU来去除重复框。
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]
    img_area = h * w

    img_show(img)
    res = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)  # 返回外部轮廓，描述点尽可能少
    contours = res[1] if cv2.__version__.split('.')[0] == '3' else res[0]
    # contours = res[1]   # opencv 3.0+
    # contours = res[0]   # opencv 4.0+
    # 观察轮廓检测情况：
    # cv2.drawContours(img, contours, -1, 127, 5), img_show(img)

    locations = []
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按轮廓面积倒序
        for cnt in contours:

            area = cv2.contourArea(cnt)
            if area / img_area > 0.8:
                continue  # 去除非常大的框

            # 寻找闭合轮廓， 近似轮廓取值为10%=0.1， 越大越接近矩形，越小越接近物体原本轮廓
            perimeter = 0.1 * cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, perimeter, closed=True)
            n_points = len(approx)
            if n_points < 4:
                continue

            approx = np.array(approx).reshape(n_points, 2)

            x, y, w, h = cv2.boundingRect(approx)
            center = (int(x + w/2), int(y + h/2))

            if len(locations) == 0:
                locations.append([])
                locations[-1].extend([center[0], center[1], w, h])
            else:
                flag = True  # 防止一个框重复，出现内外边框各一个
                for xx, yy, _, _ in locations:
                    dif = abs(xx-center[0]) + abs(yy-center[1])
                    if dif < dif_pixel:
                        flag = False
                        break
                if flag:
                    locations.append([])
                    locations[-1].extend([center[0], center[1], w, h])
        # 用于观察轮廓识别情况：
            img = img_draw_bbox(img, (x, y), (x+w, y+h), show=False)
        img_show(img)

    return img, locations


def cut_card(img, circle_locations, long_side=1600):
    """根据四个圆点定位符裁剪出答题卡区域"""
    locations = np.array(circle_locations)
    locations = sort_box_locations(locations)
    w_mean = int(locations[:, 2].mean() / 2)
    h_mean = int(locations[:, 3].mean() / 2)
    locations[0, :2] -= w_mean, h_mean
    locations[1, :2] += w_mean, -h_mean
    locations[2, :2] += -w_mean, h_mean
    locations[3, :2] += w_mean, h_mean

    # 裁剪，并将长边缩放为指定大小
    img_crop = four_point_transform(img, locations[:, :2])
    img_crop, factor = card_resize(img_crop, long_side=long_side)
    w_mean *= factor
    h_mean *= factor

    img_show(img_crop)
    return img_crop, w_mean, h_mean


def card_resize(img, long_side=1600):
    h, w = img.shape[0], img.shape[1]
    if w > h:
        factor = long_side / w
        img = cv2.resize(img, (long_side, int(h * factor)), cv2.INTER_LANCZOS4)
    else:
        factor = long_side / h
        img = cv2.resize(img, (int(w * factor), long_side), cv2.INTER_LANCZOS4)
    return img, factor


def sort_box_locations(locations):
    """ 对方形定位符排序, 先上到下，再左到右排序  """

    locations = np.array(locations)
    xy = locations[:, :2].copy()
    xy[:, 1] *= 10  # 提高y值权重， 保证从上到下排序

    xy_sum = xy[:, :2].sum(1)  # 度量每个坐标的大小  x1+y1
    locations = locations[xy_sum.argsort()]

    return locations


def getFeatureMap(img_card, iterations=2):
    feature_map = img_card.copy()  # 用于检查答题卡填涂框的特征图
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(6, 6))  # 圆形卷积和
    feature_map = cv2.dilate(feature_map, kernel, iterations=iterations)  # 先膨胀(或运算), 去除黑色前景，如边框和小格子
    feature_map = cv2.erode(feature_map, kernel, iterations=iterations)  # 腐蚀（与运算），还原黑色圆圈定位符的大小
    feature_map = cv2.adaptiveThreshold(feature_map, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7)
    return feature_map


def MatchRegion(img, template, method_id=0):
    methods_list = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    method = eval(methods_list[method_id])  # 解析相应变量字符串， 得到对应的方法号（int）， eg: eval('cv2.TM_CCOEFF') -> 4

    h, w = template.shape[0], template.shape[1]

    # 模板匹配，获取目标框的左上角点和右下角点坐标
    res = cv2.matchTemplate(image=img, templ=template, method=method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)  # 模板的右下角点可以通过左上角点 + 模板宽高得到

    # 画出识别结果
    img_draw_bbox(img, top_left, bottom_right, show=True)

    return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]


def get_studentID(id_list):
    id_list = id_list[::-1]  # 倒序
    student_id = 0
    for i, n in enumerate(id_list):
        student_id += n * (10**i)
    return student_id


def get_mark(correct_answer, student_answer, score_weight=1.0):
    """

    :param correct_answer: 正确答案
    :param student_answer: 学生答案
    :param score_weight: 每道题的分值
    :return:
    """
    # correct_answer = np.array(correct_answer)
    # student_answer = np.array(student_answer)
    # mark = (correct_answer == student_answer).sum() * score
    mark = 0
    for a1, a2 in zip(correct_answer, student_answer):
        if a1 == a2:
            mark += score_weight
    return mark

