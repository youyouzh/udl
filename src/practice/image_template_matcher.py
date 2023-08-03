"""
给定一个源图片，一般较大，在这个图片中搜索另外一个图片的位置
算是目标检测的弱化版本，比如从图片中截取一部分，这部分可能会做拉伸缩放旋转等操作后，还能从原来的图片中定位位置
应用于RPA场景，在和屏幕交互过程中，我们对某个交互区域截图（比如按钮），然后通过目标检测获取其坐标然后点击实现
这种场景只用考虑缩放和拉伸，但是对于如果某些应用宽高变化后排版会发生变化则不适用，尤其是文本匹配的场景。
"""
import collections

import cv2
import numpy
import base.u_log as log

GRAYSCALE_DEFAULT = True
DEBUG = True  # 是否显示检测中间结果
Box = collections.namedtuple('Box', 'left top width height')


def load_image_cv2(img, grayscale=False) -> numpy.ndarray:
    """
    load images if given filename, or convert as needed to opencv
    Alpha layer just causes failures at this point, so flatten to RGB.
    RGBA: load with -1 * cv2.CV_LOAD_IMAGE_COLOR to preserve alpha
    to matchTemplate, need template and image to be the same wrt having alpha
    """
    if isinstance(img, str):
        # The function imread loads an image from the specified file and
        # returns it. If the image cannot be read (because of missing
        # file, improper permissions, unsupported or invalid format),
        # the function returns an empty matrix
        # http://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html
        if grayscale:
            img_cv = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        else:
            img_cv = cv2.imread(img, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise IOError(
                "Failed to read %s because file is missing, "
                "has improper permissions, or is an "
                "unsupported or invalid format" % img
            )
    elif isinstance(img, numpy.ndarray):
        # don't try to convert an already-gray image to gray
        if grayscale and len(img.shape) == 3:  # and img.shape[2] == 3:
            img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_cv = img
    elif hasattr(img, 'convert'):
        # assume its a PIL.Image, convert to cv format
        img_array = numpy.array(img.convert('RGB'))
        img_cv = img_array[:, :, ::-1].copy()  # -1 does RGB -> BGR
        if grayscale:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError('expected an image filename, OpenCV numpy array, or PIL image')
    return img_cv


def pos_transform(position, transform_mat: numpy.ndarray) -> numpy.ndarray:
    """
    通过转换矩阵来变换坐标，position示例: (200, 200)
    """
    assert (len(position) == 2)
    return cv2.perspectiveTransform(numpy.float32([[position]]), transform_mat)[0][0]


def image_position_transform(image: numpy.ndarray, transform_mat: numpy.ndarray) -> numpy.ndarray:
    h, w, d = image.shape
    pts = numpy.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, transform_mat)


# 使用opencv中的match_template方法实现定位你
class ImageTemplateMatcher(object):

    def __init__(self):
        self.grayscale_default = True
        self.debug_mode = False
        pass

    def locate_all(self, haystack_image, needle_image, limit=10000,
                   region=None, step=1, confidence=0.95):
        """
        faster but more memory-intensive than pure python
        step 2 skips every other row and column = ~3x faster but prone to miss;
            to compensate, the algorithm automatically reduces the confidence
            threshold by 5% (which helps but will not avoid all misses).
        limitations:
          - OpenCV 3.x & python 3.x not tested
          - RGBA images are treated as RBG (ignores alpha channel)
        返回box对象： (left, top, width, height)
        """
        confidence = float(confidence)

        needle_image = load_image_cv2(needle_image, self.grayscale_default)
        needle_height, needle_width = needle_image.shape[:2]
        haystack_image = load_image_cv2(haystack_image, self.grayscale_default)

        if region:
            haystack_image = haystack_image[region[1]: region[1] + region[3], region[0]: region[0] + region[2]]
        else:
            region = (0, 0)  # full image; these values used in the yield statement
        if haystack_image.shape[0] < needle_image.shape[0] or haystack_image.shape[1] < needle_image.shape[1]:
            # avoid semi-cryptic OpenCV error below if bad size
            raise ValueError('needle dimension(s) exceed the haystack image or region dimensions')

        if step == 2:
            confidence *= 0.95
            needle_image = needle_image[::step, ::step]
            haystack_image = haystack_image[::step, ::step]
        else:
            step = 1

        # get all matches at once, credit:
        # https://stackoverflow.com/questions/7670112/finding-a-subimage-inside-a-numpy-image/9253805#9253805
        result = cv2.matchTemplate(haystack_image, needle_image, cv2.TM_CCOEFF_NORMED)
        match_indices = numpy.arange(result.size)[(result > confidence).flatten()]
        matches = numpy.unravel_index(match_indices[:limit], result.shape)

        if len(matches[0]) == 0:
            # 没有找到返回None
            return None

        # use a generator for API consistency:
        matchx = matches[1] * step + region[0]  # vectorized
        matchy = matches[0] * step + region[1]
        for x, y in zip(matchx, matchy):
            yield Box(x, y, needle_width, needle_height)

    def locate(self, haystack_image, needle_image, **kwargs):
        # Note: The gymnastics in this function is because we want to make sure to exhaust
        # the iterator so that the needle and haystack files are closed in locateAll.
        kwargs['limit'] = 1
        points = tuple(self.locate_all(haystack_image, needle_image, **kwargs))
        if len(points) > 0:
            return points[0]
        return None

    def evaluate(self):
        base_image_path = r'data\images\base.png'
        target_image_paths = [
            r'data\images\target-person.png',
            r'data\images\target-right.png',
            r'data\images\target-sm.png',
            r'data\images\target-sz.png',
            r'data\images\target-resize-1.png',
            r'data\images\target-resize-2.png',
            r'data\images\target-resize-3.png',
        ]

        base_image = cv2.imread(base_image_path)
        for target_image_path in target_image_paths:
            box = self.locate(base_image_path, target_image_path)
            if box is None:
                log.warn('can not locate result for template: {}'.format(target_image_path))
                continue
            log.info('locate result: {} for template: {}'.format(box, target_image_path))
            cv2.rectangle(base_image, (box.left, box.top),
                          (box.left + box.width, box.top + box.height), (0, 0, 255), 2)
        cv2.imshow('base image', base_image)
        cv2.waitKey(0)


# 使用图片Sift特征来进行匹配
class OrbImageTemplateMatcher(ImageTemplateMatcher):

    def __init__(self):
        super().__init__()
        self.homography_mat = None

    def locate(self, haystack_image, needle_image, **kwargs):
        haystack_image = load_image_cv2(haystack_image, self.grayscale_default)
        needle_image = load_image_cv2(needle_image, self.grayscale_default)

        # 提取sift特征，需要安装相应的库： pip install opencv-contrib-python
        feature_extractor = cv2.ORB_create(nfeatures=5000)
        haystack_kp, haystack_desc = feature_extractor.detectAndCompute(haystack_image, None)
        needle_kp, needle_desc = feature_extractor.detectAndCompute(needle_image, None)

        # FLANN特征匹配器
        index_params = dict(algorithm=6,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        matcher = cv2.FlannBasedMatcher(index_params, searchParams=dict(checks=50))

        # 特征匹配结果
        matches = matcher.knnMatch(haystack_desc, needle_desc, k=2)
        # 只取匹配好的结果
        matches_mask = [[0, 0] for i in range(len(matches))]
        good_matches = []
        for index, match in enumerate(matches):
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    matches_mask[index] = [1, 0]

        log.info('good matches number: {}'.format(len(good_matches)))

        min_match_count = 50
        if len(good_matches) < min_match_count:
            log.error('good matches is less than {}'.format(min_match_count))
            return None

        src_pts = numpy.float32([haystack_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = numpy.float32([needle_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.homography_mat = homography_mat

        # 调试模式展示结果
        if self.debug_mode:
            matches_mask = mask.ravel().tolist()
            dst = image_position_transform(haystack_image, homography_mat)
            line_image = cv2.polylines(needle_image, [numpy.int32(dst)], True, (0, 0, 255), 10, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matches_mask,  # draw only inliers
                               flags=2)
            result = cv2.drawMatches(haystack_image, haystack_kp, line_image, needle_kp,
                                     good_matches, None, **draw_params)
            result = cv2.pyrDown(result)
            cv2.imshow('Result', result)
            cv2.waitKey(1)

        # 返回匹配查找变化后的坐标
        position = image_position_transform(needle_image, homography_mat)
        position = numpy.int32(position.reshape(-1, 2))
        return Box(position[0][0], position[0][1], position[2][0] - position[0][1], position[2][1])


if __name__ == '__main__':
    # template_matcher = ImageTemplateMatcher()
    template_matcher = OrbImageTemplateMatcher()
    template_matcher.grayscale_default = False
    template_matcher.debug_mode = True
    template_matcher.evaluate()
