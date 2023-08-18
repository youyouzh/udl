"""
基于图像识别的操作交互封装
"""
# -*- coding: utf-8 -*-
# 获取屏幕信息，并通过视觉方法标定手牌与按钮位置，仿真鼠标点击操作输出
import os
import time
from enum import Enum
from typing import List, Tuple

import cv2
import numpy
import pyautogui
import numpy as np
import win32gui

from practice.image_template_matcher import ImageHomographyTransform, ImageTemplateMatcher
from practice.input_collect import get_window_rect, screenshot_by_dxcam

pyautogui.PAUSE = 0         # 函数执行后暂停时间
pyautogui.FAILSAFE = False   # 开启鼠标移动到左上角自动退出

DEBUG = False               # 是否显示检测中间结果


class Layout:
    size = (1920, 1080)                                     # 界面长宽
    duanWeiChang = (1348, 321)                              # 段位场按钮
    menuButtons = [(1382, 406), (1382, 573), (1382, 740),
                   (1383, 885), (1393, 813)]   # 铜/银/金之间按钮
    tileSize = (95, 152)                                     # 自己牌的大小


# 图片样例，从游戏中截取的图片，用于定位、匹配查找等作用
class ImageSample(Enum, str):
    MENU = 'menu'  # 主菜单
    CHI = 'chi'    # 吃按钮
    PENG = 'peng'  # 碰按钮
    GANG = 'gang'  # 杠按钮
    HU = 'hu'      # 胡按钮
    ZIMO = 'zimo'  # 自摸按钮
    TIAOGUO = 'tiaoguo'  # 跳过按钮
    LIQI = 'liqi'  # 立直按钮

    @staticmethod
    def load(name: Enum):
        root = os.path.dirname(__file__)
        path = os.path.join(root, 'images', name.name)
        return cv2.imread(path)


# 窗口映射，基于基准图像和当前运行中游戏进行坐标映射，需要将游戏切换到游戏首页
class GameWindow(ImageHomographyTransform):

    def __init__(self):
        # 获取游戏窗口句柄和坐标
        self.hwnd = win32gui.FindWindow(None, '雀魂麻将')
        self.window_rect = get_window_rect(self.hwnd)
        # 截图并计算映射矩阵
        super().__init__(ImageSample.load(ImageSample.MENU), self.screenshot())

    def screenshot(self):
        screenshot = screenshot_by_dxcam(region=self.window_rect)
        return numpy.ndarray(screenshot)

    def locate_image(self, image):
        # 定位按钮的位置
        screenshot = self.screenshot()
        # 使用cv2模板匹配
        return super(ImageTemplateMatcher, self).locate_image().locate(screenshot, image, confidence=0.9)


class GUIInterface(object):

    def __init__(self):
        self.game_window = GameWindow()  # 游戏窗口
        self.image_sample_cache = {}     # 示例图片缓存，一次性加载

    def load_image_sample(self, image_sample: ImageSample):
        if image_sample.name not in self.image_sample_cache:
            # 首次加载图片到缓存
            self.image_sample_cache[image_sample.name] = ImageSample.load(image_sample)
        return self.image_sample_cache.get(image_sample.name)

    # 点击按钮，比如吃、碰、杠、胡、自摸
    def click_button(self, button_image: ImageSample):
        button_image = self.load_image_sample(button_image)
        box = self.game_window.locate_image(button_image)

        x0, y0 = np.int32(PosTransfer([0, 0], self.M))
        x1, y1 = np.int32(PosTransfer(Layout.size, self.M))
        zoom = (x1-x0)/Layout.size[0]
        n, m, _ = buttonImg.shape
        n = int(n*zoom)
        m = int(m*zoom)
        templ = cv2.resize(buttonImg, (m, n))
        x0, y0 = np.int32(PosTransfer([595, 557], self.M))
        x1, y1 = np.int32(PosTransfer([1508, 912], self.M))
        img = screenShot()[y0:y1, x0:x1, :]
        T = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF, mask=templ.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        if DEBUG:
            T = np.exp((1-T/T.max())*10)
            T = T/T.max()
            cv2.imshow('T', T)
            cv2.waitKey(0)
        dst = img[y:y+n, x:x+m].copy()
        dst[templ == 0] = 0
        if Similarity(templ, dst) >= similarityThreshold:
            pyautogui.click(x=x+x0+m//2, y=y+y0+n//2, duration=0.2)
            time.sleep(0.5)
            pyautogui.moveTo(x=self.waitPos[0], y=self.waitPos[1])

    def forceTiaoGuo(self):
        # 如果跳过按钮在屏幕上则强制点跳过，否则NoEffect
        self.click_button(self.tiaoguoImg, similarityThreshold=0.7)

    def actionDiscardTile(self, tile: str):
        L = self._getHandTiles()
        for t, (x, y) in L:
            if t == tile:
                pyautogui.moveTo(x=x, y=y)
                time.sleep(0.3)
                pyautogui.click(x=x, y=y, button='left')
                time.sleep(1)
                # out of screen
                pyautogui.moveTo(x=self.waitPos[0], y=self.waitPos[1])
                return True
        raise Exception(
            'GUIInterface.discardTile tile not found. L:', L, 'tile:', tile)
        return False

    def actionChiPengGang(self, type_: Operation, tiles: List[str]):
        if type_ == Operation.NoEffect:
            self.click_button(self.tiaoguoImg)
        elif type_ == Operation.Chi:
            self.click_button(self.chiImg)
        elif type_ == Operation.Peng:
            self.click_button(self.pengImg)
        elif type_ in (Operation.MingGang, Operation.JiaGang):
            self.click_button(self.gangImg)

    def actionLiqi(self, tile: str):
        self.click_button(self.liqiImg)
        time.sleep(0.5)
        self.actionDiscardTile(tile)

    def actionHu(self):
        self.click_button(self.huImg)

    def actionZimo(self):
        self.click_button(self.zimoImg)

    def calibrateMenu(self):
        # if the browser is on the initial menu, set self.M and return to True
        # if not return False
        self.M = getHomographyMatrix(self.menuImg, screenShot(), threshold=0.7)
        result = type(self.M) != type(None)
        if result:
            self.waitPos = np.int32(PosTransfer([100, 100], self.M))
        return result

    def _getHandTiles(self) -> List[Tuple[str, Tuple[int, int]]]:
        # return a list of my tiles' position
        result = []
        assert(type(self.M) != type(None))
        screen_img1 = screenShot()
        time.sleep(0.5)
        screen_img2 = screenShot()
        screen_img = np.minimum(screen_img1, screen_img2)  # 消除高光动画
        img = screen_img.copy()     # for calculation
        start = np.int32(PosTransfer([235, 1002], self.M))
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 110
        tileThreshold = np.int32(0.7*(PosTransfer(Layout.tileSize, self.M)-O))
        fail = 0
        maxFail = np.int32(PosTransfer([100, 0], self.M)-O)[0]
        i = 0
        while fail < maxFail:
            x, y = start[0]+i, start[1]
            if all(img[y, x, :] > colorThreshold):
                fail = 0
                img[y, x, :] = colorThreshold
                retval, image, mask, rect = cv2.floodFill(
                    image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                    loDiff=(0, 0, 0), upDiff=tuple([255-colorThreshold]*3), flags=cv2.FLOODFILL_FIXED_RANGE)
                x, y, dx, dy = rect
                if dx > tileThreshold[0] and dy > tileThreshold[1]:
                    tile_img = screen_img[y:y+dy, x:x+dx, :]
                    tileStr = self.classify(tile_img)
                    result.append((tileStr, (x+dx//2, y+dy//2)))
                    i = x+dx-start[0]
            else:
                fail += 1
            i += 1
        return result


    def clickCandidateMeld(self, tiles: List[str]):
        # 有多种不同的吃碰方法，二次点击选择
        assert(len(tiles) == 2)
        # find all combination tiles
        result = []
        assert(type(self.M) != type(None))
        screen_img = screenShot()
        img = screen_img.copy()     # for calculation
        start = np.int32(PosTransfer([960, 753], self.M))
        leftBound = rightBound = start[0]
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 200
        tileThreshold = np.int32(0.7*(PosTransfer((78, 106), self.M)-O))
        maxFail = np.int32(PosTransfer([60, 0], self.M)-O)[0]
        for offset in [-1, 1]:
            #从中间向左右两个方向扫描
            i = 0
            while True:
                x, y = start[0]+i*offset, start[1]
                if offset == -1 and x < leftBound-maxFail:
                    break
                if offset == 1 and x > rightBound+maxFail:
                    break
                if all(img[y, x, :] > colorThreshold):
                    img[y, x, :] = colorThreshold
                    retval, image, mask, rect = cv2.floodFill(
                        image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                        loDiff=(0, 0, 0), upDiff=tuple([255-colorThreshold]*3), flags=cv2.FLOODFILL_FIXED_RANGE)
                    x, y, dx, dy = rect
                    if dx > tileThreshold[0] and dy > tileThreshold[1]:
                        tile_img = screen_img[y:y+dy, x:x+dx, :]
                        tileStr = self.classify(tile_img)
                        result.append((tileStr, (x+dx//2, y+dy//2)))
                        leftBound = min(leftBound, x)
                        rightBound = max(rightBound, x+dx)
                i += 1
        result = sorted(result, key=lambda x: x[1][0])
        if len(result) == 0:
            return True  # 其他人先抢先Meld了！
        print('clickCandidateMeld tiles:', result)
        assert(len(result) % 2 == 0)
        for i in range(0, len(result), 2):
            x, y = result[i][1]
            if tuple(sorted([result[i][0], result[i+1][0]])) == tiles:
                pyautogui.click(x=x, y=y, duration=0.2)
                time.sleep(1)
                pyautogui.moveTo(x=self.waitPos[0], y=self.waitPos[1])
                return True
        raise Exception('combination not found, tiles:',
                        tiles, ' combination:', result)
        return False

    def actionReturnToMenu(self):
        # 在终局以后点击确定跳转回菜单主界面
        x, y = np.int32(PosTransfer((1785, 1003), self.M))  # 终局确认按钮
        while True:
            time.sleep(5)
            x0, y0 = np.int32(PosTransfer([0, 0], self.M))
            x1, y1 = np.int32(PosTransfer(Layout.size, self.M))
            img = screenShot()
            S = Similarity(self.menuImg, img[y0:y1, x0:x1, :])
            if S > 0.5:
                return True
            else:
                print('Similarity:', S)
                pyautogui.click(x=x, y=y, duration=0.5)

    def actionBeginGame(self, level: int):
        # 从开始界面点击匹配对局, level=0~4 (铜/银/金/玉/王座之间)
        time.sleep(2)
        x, y = np.int32(PosTransfer(Layout.duanWeiChang, self.M))
        pyautogui.click(x, y)
        time.sleep(2)
        if level == 4:
            # 王座之间在屏幕外面需要先拖一下
            x, y = np.int32(PosTransfer(Layout.menuButtons[2], self.M))
            pyautogui.moveTo(x, y)
            time.sleep(1.5)
            x, y = np.int32(PosTransfer(Layout.menuButtons[0], self.M))
            pyautogui.dragTo(x, y)
            time.sleep(1.5)
        x, y = np.int32(PosTransfer(Layout.menuButtons[level], self.M))
        pyautogui.click(x, y)
        time.sleep(2)
        x, y = np.int32(PosTransfer(Layout.menuButtons[0], self.M))  # 四人东
        pyautogui.click(x, y)