"""
输入采集处理，如采集屏幕，音频，视频等
"""
import ctypes
import ctypes.wintypes
import time

import cv2
import dxcam
import numpy
import psutil
import win32api
import win32con
import win32gui
import win32process
from PIL import ImageGrab


def enum_hwnd(hwnd, mouse):
    hwnd_map = {}
    if (win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(
            hwnd)):  # 获取当前windows已打开的窗口
        hwnd_map.update({hwnd: win32gui.GetWindowText(hwnd)})  #
        for h, t in hwnd_map.items():
            print("【窗口名称】：{}".format(t), " 【句柄信息】：{}".format(h))


def enum_all_window():
    win32gui.EnumWindows(enum_hwnd, 0)  # 枚举所有窗体


def get_window_rect(hwnd):
    """
    获取窗口的实际坐标，返回 (right, top, left, bottom)
    自从vista系统开始，窗口有毛玻璃特效边框，而 win32gui.GetWindowRect 并没有计算上这部分，所以获取的值会偏小
    参考： https://stackoverflow.com/questions/3192232/getwindowrect-too-small-on-windows-7
    :param hwnd: 窗口句柄
    :return:
    """
    try:
        rect = ctypes.wintypes.RECT()
        extended_frame_bounds = 9
        ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(hwnd),
                                                   ctypes.wintypes.DWORD(extended_frame_bounds),
                                                   ctypes.byref(rect),
                                                   ctypes.sizeof(rect))
        return rect.left, rect.top, rect.right, rect.bottom
    except WindowsError:
        print('window error.')


def get_top_window_hwnd():
    """
    获取顶层激活的窗口句柄
    :return:
    """
    # 获取当前激活的窗口句柄
    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd)  # 窗口标题
    rect = win32gui.GetWindowRect(hwnd)  # 窗口坐标
    print('hwnd: {}, title: {}, rect: {}'.format(hwnd, title, rect))
    return hwnd


def get_window_process_info(hwnd):
    # 获取指定窗口的线程id和进程id
    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
    # 获取该窗口句柄的进程信息
    process = psutil.Process(pid)
    print('name: {}, rect: {}, exe: {}, create_time: {}'
          .format(process.name(), get_window_rect(hwnd), process.exe(), process.create_time()))
    return process


def screenshot_by_dxcam(region=None) -> numpy.ndarray | None:
    """
    获取屏幕渲染帧，如果没有新帧，则返回为空，速度非常快，支持240Hz
    参考：<https://github.com/ra1nty/DXcam>
    """
    # DXCamera instances for: [monitor0, GPU0]，显示器1，GPU0
    camera = dxcam.create(device_idx=0, output_idx=0, output_color="BGR")
    frame = camera.grab(region=region)
    return frame


def screenshot_by_pil(region=None):
    # 获取当前分辨率下的屏幕尺寸
    width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    return ImageGrab.grab(bbox=(0, 0, width, height))


def capture_screen_by_dxcam():
    """使用cv2和dxcam进行录屏"""
    target_fps = 120
    camera = dxcam.create(output_idx=0, output_color="BGR")
    camera.start(target_fps=target_fps, video_mode=True)
    writer = cv2.VideoWriter(
        "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (1920, 1080)
    )
    for i in range(600):
        writer.write(camera.get_latest_frame())
    camera.stop()
    writer.release()


if __name__ == '__main__':
    time.sleep(5)
    top_hwnd = get_top_window_hwnd()
    get_window_process_info(top_hwnd)
