# coding: utf-8

import os
import sys
import asyncio
import signal
from pathlib import Path
from platform import system
from typing import List

import typer
import colorama
import keyboard

from CapsWriterOffline.config import ClientConfig as Config
from CapsWriterOffline.util.client_cosmic import console, Cosmic
from CapsWriterOffline.util.client_stream import stream_open, stream_close
from CapsWriterOffline.util.client_shortcut_handler import shortcut_handler
from CapsWriterOffline.util.client_recv_result import recv_result
from CapsWriterOffline.util.client_show_tips import show_mic_tips, show_file_tips
from CapsWriterOffline.util.client_hot_update import update_hot_all, observe_hot

from CapsWriterOffline.util.client_transcribe import transcribe
from CapsWriterOffline.util.client_adjust_srt import adjust_srt

from CapsWriterOffline.util.empty_working_set import empty_current_working_set

# 确保根目录位置正确，用相对路径加载模型
BASE_DIR = os.path.dirname(__file__)
os.chdir(BASE_DIR)

# 确保终端能使用 ANSI 控制字符
colorama.init()

# MacOS 的权限设置
if system() == 'Darwin' and not sys.argv[1:]:
    if os.getuid() != 0:
        print('在 MacOS 上需要以管理员启动客户端才能监听键盘活动，请 sudo 启动')
        input('按回车退出')
        sys.exit()
    else:
        os.umask(0o000)


async def main_mic():
    Cosmic.loop = asyncio.get_event_loop()
    Cosmic.queue_in = asyncio.Queue()
    Cosmic.queue_out = asyncio.Queue()

    show_mic_tips()

    # 更新热词
    update_hot_all()

    # 实时更新热词
    observer = observe_hot()

    # 打开音频流
    Cosmic.stream = stream_open()

    # Ctrl-C 关闭音频流，触发自动重启
    signal.signal(signal.SIGINT, stream_close)

    # 绑定按键
    keyboard.hook_key(Config.shortcut, shortcut_handler)

    # 清空物理内存工作集
    if system() == 'Windows':
        empty_current_working_set()

    # 接收结果
    while True:
        async for value in recv_result():
            yield value


async def main_file(files: List[Path]):
    show_file_tips()

    for file in files:
        if file.suffix in ['.txt', '.json', 'srt']:
            adjust_srt(file)
        else:
            await transcribe(file)

    if Cosmic.websocket:
        await Cosmic.websocket.close()
    input('\n按回车退出\n')


def init_mic():
    try:
        asyncio.run(main_mic())
    except KeyboardInterrupt:
        console.print(f'再见！')
    finally:
        print('...')


def init_file(files: List[Path]):
    """
    用 CapsWriter Server 转录音视频文件，生成 srt 字幕
    """
    try:
        asyncio.run(main_file(files))
    except KeyboardInterrupt:
        console.print(f'再见！')
        sys.exit()


if __name__ == "__main__":
    # 如果参数传入文件，那就转录文件
    # 如果没有多余参数，就从麦克风输入
    if sys.argv[1:]:
        typer.run(init_file)
    else:
        init_mic()
