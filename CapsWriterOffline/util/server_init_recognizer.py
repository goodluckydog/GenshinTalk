import time
import sherpa_onnx
from multiprocessing import Queue
import signal
from platform import system
from CapsWriterOffline.config import ServerConfig as Config
from CapsWriterOffline.config import ParaformerArgs, ModelPaths
from CapsWriterOffline.util.server_cosmic import console
from CapsWriterOffline.util.server_recognize import recognize
from CapsWriterOffline.util.empty_working_set import empty_current_working_set



def disable_jieba_debug():
    # 关闭 jieba 的 debug
    import jieba
    import logging
    jieba.setLogLevel(logging.INFO)


def init_recognizer(queue_in: Queue, queue_out: Queue):

    # Ctrl-C 退出
    signal.signal(signal.SIGINT, lambda signum, frame: exit())

    # 导入模块
    with console.status("载入模块中…", spinner="bouncingBall", spinner_style="yellow"):
        import sherpa_onnx
        from funasr_onnx import CT_Transformer
        disable_jieba_debug()
    console.print('[green4]模块加载完成', end='\n\n')

    # 载入语音模型
    console.print('[yellow]语音模型载入中', end='\r'); t1 = time.time()
    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        **{key: value for key, value in ParaformerArgs.__dict__.items() if not key.startswith('_')}
    )
    console.print(f'[green4]语音模型载入完成', end='\n\n')

    # 载入标点模型
    punc_model = None
    console.print('[yellow]标点模型载入中', end='\r')
    punc_model = CT_Transformer(ModelPaths.punc_model_dir, quantize=True)
    console.print(f'[green4]标点模型载入完成', end='\n\n')

    console.print(f'模型加载耗时 {time.time() - t1 :.2f}s', end='\n\n')

    # 清空物理内存工作集
    if system() == 'Windows':
        empty_current_working_set()

    queue_out.put(True)  # 通知主进程加载完了

    while True:
        task = queue_in.get()       # 从队列中获取任务消息
        result = recognize(recognizer, punc_model, task)   # 执行识别
        queue_out.put(result)      # 返回结果

