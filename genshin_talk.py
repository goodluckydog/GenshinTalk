# 首先基于语义识别模块识别语音，实现第一层的输入
# 将语音识别结果作为数据传入大模型，得到“角色“输出
# 将输出传入BertVits模型, 实现音频输出

import time

import yaml
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import simpleaudio as sa
from webui import tts_fn  # 音频输出

import asyncio
from CapsWriterOffline.config import ClientConfig as Config
from CapsWriterOffline.util.client_cosmic import console, Cosmic
from CapsWriterOffline.util.client_stream import stream_open, stream_close
from CapsWriterOffline.util.client_shortcut_handler import shortcut_handler
from CapsWriterOffline.util.client_recv_result import recv_result
from CapsWriterOffline.util.client_show_tips import show_mic_tips, show_file_tips
from CapsWriterOffline.util.client_hot_update import update_hot_all, observe_hot
import signal
from platform import system
import keyboard

from CapsWriterOffline.util.empty_working_set import empty_current_working_set
import re


def get_config(file_path="D:\learn\ScientificResearch\LLM\Bert-VITS2-2.3\LLM\config.yaml"):
    # 打开并读取YAML文件
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class GenshinTalk:
    def __init__(self, file_path):
        """
        设计一个与原神角色对话的程序，增强玩游戏时的体验
        整个程序可以解构为三个功能
            1.语音识别: 基于CapsWriter实现, github地址:https://github.com/HaujetZhao/CapsWriter-Offline
            2.大模型数据处理: 基于Atom模型+LoRA微调实现, llama中文社区:https://github.com/LlamaFamily/Llama-Chinese(我还没进QQ群，不知道答案是啥...)
            3.声纹输出: 基于原神角色音频数据训练, 使用BertVITS2完成声音合成, 参考视频:https://www.bilibili.com/read/cv27647393/
        """
        self._my_text = None  # 用于监控文本的变量
        self.messages = get_config(file_path)

        self.tokenizer, self.LLM_model = self.loadLLM()

    def start(self):
        asyncio.run(self.main_mic())

    async def main_mic(self):
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
            async for text in recv_result():
                self.speak(self.LLM_predict(text))

    def loadLLM(self):
        """
        加载大模型
        :return:
        """
        print("*****************开始加载大模型*****************")
        start_time = time.time()
        checkpoint = self.messages["LLM_checkpoint"]
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # 加载基座模型
        atom_model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cuda:0", load_in_8bit=True)
        # 加载自己微调后的大模型
        model = PeftModel.from_pretrained(atom_model, self.messages["LoRA_save_path"], is_trainable=False)
        print("从{}加载断点成功, 已加载大模型!".format(self.messages["LoRA_save_path"]))
        print("*****************总耗时:{:.2f}s*****************".format(time.time() - start_time))
        return tokenizer, model

    def LLM_predict(self, input_text):
        print("*****************大模型开始推理*****************")
        start_time = time.time()
        input_list = "<s>Human: " + input_text + "\n" + "</s><s>Assistant:"
        input_message = self.tokenizer(input_list, return_tensors="pt", add_special_tokens=False)  # 对输入进行处理
        generate_input = {
            "input_ids": input_message.input_ids.to('cuda:0'),
            "attention_mask": input_message.attention_mask.to('cuda:0'),
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        generate_ids = self.LLM_model.generate(**generate_input)
        text = self.tokenizer.decode(generate_ids[0])  # 将输出转化为文本

        pattern = r"Assistant: (.*?)</s>"  # 利用正则化提取对应的输出文本
        match = re.search(pattern, text)
        if match:
            extracted_text = match.group(1)
            text = extracted_text
        else:
            print("没有找到匹配的内容")
        print("输出文本:{}".format(text))
        print("*****************推理结束, 耗时:{:.2f}s*****************".format(time.time() - start_time))
        return text

    def speak(self, input_text):
        print("*****************开始音频处理*****************")
        start_time = time.time()
        result = tts_fn(
            text=input_text,
            speaker=self.messages["speaker"],
            sdp_ratio=self.messages["sdp_ratio"],
            noise_scale=self.messages["noise_scale"],
            noise_scale_w=self.messages["noise_scale_w"],
            length_scale=self.messages["length_scale"],
            language=self.messages["language"],
            reference_audio=self.messages["reference_audio"],
            emotion=self.messages["emotion"],
            prompt_mode=self.messages["prompt_mode"],
            style_text=None,
            style_weight=0)
        sample_rate = result[1][0]
        audio_data = result[1][1]
        # 创建一个WaveObject，simpleaudio需要音频数据为字节格式
        wave_obj = sa.WaveObject(audio_data.tobytes(), 1, 2, sample_rate)
        print("*****************音频处理完毕, 耗时:{:.2f}s*****************".format(time.time() - start_time))

        # 播放音频
        play_obj = wave_obj.play()
        play_obj.wait_done()  # 等待音频播放完成


talk_model = GenshinTalk(file_path="D:\learn\ScientificResearch\LLM\Bert-VITS2-2.3\LLM\config.yaml")
talk_model.start()
