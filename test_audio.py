import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import simpleaudio as sa
from webui import tts_fn

input_text = "很高兴见到你，旅行者！"
result = tts_fn(
    text=input_text,
    speaker="keqing",
    sdp_ratio=0.5,
    noise_scale=0.6,
    noise_scale_w=0.9,
    length_scale=1,
    language="ZH",
    reference_audio="",
    emotion="",
    prompt_mode="",
    style_text=None,
    style_weight=0)
sample_rate = result[1][0]
audio_data = result[1][1]
# 创建一个WaveObject，simpleaudio需要音频数据为字节格式
wave_obj = sa.WaveObject(audio_data.tobytes(), 1, 2, sample_rate)

# 播放音频
play_obj = wave_obj.play()
play_obj.wait_done()  # 等待音频播放完成
