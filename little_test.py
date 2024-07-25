import re

text = "Assistant: 和你在一起，时间变短了，动作变慢了，仿佛连呼吸和心跳都有了双重意义——毕竟「璃月七星」，每一个字都含金量极高、价值连城。能够成为你的朋友，实在是我的荣幸。</s>"
pattern = r"Assistant: (.*?)</s>"

match = re.search(pattern, text)
if match:
    extracted_text = match.group(1)
    print(extracted_text)
else:
    print("没有找到匹配的内容")
