# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, default_data_collator
import json
from tqdm import tqdm
import datasets
import os
import torch
from transformers import Trainer
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import yaml


def get_config(file_path="D:\learn\ScientificResearch\LLM\Bert-VITS2-2.3\LLM\config.yaml"):
    # 打开并读取YAML文件
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 超参数设定
config = get_config()
# Lora 秩
lora_rank: int = 8
# 最大文本长度
max_seq_length: int = 1024
# 是否从断点继续训练
continue_training: bool = False
# 断点路径，如果从断点继续训练需要传入
pretrain_model_path: str = "./keqing_LoRA"
# 数据集路径
dataset_path = config["dataset_path"]
# 基座模型
checkpoint = "FlagAlpha/Atom-7B-Chat"
# 输出路径
output_dir = "./output"
save_model_path = config["LoRA_save_path"]
batch_size = 2

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 加载基座模型
atom_model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cuda:0", load_in_8bit=True)

# 设定LoRA参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务为语言模型建模
    inference_mode=False,  # 训练模式
    r=lora_rank,  # LoRA矩阵的秩
    lora_alpha=32,  # LoRA矩阵的缩放因子
    lora_dropout=0.1,
    target_modules=['W_pack', 'down_proj', 'o_proj', 'gate_proj', 'up_proj']
)


# 数据预处理:针对LoRA指令微调
def preprocess(tokenizer, config, example, max_seq_length=200):
    """
    args:
    tokenizer：分词器，导入的 Atom 模型分词器
    config：模型配置(就是模型)，导入的 Atom 模型配置
    example: 待处理的样本
    max_seq_length：文本的最大长度(LlaMa2最高支持4096个单词)
    returns：字典，包括 inputs_id 和 seq_len
    """
    # 将 instruction 和 input 按照 Atom SFT 时的格式拼接起来
    prompt = "<s>Human: " + example["instruction"] + example["input"] + "\n" + "</s><s>Assistant:"
    target = example["output"]
    # 使用分词器进行编码，设置 truncation 为 True，避免出现过长的样本
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)  # 会自动添加上开始符和结束符
    target_ids = tokenizer.encode(  # 未添加特殊字符
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    # 加入结束符 EOS
    input_ids = prompt_ids + target_ids + [config.eos_token_id]  # 最终的输入token
    # 将 inputs_ids 和 seq_len 一起传回，后续会根据 seq_len 来切分 inputs 和 labels
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}  # seq_len是问题部分的token长度


# 读取JSON数据并进行数据处理
def read_json(path, max_seq_length=200, model_path="FlagAlpha/Atom-7B-Chat"):
    """
    args:
    path：训练数据路径
    max_seq_length：文本的最大长度
    model_path：模型路径(模型名称), 默认使用Atom-7B-Chat模型
    returns：使用 yield 返回格式化的特征(yield返回可迭代的数据, 方便后续处理)
    """
    # 加载模型的分词器和配置参数
    tokenizer = AutoTokenizer.from_pretrained(
        model_path)
    config = AutoConfig.from_pretrained(
        model_path, device_map='cuda:0')
    # 读取源文件
    with open(path, "r", encoding="utf-8") as json_file:  # 打开文件
        lst = json.load(json_file)  # 加载数据
        print("加载jsonl数据集，数据总量为{}".format(len(lst)))  # 统计样本条数
        # 依次处理每一个样本
        for example in tqdm(lst):
            # 调用上文的预处理函数
            feature = preprocess(tokenizer, config, example, max_seq_length)
            # 通过 yield 返回迭代器
            yield feature


def data_collator(features: list, tokenizer) -> dict:
    """
    args:
    features: 一个批量的数据
    tokenizer：分词器
    returns：格式化的特征
    """
    # 统计 batch 内所有数据的长度，将它们补齐
    len_ids = [len(feature["input_ids"]) for feature in features]
    # 补齐至最大长度
    longest = max(len_ids)
    # 分别存放 input_ids 和 labels
    input_ids = []
    labels_list = []
    # 有的模型没有定义 PAD，那么我们就用 UNK 来代替
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    # 从最长的文本开始处理，可以优化内存使用
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        # labels 是将输入 PAD 之后保留输出的结果，用-100表示遮蔽，并且进行补齐，计算 loss 时会自动忽略 -100
        labels = (
                [-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    # 在第0维进行拼接，也就是组成 batch_size*n*n 的矩阵
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# 自定义 Trainer，继承自 transformers.trainer
class ModifiedTrainer(Trainer):

    # 重写损失计算函数，避免 LLaMA 类模型未定义 loss 的计算
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        # 7B
        loss = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss
        self.loss += loss.item()
        if self.state.global_step % 20 == 0 and self.state.global_step >= 10:
            print("epoch:{}  loss:{}".format(self.state.epoch, self.loss))
            self.loss = 0
        return loss

    # 重写模型保存函数，从而保存模型的 Lora 参数
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            return
        from transformers.trainer import TRAINING_ARGS_NAME
        # 如果输出路径不存在，创建一个
        os.makedirs(output_dir, exist_ok=True)
        # 保存了模型训练的各种超参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # 选出了所有梯度没有被冻结的参数，也就是所有参与更新的 Lora 参数
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        # 保存所有 Lora 参数
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


# 加载数据集
try:
    # 调用上述定义函数生成迭代器
    dataset = datasets.Dataset.from_generator(
        lambda: read_json(dataset_path)
    )
    print("从{}加载数据集成功".format(dataset_path))
except Exception as e:
    print("从{}加载数据集失败".format(dataset_path))
    print("错误信息为：")
    print(e.__repr__())
    raise e

# 是否从断点继续训练
# 源点训练
if not continue_training:  # 不从断点开始训练
    # 对基座模型进行 Lora 融合
    model = get_peft_model(atom_model, lora_config)  # 将基座模型转化为LoRA模型
    print("加载 LoRA 参数成功")
else:  # 从断点开始开始训练
    if pretrain_model_path is None:
        print("断点训练需要给出 checkpoint 地址")
        raise ValueError("断点训练需要给出 checkpoint 地址")
    # 断点继续训练则直接加载断点的 Lora 参数
    model = PeftModel.from_pretrained(atom_model, pretrain_model_path, is_trainable=True)
    print("从{}加载断点成功".format(pretrain_model_path))

# 加载自定义 trainer
args = TrainingArguments(
    save_strategy="steps",
    max_steps=12000,
    save_steps=200,
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16=True,
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    remove_unused_columns=False,
)

trainer = ModifiedTrainer(
    model=model,  # 待训练模型
    train_dataset=dataset,  # 数据集
    args=args,  # 训练参数
    data_collator=lambda x: data_collator(x, tokenizer),
)
#
print("成功加载 Trainer")
# 进行训练
trainer.train()
# 保存模型
model.save_pretrained(save_model_path)
print("模型参数保存在{}".format(save_model_path))

# 开始测试
print("开始测试")
while True:
    # element = input("请输入(输入0退出): ")
    str = input("输入你的问题(输入0退出): ")
    if str == "0":
        break
    list = "<s>Human: " + str + "\n" + "</s><s>Assistant:"
    input_message = tokenizer(list, return_tensors="pt", add_special_tokens=False)  # 对输入进行处理
    generate_input = {
        "input_ids": input_message.input_ids.to('cuda:0'),
        "attention_mask": input_message.attention_mask.to('cuda:0'),
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    generate_ids = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0])  # 将输出转化为文本
    print(text)
    print('\n')
