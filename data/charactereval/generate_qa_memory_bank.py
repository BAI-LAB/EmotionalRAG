import json
import os
import re
import sys
from collections import defaultdict

from FlagEmbedding import FlagModel
from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.config import BGEModel_PATH
from utils.functions import call_llm

system_prompt = """
你是一个情感分析大师，你能够仔细分辨每段对话角色的情绪状态。
假设每个角色共有8种基本情绪，包括 joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation 。
接下来我会输入一段{role}的对话，你的任务是分析{role}在这8个情绪维度上的得分，最低为1分，最高为10分，得分越高表明{role}在这个情绪维度上表达越强烈。
请分析{role}在8个情绪维度上的表现，给出打分理由和得分，最后以 python list 的形式输出结果，如下所示：
[
    {{"analysis": <REASON>, "dim": "joy", "score": <SCORE>}},
    {{"analysis": <REASON>, "dim": "acceptance", "score": <SCORE>}},
    ...
    {{"analysis": <REASON>, "dim": "anticipation", "score": <SCORE>}}
]
你的回答必须是一个有效的 python 列表以保证我能够直接使用 python 解析它，不要有多余的内容！请给出尽可能准确的、符合大多数人直觉的结果。
"""

embedding_model = FlagModel(
    BGEModel_PATH,
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True,
)


def str2vector(text):
    match = re.findall(r"【(.*?)】", text)
    emotion_str = match[-1]

    emotion_str = emotion_str.replace(": ", "：")
    emotion_str = emotion_str.replace(":", "：")
    emotion_str = emotion_str.replace(", ", "，")
    emotion_str = emotion_str.replace(",", "，")

    emotion_list = emotion_str.split("，")
    emotion_embedding = [float(e.split("：")[1]) for e in emotion_list]
    return emotion_embedding


def get_result(data):
    context_embedding = embedding_model.encode(data["context"]).tolist()

    messages = [
        {"role": "system", "content": system_prompt.format(role=data["role"])},
        {"role": "user", "content": data["context"]},
    ]
    response, _ = call_llm("gpt-3.5-turbo", messages)  # analyze sentiment using entire conversation history
    try:
        emotion_list = json.loads(response)
    except:
        print('gpt reply format error!')
        return None
    emotions = [
        "joy",
        "acceptance",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    ]
    emotion_embedding = [1, 1, 1, 1, 1, 1, 1, 1]
    for item in emotion_list:
        if item["dim"] in emotions:
            emotion_embedding[emotions.index(item["dim"])] = item["score"]

    result = {
        "id": data["id"],
        "context": data["context"],
        "gpt_output": "response",
        "emotion_embedding": emotion_embedding,
        "context_embedding": context_embedding,
    }

    return result


with open("data/CharacterEval/test_data.jsonl", "r", encoding="utf-8") as f:
    datas = json.load(f)

# Extract QA pairs
res_dict = defaultdict(list)
for data in datas:
    role = data["role"]
    dialogue_list = data["context"].split("\n")
    for i in range(1, len(dialogue_list)):
        speaker = dialogue_list[i].split("：")[0]
        if speaker == role:
            qa_pair = dialogue_list[i - 1] + "\n" + dialogue_list[i]
            memory_fragment = {"id": data["id"], "role": role, "context": qa_pair}
            res_dict[role].append(memory_fragment)

all_memeory_bank = {}
for role in res_dict:
    role_memory_bank = [get_result(data) for data in tqdm(res_dict[role])]
    all_memeory_bank[role] = role_memory_bank

with open("data/CharacterEval/all_memory_bank.jsonl", "w", encoding="utf-8") as f:
    json.dump(all_memeory_bank, f, ensure_ascii=False, indent=2)
