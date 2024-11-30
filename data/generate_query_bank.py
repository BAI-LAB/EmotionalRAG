import json
import os
import re
import sys

from FlagEmbedding import FlagModel
from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.config import BGEModel_PATH
from utils.functions import call_llm

system_prompt = """
你是一个情感分析大师，你能够仔细分辨采访者问题中蕴含的情绪状态。
假设共有8种基本情绪，包括 joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation 。
接下来我会输入一个采访者的问题，你的任务是分析该问题在这8个情绪维度上的得分，最低为1分，最高为10分，得分越高表明该问题在这个情绪维度上表达越强烈。
请分析该问题在8个情绪维度上的表现，给出打分理由和得分，最后以 python list 的形式输出结果，如下所示：
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
        {"role": "system", "content": system_prompt},
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

    data["emotion_embedding"] = emotion_embedding
    data["context_embedding"] = context_embedding

    return data


with open("data//16Personalities.json", "r", encoding="utf-8") as f:
    datas = json.load(f)

query_bank_16P_zh = []
query_bank_16P_en = []
for id, question in tqdm(datas["questions"].items()):
    query_zh = {"id": id, "context": question["rewritten_zh"]}
    response_zh = get_result(query_zh)
    if response_zh:
        query_bank_16P_zh.append(response_zh)
    query_en = {"id": id, "context": question["rewritten_en"]}
    response_en = get_result(query_en)
    if response_en:
        query_bank_16P_en.append(response_en)

with open("data/query_bank_16P_zh.jsonl", "w", encoding="utf-8") as f:
    json.dump(query_bank_16P_zh, f, ensure_ascii=False, indent=2)

with open("data/query_bank_16P_en.jsonl", "w", encoding="utf-8") as f:
    json.dump(query_bank_16P_en, f, ensure_ascii=False, indent=2)
