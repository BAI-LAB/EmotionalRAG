import torch
from openai import OpenAI
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from utils.config import (OPENAI_API_BASE, OPENAI_API_KEY, ChatGLM_PATH,
                          Qwen_PATH)


def load_model_and_tokenizer(model_name, device):
    if model_name == "qwen":
        model_path = Qwen_PATH
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
    elif model_name == "chatglm":
        model_path = ChatGLM_PATH
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half()
        model = model.eval()
    else:
        return None, None
    model.to(device)
    return model, tokenizer


def call_llm(model, messages):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )

    completion = client.chat.completions.create(model=model, messages=messages)

    content = completion.choices[0].message.content
    usage = completion.usage
    return content, usage


def concat_messages(conversations, role, system):
    history = []
    first_query = system
    if conversations[0]["from"] == role:
        first_response = (
            f"好的！现在我来扮演{role}。" + "我首先发话：" + conversations[0]["value"]
        )
    else:
        first_response = f"好的！现在我来扮演{role}。"

    history.append({"role": "user", "content": first_query})
    history.append({"role": "assistant", "content": first_response})

    for i in range(len(conversations)):
        if conversations[i]["from"] == role:
            if i == 0:
                continue
            else:
                assert conversations[i - 1]["from"] != role
                query = (
                    f"{conversations[i-1]['from']}：" + conversations[i - 1]["value"]
                )
                response = f"{conversations[i]['from']}：" + conversations[i]["value"]
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
    assert conversations[-1]["from"] != role

    query = f"{conversations[-1]['from']}：" + conversations[-1]["value"]
    return history, query


def make_inputs(context):
    dialogues = context.split("\n")
    inputs = []
    for dial in dialogues:
        role = dial.split("：")[0]
        dial = "：".join(dial.split("：")[1:])
        inputs.append({"from": role, "value": dial})
    return inputs


# def call_gpt(role, role_system, context):
#     messages = [{"role": "system", "content": role_system}]

#     conversations = make_inputs(context)
#     for conv in conversations:
#         if conv["from"] == role:
#             messages.append({
#                 "role": "assistant",
#                 "content": f"{conv['from']}：" + conv["value"]
#             })
#         else:
#             messages.append({
#                 "role": "user",
#                 "content": f"{conv['from']}：" + conv["value"]
#             })

#     response, _ = call_llm("gpt-3.5-turbo", messages)
#     return response


def call_gpt2(role_system):
    messages = [{"role": "system", "content": role_system}]
    response, _ = call_llm("gpt-3.5-turbo", messages)
    return response


# def call_qwen(role, role_system, context, model, tokenizer, device):
#     messages = [
#         {"role": "system", "content": role_system},
#         {"role": "user", "content": context}
#     ]

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     torch.cuda.empty_cache()
#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     response = response.split("\n")[0]
#     return response


def call_qwen2(role_system, model, tokenizer, device):
    messages = [{"role": "system", "content": role_system}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    torch.cuda.empty_cache()
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.split("\n")[0]
    return response


# def call_chatglm(role, role_system, context, model, tokenizer):
#     messages, query = concat_messages(make_inputs(context), role, role_system)
#     response, _ = model.chat(tokenizer, query, messages)
#     response = response.split("\n")[0]
#     return response


def call_chatglm2(role_system, model, tokenizer):
    response, _ = model.chat(tokenizer, role_system)
    response = response.split("\n")[0]
    return response
