"""
大模型调用
"""
import json
import re
import os
from openai import OpenAI
from values.global_config_variable import LLM_MODE

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

# 拆分模型输出为思考部分和结果部分
def extract_text_after_think(input_text):
    think_match = re.search(r"<think>(.*?)</think>", input_text, re.DOTALL)
    result_match = re.search(r"</think>\s*(.*)", input_text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        result_content = result_match.group(1).strip()
        return think_content, result_content
    else:
        return None, input_text

# 调用本地大模型进行推理----Qwen3
# messages: [{"role": "user", "content": "你好啊"}](对话列表)
# tokenizer: API调用下是url；本地模型直接调用是分词器
# model: API调用下是模型名称；本地模型直接调用是加载的模型
# enable_thinking_flag: 是否开启思考模式（默认不思考）
# max_tokens: 最大输出长度
# 用例：
# from values.global_config_variable import LLM_model_name,base_url
# messages = [{"role": "user", "content": "你好啊"}]
# _, result = chat_local_llm(messages, base_url, LLM_model_name)
# print(result)

def chat_local_llm(messages, tokenizer, model, enable_thinking_flag=False, max_tokens=20000):
    try:
        if LLM_MODE == 0:  # 本地用
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking_flag
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
            if enable_thinking_flag:                 
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "do_sample": True,
                    "top_k": 20,
                }
            else:
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "do_sample": True,
                    "top_k": 20,
                } 

            generated_ids = model.generate(
                **model_inputs,
                **generation_kwargs
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
        elif LLM_MODE == 1:  # API调用
            client = OpenAI(
                api_key=model,
                base_url=tokenizer,
                timeout=60.0
            )
            if enable_thinking_flag:
                msg = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    extra_body={
                        "top_k": 20,
                    }
                )
            else:
                msg = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,  # 降低temperature以获得更稳定的输出
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20, 
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            # print(msg)
            response = msg.choices[0].message.content
            print(response)
            thinking_content, content = extract_text_after_think(response)
        
        elif LLM_MODE == 2:  # vllm调用
            if enable_thinking_flag:
                sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens)
            else:
                sampling_params = SamplingParams(temperature=0.3, top_p=0.8, top_k=20, max_tokens=max_tokens)

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking_flag
            )
            outputs = model.generate([text], sampling_params)

            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            thinking_content, content = extract_text_after_think(generated_text)
        else:
            print("大模型调用模式错误！！！请重新检查！！！")
    except RuntimeError as e:
        raise e
    return thinking_content, content

def chat_local_llm_stream(messages, tokenizer, model, max_tokens=20000):
    try:
        if LLM_MODE == 1:
            client = OpenAI(
                api_key=model,
                base_url=tokenizer,
                timeout=600.0
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.8,
                stream=True, 
                extra_body={
                    "top_k": 20, 
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
            
            for chunk in response:
                # 检查 delta 是否存在且有 content
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
    except Exception as e:
        # 这里不要打印，直接抛出让外层 API 捕获
        raise e
    
# 输出内容格式化为json
def match_pattern(raw_json):
    try:
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, raw_json)

        for match in matches:
            try:
                event = json.loads(match)
                return event
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Error during pattern matching: {e}")
        return []
    
if __name__ == '__main__':
    from values.global_config_variable import LLM_model_name,base_url
    messages = [{"role": "user", "content": "你好啊"}]
    thinking_content, content = chat_local_llm(messages, base_url, LLM_model_name)
    print(thinking_content, content)