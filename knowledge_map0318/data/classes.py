"""
大模型加载类
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
from vllm import LLM

# 获取当前文件的绝对路径
project_root_path = os.path.dirname(os.path.abspath(__file__))
# 大模型存储地址
supported_llms_path = os.path.join(project_root_path, "supported_llms")

# Qwen2.5-32B-Int8模型的路径
qwen25_32b_int8_instruct_path = os.path.join(supported_llms_path, "qwen/Qwen2___5-32B-Instruct-GPTQ-Int8")

# Qwen2.5-32B-Int8模型的路径
qwen25_32b_instruct_path = "/home/data1/algorithm/data/models_llm/Qwen/Qwen2___5-32B-Instruct/"

# Qwen2-7B-Instruct模型的路径
qwen2_7b_instruct_path = os.path.join(supported_llms_path, "qwen/qwen2-7b-instruct")
# ChatGLM2-6B模型的路径
chatglm2_6b_path = os.path.join(supported_llms_path, "chatglm2-6b")
# YAYI-UIE模型的路径
yayi_uie_path = os.path.join(supported_llms_path, "yayi-uie")
# GLM-4-9B-Chat模型的路径
glm_4_9b_chat_path = os.path.join(supported_llms_path, "glm-4-9b-chat")

class ModelLoader:
    def __init__(self):
        self.supported_llms = {
            0: {
                "model_id": "0",
                "current_model_name": "qwen25_32b_int8_instruct",
                "path": f"{qwen25_32b_int8_instruct_path}",
                "load_function": self.load_qwen_instruct
            },
            1: {
                "model_id": "1",
                "current_model_name": "qwen2-7b-instruct", 
                "path": f"{qwen2_7b_instruct_path}",
                "load_function": self.load_qwen_instruct
            },
            2: {
                "model_id": "3",
                "current_model_name": "yayi-uie",
                "path": f"{yayi_uie_path}",
                "load_function": self.load_yayi_uie
            },
            3: {
                "model_id": "4",
                "current_model_name": "glm-4-9b-chat",
                "path": f"{glm_4_9b_chat_path}",
                "load_function": self.load_glm_4_9b_chat
            },
            4: {
                "model_id": "5",
                "current_model_name": "qwen25-32b-instruct",
                "path": f"{qwen25_32b_instruct_path}",
                "load_function": self.load_qwen25_32b_instruct
            }
        }
        # 当前加载的模型（初始化为空）
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_id = None  # 用于跟踪当前加载的模型 ID

    def unload_model(self):
        # 如果当前模型已加载，则卸载当前模型
        if self.current_model:
            print(f"Unloading current model {self.current_model_id}...")
            del self.current_model
            del self.current_tokenizer
            torch.cuda.empty_cache()  # 清理显存
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_id = None  # 重置当前模型 ID
            print("Model unloaded successfully.")
        else:
            print("No model is currently loaded.")

    def load_model(self, model_id = 0):
        # 判断模型 ID 是否存在
        model_info = self.supported_llms.get(model_id)
        if not model_info:
            print(f"Model with ID {model_id} not found.")
            return None

        # 检查新模型与当前模型是否相同
        if self.current_model_id == model_id:
            print(f"Model {model_id} is already loaded. No need to unload.")
            return self.current_tokenizer, self.current_model

        # 如果新模型不同于当前加载的模型，则先卸载当前模型
        self.unload_model()

        # 获取模型路径和加载函数
        model_path = model_info["path"]
        load_function = model_info["load_function"]
        
        # 加载新模型
        print(f"Loading new model {model_path}...")
        return load_function(model_path)

    def load_qwen_instruct(self, model_path):
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.current_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto").half()
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # "auto"
            device_map="auto"
        )
        # # TODO: 卡数量 tensor_parallel_size ，可配置
        # self.current_model = LLM(model=model_path, tensor_parallel_size=2, quantization="gptq", gpu_memory_utilization=0.95, max_model_len=10240, dtype="half")
        # 大模型开启评估模式，表示不会进行训练，而是用于推理，这样可以提高推理速度，减少显存和内存的占用
        # self.current_model.eval()
        return self.current_tokenizer, self.current_model
    
    def load_qwen25_32b_instruct(self, model_path):
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.current_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto").half()
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # "torch.float16"
            device_map="auto"
        )
        # # TODO: 卡数量 tensor_parallel_size ，可配置
        # self.current_model = LLM(model=model_path, tensor_parallel_size=2, quantization="gptq", gpu_memory_utilization=0.95, max_model_len=10240, dtype="half")
        # 大模型开启评估模式，表示不会进行训练，而是用于推理，这样可以提高推理速度，减少显存和内存的占用
        # self.current_model.eval()
        return self.current_tokenizer, self.current_model
    
    def load_yayi_uie(self, model_path):
        self.current_tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True)
        self.current_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                            trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(model_path)
        # 评估模式
        self.current_model.eval()
        return self.current_tokenizer, self.current_model


    def load_glm_4_9b_chat(self, model_path):
        self.current_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.current_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                            device_map="auto",
                                                            trust_remote_code=True)
        # 评估模式
        self.current_model.eval()
        return self.current_tokenizer, self.current_model

# 自定义异常类
class InferenceError(Exception):
    pass




