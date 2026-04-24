from openai import OpenAI
import os
from judge_agent.llm_core.api_keys import OPENAI_API_KEY
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 
# 确保你的环境变量里已经配置了 OPENAI_API_KEY
# 如果没有，可以在这里直接传入: client = OpenAI(api_key="sk-你的真实密钥")
client = OpenAI()

try:
    # 获取当前 API Key 可用的所有模型列表
    models = client.models.list()
    
    print("=== 当前 API Key 可用的模型列表 ===")
    
    # 提取模型名称并按字母排序，方便查看
    available_models = sorted([model.id for model in models.data])
    
    for model_id in available_models:
        # 你可以过滤一下，只看 gpt 相关的模型
        if "gpt" in model_id:
            print(f"- {model_id}")
            
except Exception as e:
    print(f"查询失败: {e}")