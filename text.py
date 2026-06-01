python -c "
import os, sys
sys.path.insert(0, '.')
from app.config import settings

# 打印 Key 的精确信息（只显示前后各4个字符）
key = settings.dashscope_api_key
print(f'Key 长度: {len(key)}')
print(f'Key 前缀: {key[:7]}...')
print(f'Key 后缀: ...{key[-4:]}')
print(f'Key 是否以 sk- 开头: {key.startswith(\"sk-\")}')
print(f'Key 包含空白字符: {repr(key[:20])}')

print()
print('直接调用 DashScope Embedding API...')
from openai import OpenAI
client = OpenAI(api_key=key, base_url=settings.dashscope_base_url)
try:
    resp = client.embeddings.create(input='测试', model='text-embedding-v2')
    print(f'成功! 向量维度: {len(resp.data[0].embedding)}')
except Exception as e:
    print(f'失败: {e}')
"