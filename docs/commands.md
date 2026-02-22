
# 常用命令

## 研究助手

```bash
# 端到端运行（Planner → Executor → Writer）
.venv/bin/python research_agent/run_example.py

# 单独测试 Serper API key 是否可用
.venv/bin/python -c "
import requests
resp = requests.post('https://google.serper.dev/search',
    headers={'X-API-KEY': 'YOUR_KEY', 'Content-Type': 'application/json'},
    json={'q': 'test', 'num': 3}, timeout=10)
print(f'Status: {resp.status_code}')
print(resp.text[:200])
"

# 验证 scrl 包可正常导入
.venv/bin/python -c "from scrl.handler.handler import Handler; print('OK')"

# 验证 research_agent 可正常导入
.venv/bin/python -c "from research_agent.graph import create_research_graph; print('OK')"
```

## 训练流程（需要 GPU 环境）

```bash
# 启动搜索服务
python -m scrl.handler.server_handler

# 启动训练协调器
python -m scrl.handler.handler

# 训练
export PET_NODE_RANK=0
export VLLM_ATTENTION_BACKEND=XFORMERS
ray start --head
bash train_grpo.sh

# 评估
bash evaluate.sh
python ./evaluate/cacluate_metrics.py {experiment_name}
```

## 开发

```bash
# 安装依赖
.venv/bin/python -m pip install -r requirements.txt

# 代码格式化
bash scripts/format.sh

# 测试（需要 Ray + GPU）
pip install -e ".[test]"
pytest tests/
```
