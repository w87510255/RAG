# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools  import tool


deepseek_llm = ChatOllama(model="qwen3.5:4b", base_url="http://localhost:11434",temperature=0)

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


def Weather(location: str) -> str:
    """
    获取天气，参数是位置
    """
    return f"今天多云,空气中很多杨絮 {location}"

def Location(location: str) -> str:
    """
    获取当前是什么城市
    """
    return f"this is beijing {location}"

agent = create_agent(model=deepseek_llm, tools=[Weather,Location])

response = agent.invoke({ "messages": [{"role":"user", "content":"今天北京天气怎么样"}]})

for res in response["messages"]:
    print(res.model_dump_json(indent=2))

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
