from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from rag import create_rag_chain
from pathlib import Path
import socket
# ====================== FastAPI 服务配置 ======================
app = FastAPI(title="RAG 问答系统")

BASE_DIR = Path(__file__).parent
# 挂载静态文件和模板目录
app.mount("/static", StaticFiles(directory=BASE_DIR / "web" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "web" / "templates")

# ✅ 核心修改1：不再使用普通的 global 变量，改用 app.state 来存储 RAG 链

@app.on_event("startup")
async def startup_event():
    """应用启动时自动同步执行初始化"""
    print("🚀 服务器启动中，正在同步初始化 RAG 系统，请耐心等待...")
    try:
        # 将初始化好的 RAG 链存入 FastAPI 官方推荐的全局状态容器中
        app.state.rag_chain = create_rag_chain()
        print("✅ RAG 系统初始化完成！服务已就绪，可以开始提问。")
    except Exception as e:
        print(f"❌ RAG 初始化发生严重错误: {e}")
        import traceback
        traceback.print_exc()

# API 接口：接收问题并返回回答
@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    rag_chain = getattr(app.state, "rag_chain", None)

    if rag_chain is None:
        # 当前进程首次请求时初始化
        print("🔄 当前进程首次请求，初始化 RAG 系统...")
        try:
            app.state.rag_chain = create_rag_chain()
            rag_chain = app.state.rag_chain
        except Exception as e:
            return {"error": f"RAG 初始化失败: {str(e)}"}

    try:
        print("question: ", question)
        response = rag_chain.invoke(question)
        answer_str = response if isinstance(response, str) else str(response)
        print("answer_str: ", answer_str)
        return {"answer": answer_str}
    except Exception as e:
        return {"error": str(e)}

# 测试用的 GET 接口
@app.get("/test")
async def test_api():
    print("✅ 前端成功访问了 /test 接口！")
    return {"status": "success", "message": "后端通信正常"}

# 根路径：返回前端对话框页面
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

# ====================== 启动服务 ======================
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=6525)