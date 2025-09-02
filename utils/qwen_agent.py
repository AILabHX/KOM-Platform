# utils/qwen_agent.py
from dashscope import Application
from http import HTTPStatus

def call_qwen_agent(prompt: str, app_id: str, api_key: str) -> str:
    try:
        response = Application.call(
            api_key=api_key,
            app_id=app_id,
            prompt=prompt,
            timeout=60
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.text
        else:
            return f"【Qwen 错误】状态码：{response.status_code}, 消息：{response.message}"
    except Exception as e:
        return f"【调用出错】：{e}"
