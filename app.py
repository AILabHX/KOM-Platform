import streamlit as st
from PIL import Image
import base64
import time
import streamlit.components.v1 as components
import json
from typing import Dict, List, Optional
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import re
import streamlit_image_select as sis
from fpdf import FPDF
import io
import markdown
from utils.qwen_agent import call_qwen_agent
import os
import math
import html


# =============================================================================
# 配置和常量
# =============================================================================
PAGE_CONFIG = {
    "page_title": "Knee OA Demo",
    "layout": "wide"
}

CHAT_CONFIG = {
    "update_interval": 2,
    "height": 400,
    "max_width": "80%"
}

IMAGE_PATHS = {
    "logo": "images/logo.png",
    "framework": "images/framework.png",
    "status_framework": "images/status_framework.png",
    "predicting_framework": "images/predicting_framework.png",
    "recommendation_framework": "images/Recommendation_framework.png"
}

CASES_FILE = "cases.json"
PARAMS_FILE = "predict_params.json"
PREDICT_FILE = "predict_params_ori.json"


# =============================================================================
# 工具函数
# =============================================================================
@st.cache_data
def get_base64_image(image_path: str) -> Optional[str]:
    """安全获取图片base64编码"""
    try:
        if Path(image_path).exists():
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"无法加载图片 {image_path}: {e}")
    return None

@st.cache_data
def load_initial_chat_history(file_path: str = "initial_chat.json") -> List[Dict]:
    try:
        if Path(file_path).exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            st.warning(f"找不到初始对话文件：{file_path}")
    except Exception as e:
        st.error(f"加载初始聊天对话失败: {e}")
    return []

@st.cache_data
def load_analysis_report(json_file="assess_result.json") -> Dict:
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Unable to load the analysis report file: {e}")
        return {}


@st.cache_data
def load_case_data() -> Dict:
    try:
        if Path(CASES_FILE).exists():
            with open(CASES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"无法加载病例数据: {e}")
    return {}


def load_plan(agent_type: str):
    path = f"{agent_type}_plan.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text_for_pdf(text: str) -> str:
    replacements = {
        "–": "-",  
        "—": "-",  
        "“": "\"",  
        "”": "\"",
        "’": "'",
        "•": "-", 
        "→": "->",
        "…": "...",
        "©": "(c)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")


def strip_non_latin1(text: str) -> str:
    """去除无法被 Latin-1 编码的字符（如 emoji、中文）"""
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def generate_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.split("\n"):
        clean_line = strip_non_latin1(line)
        pdf.multi_cell(0, 10, txt=clean_line)

    return pdf.output(dest="S").encode("latin1")

def safe_image_display(image_path: str, caption: str = "", **kwargs):
    """显示图片"""
    try:
        if Path(image_path).exists():
            st.image(image_path, caption=caption, **kwargs)
        else:
            st.warning(f"⚠️ 图片未找到: {image_path}")
    except Exception as e:
        st.error(f"显示图片时出错: {e}")

def generate_report_text_from_prediction(params: dict) -> str:
    lines = []
    
    # 1. Symptom Trajectory Forecast
    lines.append("📊 Symptom Trajectory Forecast (KOOS, 0–100)")
    symptom_rows = [
        ("Right Knee Pain", "symptom_trajectory.right_knee.pain"),
        ("Right Knee Symptoms", "symptom_trajectory.right_knee.symptoms"),
        ("Left Knee Pain", "symptom_trajectory.left_knee.pain"),
        ("Left Knee Symptoms", "symptom_trajectory.left_knee.symptoms"),
        ("Sport/Recreation Function", "symptom_trajectory.right_knee.sport_recreation_function"),
        ("Quality of Life", "symptom_trajectory.right_knee.quality_of_life")
    ]
    for label, base_key in symptom_rows:
        v00 = params.get(f"{base_key}.v00", "N/A")
        v01 = params.get(f"{base_key}.v01", "N/A")
        v04 = params.get(f"{base_key}.v04", "N/A")
        lines.append(f"- {label}:  Current={v00}, Year 2={v01}, Year 4={v04}")
    lines.append("")

    # 2. Imaging Trajectory Forecast
    lines.append("🦴 Imaging Trajectory (KL grade, 0–4)")
    for side in ["right", "left"]:
        v00 = params.get(f"imaging_trajectory.{side}_knee.pain.v00", "N/A")
        v01 = params.get(f"imaging_trajectory.{side}_knee.pain.v01", "N/A")
        v04 = params.get(f"imaging_trajectory.{side}_knee.pain.v04", "N/A")
        lines.append(f"- {side.capitalize()} Knee:  Current={v00}, Year 2={v01}, Year 4={v04}")
    lines.append("")

    # 3. SHAP Key Factors
    lines.append("💡 Key Contributing Factors (SHAP)")
    shap_data = params.get("key_factors.right_knee_symptoms_year2", [])
    for item in shap_data:
        feature = item.get("feature", "Unknown")
        impact = item.get("impact", "N/A")
        effect = item.get("effect", "")
        lines.append(f"- {feature}: {impact} ({effect})")

    return "\n".join(lines)


# =============================================================================
# 样式定义
# =============================================================================
def get_navigation_styles(logo_base64: str) -> str:
    """获取导航栏样式"""
    return f"""
    <style>
    .nav-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #ddd;
        flex-wrap: wrap;
        margin-bottom: 0;
    }}
    .left-section {{
        display: flex;
        align-items: center;
        gap: 20px;
        flex: 1;
        min-width: 300px;
    }}
    .app-title {{
        color: #4B6EAF;
        font-family: "Segoe UI", sans-serif;
    }}
    .app-title div:first-child {{
        font-size: 20px;
        font-weight: bold;
    }}
    .app-title div:last-child {{
        font-size: 18px;
    }}
    .logo-img {{
        height: 70px;
    }}
    .nav-buttons {{
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        justify-content: flex-end;
    }}
    .nav-buttons form {{ margin: 0; }}
    .agent-button {{
        font-size: 16px;
        padding: 10px 16px;
        display: inline-flex;
        align-items: center;
        background-color: #f0f0f0;
        border: 2px solid #ccc;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s ease;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }}
    .agent-button:hover {{ background-color: #e0e0e0; }}
    .agent-button img {{
        width: 24px;
        height: 24px;
        margin-right: 8px;
    }}
    .arrow-icon {{
        font-size: 20px;
        color: #888;
    }}
    </style>

    <div class="nav-container">
        <div class="left-section">
            <img src="data:image/png;base64,{logo_base64}" class="logo-img" />
            <div class="app-title">
                <div>Knee Osteoarthritis Management Platform</div>
                <div>膝骨关节炎人工智能平台</div>
            </div>
        </div>
        <div class="nav-buttons">
            <form action="/" method="get" title="Home Page">
                <input type="hidden" name="page" value="Home">
                <button type="submit" class="agent-button" style="font-weight: 600;">Home</button>
            </form>
            <form action="/" method="get" title="Assess status">
                <input type="hidden" name="page" value="Assessing Current Status">
                <button type="submit" class="agent-button">
                    <img src="https://www.svgrepo.com/download/285252/robot.svg" alt="robot icon">
                    评估(Assessment Agent)
                </button>
            </form>
            <div class="arrow-icon">➡️</div>
            <form action="/" method="get" title="Predict risk">
                <input type="hidden" name="page" value="Predicting Progression Risk">
                <button type="submit" class="agent-button">
                    <img src="https://www.svgrepo.com/download/285252/robot.svg" alt="robot icon">
                    预测(Risk Agent)
                </button>
            </form>
            <div class="arrow-icon">➡️</div>
            <form action="/" method="get" title="Recommend therapy">
                <input type="hidden" name="page" value="Tailored Therapy Recommendation">
                <button type="submit" class="agent-button">
                    <img src="https://www.svgrepo.com/download/285252/robot.svg" alt="robot icon">
                    处方(Therapy Agent)
                </button>
            </form>
        </div>
    </div>
    """


def get_chat_styles() -> str:
    """返回优化后的聊天气泡样式"""
    return """
    <style>
    .chat-bubble {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        border-left: 5px solid transparent;
        background-color: #f9f9f9;
    }

    .chat-content {
        white-space: normal !important;
        word-break: break-word;
        overflow-wrap: break-word;
        line-height: 1.6;
    }

    .exercise {
        background-color: #f0fff4;
        border-left-color: #34c759;
    }

    .pharma {
        background-color: #f0f8ff;
        border-left-color: #1e90ff;
    }

    .nutrition {
        background-color: #fffaf0;
        border-left-color: #f4b400;
    }

    .summary {
        background-color: #fff0f0;
        border-left-color: #ff6b6b;
    }

    .chat-icon {
        font-weight: bold;
        margin-bottom: 6px;
        display: block;
        color: #333;
    }

    .chat-bubble strong {
        display: inline-block;
        margin-bottom: 4px;
        color: #222;
    }
    </style>
    """



# =============================================================================
# 聊天功能
# =============================================================================
class ChatManager:
    def __init__(self, initial_chat_file: str = "assess_chat.json"):
        self.initial_history = load_initial_chat_history(initial_chat_file)

    def initialize_state(self):
        """初始化聊天状态"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = self.initial_history.copy()
            st.session_state.chat_step = 1
            st.session_state.last_update_time = time.time()



    def update_progress(self):
        """更新聊天进度（非阻塞）"""
        current_time = time.time()
        if (st.session_state.chat_step < len(st.session_state.chat_history) and
                current_time - st.session_state.last_update_time > CHAT_CONFIG["update_interval"]):
            st.session_state.chat_step += 1
            st.session_state.last_update_time = current_time
    # def update_progress(self):
    #     current_time = time.time()
    #     step = st.session_state.chat_step
    #     history = st.session_state.chat_history
    
    #     # 如果未显示完全部对话，并且刷新间隔已达
    #     if step < len(self.initial_history) and current_time - st.session_state.last_update_time > CHAT_CONFIG["update_interval"]:
    
    #         next_msg = self.initial_history[step]
    
    #         # 👤 如果下一条是用户内容，直接添加
    #         if next_msg["role"] == "user":
    #             history.append(next_msg)
    
    #         # 🤖 如果下一条是 AI 回答：动态生成（用上一条 user 消息作为 prompt）
    #         elif next_msg["role"] == "assistant":
    #             # ⛔ 安全校验：若 history 为空或上一条不是 user，跳过
    #             if len(history) == 0 or history[-1]["role"] != "user":
    #                 st.warning("⚠️ 无法生成 AI 回答：找不到上一条用户消息")
    #                 return
    
    #             user_prompt = history[-1]["content"]
    
    #             app_id = "c968f91131ac432787f5ef81f51922ba"
    #             api_key = os.getenv("DASHSCOPE_API_KEY")
    #             ai_reply = self.generate_response(user_prompt, app_id, api_key)
    
    #             history.append({"role": "assistant", "content": ai_reply})
    
    #         # ✅ 每推进一条，step +1，更新时间
    #         st.session_state.chat_step += 1
    #         st.session_state.last_update_time = current_time


   
    def render_message(self, role: str, content: str) -> str:
        """渲染单条消息（AI左侧，用户右侧）"""
        if role == "user":
            return f"""
                <div style="display: flex; justify-content: flex-end; margin: 5px 0;">
                    <div style="background-color: #DCF8C6; color: black;
                                padding: 8px 12px; border-radius: 12px; max-width: {CHAT_CONFIG['max_width']};
                                text-align: left;">
                        🧍 {content}
                    </div>
                </div>
            """
        else:
            return f"""
                <div style="display: flex; justify-content: flex-start; margin: 5px 0;">
                    <div style="background-color: #F1F0F0; color: black;
                                padding: 8px 12px; border-radius: 12px; max-width: {CHAT_CONFIG['max_width']};
                                text-align: left;">
                        👨‍⚕️ {content}
                    </div>
                </div>
            """

    def render_chat_interface(self):
        for msg in st.session_state.chat_history[:st.session_state.chat_step]:
            print("原始 content 内容：", repr(msg["content"]))

        # # def process_content(content):
        # #     # 将换行符转换为 HTML 换行标签
        # #     return content.replace("\n", "<br>")
        # def process_content(content):
        #     print("替换前：", content[:50])  # 打印前50字符
        #     content = content.replace("\n", "<br>")
        #     content = content.replace("   ", "&nbsp;&nbsp;&nbsp;")
        #     print("替换后：", content[:50])  # 打印替换后的前50字符
        #     return content
        
        # chat_html = "".join([
        #     self.render_message(msg["role"], process_content(msg["content"]))  # 应用换行处理
        #     for msg in st.session_state.chat_history[:st.session_state.chat_step]
        # ])
        chat_html = "".join([
            self.render_message(
                msg["role"],
                # 关键修改：先处理字面意义的 \\n（\和n组成的字符），再转<br>
                msg["content"]
                    .replace("\\n", "\n")  # 第一步：将字面的 \n 转为真正的换行符
                    .replace("\n", "<br>")  # 第二步：将真正的换行符转为 HTML 换行
                    .replace("   ", "&nbsp;&nbsp;&nbsp;")  # 保留缩进
            )
            for msg in st.session_state.chat_history[:st.session_state.chat_step]
        ])
    
        height = CHAT_CONFIG["height"]
        components.html(f"""
            <div style="height: {height}px; overflow-y: auto; padding: 10px 10px 40px 10px; border: 1px solid #ccc; 
                        border-radius: 8px; background-color: white;box-sizing: border-box;" id="chat-box">
                {chat_html}
                <div id="bottom"></div>
            </div>
            <script>
                const chatBox = document.getElementById("chat-box");
                chatBox.scrollTop = chatBox.scrollHeight;
                setTimeout(() => {{
                    window.parent.postMessage({{ isStreamlitMessage: true, type: 'streamlit:rerun' }}, '*');
                }}, {int(CHAT_CONFIG['update_interval'] * 1000)});
            </script>
        """, height=height)


  
    def handle_user_input(self):
        user_input = st.chat_input("Please enter your symptoms, medical history or problems...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})            
            app_id = "c968f91131ac432787f5ef81f51922ba"
            api_key = os.getenv("DASHSCOPE_API_KEY")
            response = self.generate_response(user_input, app_id, api_key)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
            st.session_state.chat_step = len(st.session_state.chat_history)
            st.rerun()


    
    # def generate_response(self, user_input: str) -> str:
    #     """生成助手回复"""
    #     responses = {
    #         "Pain": "Please describe in detail the nature, frequency and triggering factors of the pain.",
    #         "Swelling": "When does swelling usually occur? Is there any accompanying fever?",
    #         "Stiffness": "How long does morning stiffness last? Was there any improvement after the activity?",
    #         "Cracking": "Is joint cracking accompanied by pain?"
    #     }
        
    #     for keyword, response in responses.items():
    #         if keyword in user_input:
    #             return response
        
    #     return "Thank you for your feedback. I will conduct an analysis based on this information. Please continue to describe your symptoms."
    def generate_response(self, user_input: str, app_id: str, api_key: str) -> str:
        if not api_key:
            return "❌ 请在 Hugging Face 的 Secrets 中配置 DASHSCOPE_API_KEY。"
    
        try:
            st.write(f"🚀 正在调用 Qwen，输入：{user_input}")
            response = call_qwen_agent(user_input, app_id, api_key)
            st.write(f"✅ Qwen 返回前200字：{response[:200]}")
            return response
        except Exception as e:
            st.write(f"❌ Qwen 调用失败：{e}")
            return "调用 Qwen API 出错，请稍后再试。"


# =============================================================================
# 页面渲染函数
# =============================================================================
def render_navigation():
    """渲染导航栏"""
    logo_base64 = get_base64_image(IMAGE_PATHS["logo"])
    if logo_base64:
        st.markdown(get_navigation_styles(logo_base64), unsafe_allow_html=True)
    else:
        st.title("Knee Osteoarthritis Management Platform")

def inject_agent_styles():
    """注入各智能体颜色样式"""
    st.markdown("""
    <style>
        .chat-bubble {
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .chat-bubble.exercise {
            background-color: #e6f4ea;
            border-left: 6px solid #34a853;
        }
        .chat-bubble.surgical, .chat-bubble.pharma {
            background-color: #e8f0fe;
            border-left: 6px solid #4285f4;
        }
        .chat-bubble.nutrition, .chat-bubble.psychology {
            background-color: #fff8e1;
            border-left: 6px solid #fbbc04;
        }
        .chat-icon {
            font-weight: bold;
            margin-bottom: 8px;
        }
        .chat-bubble.decision {
            background-color: #f1f3f4;
            border-left: 6px solid #5f6368;
        }
    </style>
    """, unsafe_allow_html=True)


def render_home_page():
    """渲染首页"""
    abstract_col, figure_col = st.columns([0.9, 1.1])
    
    with abstract_col:
        st.markdown('<h4 style="font-size:22px;">About</h4>', unsafe_allow_html=True)
        st.markdown("""
Code and Data are available at: https://github.com/jacobliuweizhi/KOM

Introduction to KOM
Knee Osteoarthritis Management (KOM) system is an intelligent, multi-agent (Multi-Agent) AI system that supports the full KOA care management pathway—assessment → risk prediction → individualized therapy—to enable precise, standardized, and scalable disease management for knee osteoarthritis.

Quick Start
In the top-right corner of the page, you’ll see three buttons, each mapping to a core module. Click from left to right to experience the end-to-end AI-assisted diagnosis and prescription flow.

Modules & Capabilities
1.	Assessment Agent: helps complete missing information, explains medical terms, and guides KOOS collection. Uses AI to analyze X-rays for knee positioning, KOA grading, joint space narrowing, and bone changes. Output: one-click case evaluation report in clinical style.
2.	Risk Agent: Predicts how symptoms and X-ray findings will change over 4 years. Shows which factors (like bone spurs, pain levels, and muscle strength) contribute most to each patient's risk, helping guide treatment decisions.
3.	Treatment Multi-Agent Cluster (Therapy Agent) MDT via multi-agent collaboration: Creates personalized treatment plans through teamwork between specialized agents for Exercise/Rehab, Orthopedics, Psycho-Nutrition, and Clinical Integration. Recommendations are based on extensive medical literature (over 4,000 research entries) and follow established frameworks for exercise and nutrition while prioritizing safety and practical advice.
This website is at an early stage of development and intended for research purposes only. For collaboration or to report bugs, please contact us at lijian_sportsmed@163.com. Thank you!

        """)

    with figure_col:
        st.markdown('<h4 style="font-size:22px;">General Framework</h4>', unsafe_allow_html=True)
        safe_image_display(IMAGE_PATHS["framework"], "Framework Overview", use_container_width=True)

    st.markdown("---")
    st.markdown("⚠️This website is at an early stage of development and intended for research purposes only. Thank you! 本网页仅用于研究用途")
    # 每秒刷新一次（根据需要调整频率），最多刷新 N 次


def generate_report_text_from_json(json_path: str = "structured_report_template.json") -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        report_dict = json.load(f)

    lines = []
    for section, contents in report_dict.items():
        lines.append(section)
        for line in contents:
            lines.append(line)
        lines.append("")  # 每个 section 之间空一行
    return "\n".join(lines)


def render_centered_image_full(image_path, width=300):
    import base64
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        encoded = base64.b64encode(img_bytes).decode("utf-8")
    
    html = f'''
        <div style="width: 100%; text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,{encoded}" width="{width}" />
        </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def spacer(height_px=24):
    st.markdown(f"<div style='height: {height_px}px;'></div>", unsafe_allow_html=True)

def render_assessment_page():
    """渲染评估页面"""
    st.markdown("""
        <style>
        /* 按钮字体 */
        .stButton > button,
        .stDownloadButton > button {
            font-size: 16px !important;
        }
        /* 调整上传图片按钮宽度为100% */

        div[data-testid="stButton-upload_image_btn"] > button {
            width: 100% !important;
            padding: 0.5rem 1rem !important;
            min-width: 100% !important;
            box-sizing: border-box !important;
        }
        
        /* 下拉框输入框和选中项 */
        div[data-baseweb="select"] > div > div > input,
        div[data-baseweb="select"] > div > div > div,
        div[data-baseweb="select"] ul > li {
            font-size: 16px !important;
        }
        /* 标签字体 */
        label, .stTextInput label, .stSelectbox label, .stMultiSelect label {
            font-size: 16px !important;
        }
        .st-emotion-cache-tn0cau {  
            gap: 0 !important;  /* 覆盖1remgap */
            margin-top: 0 !important;  
            padding-top: 0 !important; 
        }
        
        .stColumns {
            gap: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    chat_manager = ChatManager()
    chat_manager.initialize_state()
    chat_manager.update_progress() 
    
    if st.session_state.chat_step < len(st.session_state.chat_history):
        st_autorefresh(interval=1500, key="chat_autorefresh")
    

    for key, default in {
        "show_sidebar": False,
        "selected_image_path": None,
        "selected_image_label": None,
        "typing_index": 0,       
        "chat_history": None,     
        "chat_step": 1,
        "last_update_time": time.time(),
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.show_sidebar:
        with st.sidebar:
            st.markdown("### 📂 Select a sample image")
            PREDEFINED_IMAGES = {
                "Knee Image A": "images/knee_sample_1.png",
            }
            for idx, (label, path) in enumerate(PREDEFINED_IMAGES.items()):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(path, width=150, caption=label)
                    if st.button("✅ Select", key=f"select_{idx}"):
                        st.session_state.selected_image_path = path
                        st.session_state.selected_image_label = label
                        st.session_state.show_sidebar = False
                        st.rerun()
    

    st.markdown('<p class="chat-note">Demo chat interface (display only).</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        chat_manager.render_chat_interface()
        chat_manager.update_progress()
        chat_manager.handle_user_input()

        st.divider()

        if st.button("📷 Upload Image", key="upload_image_btn", use_container_width=True):
            st.session_state.show_sidebar = True

 
        if st.session_state.selected_image_path and st.session_state.selected_image_label:
            st.success(f"✅ Selected: {st.session_state.selected_image_label}")

            render_centered_image_full(st.session_state.selected_image_path, width=450)

            spacer(24)

            analysis_data = load_analysis_report()
            report = analysis_data.get("default")

            if report:
                with st.expander("📝 View Structured Analysis Report"):
                    for knee, sections in report.items():
                        st.markdown(f"### 🦵 {knee}")
                        for section_title, items in sections.items():
                            st.markdown(f"**{section_title}**")
                            for item in items:
                                st.markdown(f"- {item}")
                report_text = generate_report_text_from_json()
                report_text = clean_text_for_pdf(report_text)
                pdf_bytes = generate_pdf(report_text)

                with open("custom_patient_report_ori.json", "r", encoding="utf-8") as f:
                    custom_json_data = json.load(f)
                json_bytes = io.BytesIO(json.dumps(custom_json_data, indent=2).encode('utf-8'))

                # left_col, right_col = st.columns([4, 1])
                # with left_col:
                st.download_button(
                    label="📄 Download Structured Analysis Report as PDF",
                    data=pdf_bytes,
                    file_name="knee_report.pdf",
                    mime="application/pdf"
                )

                # with right_col:
                st.download_button(
                    label="📄 Download the JSON file",
                    data=json_bytes,
                    file_name="knee_report.json",
                    mime="application/json"
                )
            else:
                st.warning("⚠️ The structured analysis report failed to load")

    with col2:
        safe_image_display(IMAGE_PATHS["status_framework"], "Status framework", use_container_width=True)


def render_chat(role, message=None, table_df=None):
    avatar = {
        "User": "🧑",
        "AI": "🩺"
    }.get(role, "💬")

    bg_color = {
        "User": "#f1f8e9",  
        "AI": "#F2F2F2"
    }.get(role, "#eeeeee")

    content_html = ""
    if message:
        content_html += f"<div style='margin-bottom:10px;'>{message}</div>"
    if table_df is not None:
        table_html = table_df.to_html(index=False)
        table_html = f"""
        <style>
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: center !important; }}
            td {{ text-align: center; }}
        </style>
        {table_html}
        """
        content_html += f"<div style='overflow-x:auto'>{table_html}</div>"

    st.markdown(f"""
    <div style="background-color:{bg_color}; padding:14px 18px; border-radius:10px; margin-bottom:18px; display:flex;">
        <div style="font-size:22px; margin-right:12px;">{avatar}</div>
        <div style="font-size:15px; line-height:1.6; width:100%;">{content_html}</div>
    </div>
    """, unsafe_allow_html=True)

    
def render_centered_table(df):
    html_table = df.to_html(index=False)
    centered_html = f"""
    <div style="display: flex; justify-content: center;">
        <div style="width: 80%;">
            {html_table}
        </div>
    </div>
    """
    st.markdown(centered_html, unsafe_allow_html=True)

def render_prediction_report(params):
    st.markdown("<h4>📝 Comprehensive Prediction Report</h4>", unsafe_allow_html=True)

    render_chat("AI", """
    Excellent!  
    I’ve received your case report—thanks for submitting it!
    With this complete dataset, I can now provide you with a comprehensive forecast of how your knee condition may evolve over time. Here’s what the model predicts:
    """)

    st.markdown("<h5>📊 Symptom Trajectory Forecast (KOOS, 0–100)</h5>", unsafe_allow_html=True)
    symptom_table = {
        "Metric": ["Right Knee Pain", "Right Knee Symptoms", "Left Knee Pain", "Left Knee Symptoms","Sport/Recreation Function", "Quality of Life"],
        "Current (V00)": [
            params["KOOSPain_R"],
            params["KOOSSym_R"],
            params["LKPain_V00"],
            params["LKSym_V00"],
            params["KOOSSport"],
            params["KQOL_V00"]
            
        ],
        "Year 2 (V01)": [
            97,
            93,
            89,
            73,
            58,
            31  
        ],
        "Year 4 (V04)": [
            97,
            91,
            84,
            66,
            75,
            50
        ]
    }
    render_chat("AI", "Here is the forecast of your knee-related symptoms over the coming years:", pd.DataFrame(symptom_table))

    # 影像预测（KL）
    st.markdown("<h5>🦴 Imaging Trajectory (KL grade, 0–4)</h5>", unsafe_allow_html=True)
    imaging_table = {
        "Knee": ["Right", "Left"],
        "Current": [
            params["RKImg_V00"],
            params["LKImg_V00"]
        ],
        "Year 2": [
            "Severe",
            "Mild"
        ],
        "Year 4": [
            "Severe",
            "Mild"
        ]
    }
    render_chat("AI", "Here’s how your knee structure may change over time, based on imaging predictions:", pd.DataFrame(imaging_table))

    # SHAP 解释
    st.markdown("<h5>💡 Key Contributing Factors (SHAP)</h5>", unsafe_allow_html=True)
    shap_data = params["key_factors.right_knee_symptoms_year2"]
    shap_table = {
        "Feature": [item["feature"] for item in shap_data],
        "Impact on KOOS Symptoms": [f"{item['impact']} ({item['effect']})" for item in shap_data]
    }
    render_chat("AI", "These are the most impactful factors influencing your right knee symptoms at Year 2:", pd.DataFrame(shap_table))

# def load_default_params():
#     with open(PARAMS_FILE, "r") as f:
#         return json.load(f)

def load_default_params(file_path: str) -> dict:
    
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"参数文件不存在: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"参数文件格式错误（非有效的JSON）: {file_path}")
    except Exception as e:
        raise Exception(f"加载参数文件时出错: {str(e)}")


def multi_column_radio(label, options, cols=6, index=0):

    key = f"multi_col_radio_{label}"
    if key not in st.session_state:
        st.session_state[key] = options[index] if options else None
    
    items_per_col = math.ceil(len(options) / cols) if options else 0
    columns = st.columns(cols)

    for col_idx in range(cols):
        start_idx = col_idx * items_per_col
        end_idx = start_idx + items_per_col
        column_options = options[start_idx:end_idx]
        
        with columns[col_idx]:
            for option in column_options:
                # 为每个选项创建单选按钮
                is_selected = (st.session_state[key] == option)
                # 使用唯一key，但不直接修改其他选项的状态
                if st.checkbox(option, value=is_selected, 
                              key=f"{key}_{col_idx}_{option}"):
                    if not is_selected:
                        st.session_state[key] = option
                        # 触发重新渲染以更新所有选项状态
                        st.rerun()
    
    return st.session_state[key]


def render_prediction_page():
    """渲染预测页面"""

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        params = load_default_params(PARAMS_FILE)
        predict = load_default_params(PREDICT_FILE)

        param_display_list = []
        display_to_key = {}
        exclude_key = "key_factors.right_knee_symptoms_year2"
        for k, v in params.items():
            # if isinstance(v, (int, float, str)):
            if k == exclude_key:
                continue
            label = f"{k}"
            # else:
                # label = f"{k} (complex)"
            param_display_list.append(label)
            display_to_key[label] = k
    

        st.markdown("**Parameter Mode: `fixed parameter from Assessment Agent`**")

        with st.expander("Click to view parameters"):
            st.markdown("**The patient parameters are listed below, Click the box to view the values.**")
            selected_display = multi_column_radio(" ", param_display_list, cols=6)
            
            model = display_to_key[selected_display]
            threshold = params[model]
        
            st.markdown("**Selected value:**")
            if isinstance(threshold, dict):
                st.json(threshold)
            elif isinstance(threshold, list):
                for i, item in enumerate(threshold):
                    st.markdown(f"**Item {i+1}:**")
                    st.json(item)
            else:
                st.write(threshold)

            with st.expander("View abbreviation explanations"):
                col1_exp, col2_exp = st.columns(2) 

                with col1_exp:
                    st.markdown("**X-ray parameters (Knee):**")
                    x_ray_params = {
                        "XRKL_L": "Kellgren–Lawrence grade, left knee",
                        "XRKL_R": "Kellgren–Lawrence grade, right knee",
                        "XRJSL_L": "Joint space narrowing, lateral, left knee",
                        "XRJSM_L": "Joint space narrowing, medial, left knee",
                        "XROSFL_L": "Osteophytes, femur lateral, left knee",
                        "XROSFM_L": "Osteophytes, femur medial, left knee",
                        "XROSTL_L": "Osteophytes, tibia lateral, left knee",
                        "XROSTM_L": "Osteophytes, tibia medial, left knee",
                        "XRJSL_R": "Joint space narrowing, lateral, right knee",
                        "XRJSM_R": "Joint space narrowing, medial, right knee",
                        "XROSFL_R": "Osteophytes, femur lateral, right knee",
                        "XROSFM_R": "Osteophytes, femur medial, right knee",
                    }
                    for abbr, full_name in x_ray_params.items():
                        st.markdown(f"- **{abbr}**: {full_name}")
                    
                    st.markdown("**Demographics / Basics:**")
                    demo_params = {
                        "AGE": "Age at baseline",
                        "BMI": "Body Mass Index",
                        "WEIGHT": "Body weight (kg)",
                    }
                    for abbr, full_name in demo_params.items():
                        st.markdown(f"- **{abbr}**: {full_name}")
                
                with col2_exp:
                    st.markdown("**X-ray parameters (cont.):**")
                    x_ray_params_cont = {
                        "XROSTL_R": "Osteophytes, tibia lateral, right knee",
                        "XROSTM_R": "Osteophytes, tibia medial, right knee",
                        "XRSCFL_R": "Subchondral cyst, femur lateral, right knee",
                        "RKImg_V00": "Radiographic grade, right knee, baseline", 
                        "LKImg_V00": "Radiographic grade, left knee, baseline",  
                    }
                    for abbr, full_name in x_ray_params_cont.items():
                        st.markdown(f"- **{abbr}**: {full_name}")
                    
                    st.markdown("**Biomechanics (Force):**")
                    bio_params = {
                        "RFmaxF": "Right foot maximum forward force",
                        "REmaxF": "Right foot maximum eversion force",
                        "LFmaxF": "Left foot maximum forward force",
                        "LEmaxF": "Left foot maximum eversion force",
                        "RFmaxF_BMI": "Right foot max forward force normalized by BMI",
                        "REmaxF_BMI": "Right foot max eversion force normalized by BMI",
                        "LFmaxF_BMI": "Left foot max forward force normalized by BMI",
                        "LEmaxF_BMI": "Left foot max eversion force normalized by BMI",
                    }
                    for abbr, full_name in bio_params.items():
                        st.markdown(f"- **{abbr}**: {full_name}")
                    
                    st.markdown("**KOOS Questionnaire:**")
                    koos_params = {
                        "KOOSPain_R": "KOOS pain score, right knee, baseline",
                        "KOOSSym_R": "KOOS symptoms score, right knee, baseline",
                        "KOOSPain_L": "KOOS pain score, left knee, baseline",
                        "KOOSSym_L": "KOOS symptoms score, left knee, baseline",
                        "KOOSSport": "KOOS sport/recreation score, baseline",
                        "KOOSQOL": "KOOS quality of life score, baseline"
                    }
                    for abbr, full_name in koos_params.items():
                        st.markdown(f"- **{abbr}**: {full_name}")

        if "prediction_done" not in st.session_state:
            st.session_state["prediction_done"] = False

        if st.button("Starting prediction", type="primary"):
            with st.spinner("Analysing"):
                time.sleep(3)
            st.success("Prediction completed!")
            st.session_state["prediction_done"] = True
        
        if st.session_state["prediction_done"]:
            render_prediction_report(params)
            spacer(16)
        
            export_col1, export_col2 = st.columns([1, 1])
        
            with export_col1:
                pdf_text = generate_report_text_from_prediction(predict)
                pdf_bytes = generate_pdf(pdf_text)
        
                st.download_button(
                    label="📄 Download Prediction Report as PDF",
                    data=pdf_bytes,
                    file_name="prediction_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
            with export_col2:
                with open(PARAMS_FILE, "rb") as f:
                    json_bytes = f.read()
            
                st.download_button(
                    label="Download Prediction Report JSON",
                    data=json_bytes,
                    file_name="predict_params.json",
                    mime="application/json",
                    use_container_width=True
                )

    with col2:
        safe_image_display(IMAGE_PATHS["predicting_framework"], "Framework for predicting progress risks", use_container_width=True)


def render_agent_message_return_html(role: str, action_html: str, style_class: str) -> str:
    return f"""
    <div class="chat-bubble {style_class}">
        <div class="chat-icon"><strong>{role}</strong></div>
        <div class="chat-content" style="margin-top: 8px; text-align: left;">
            {action_html}
        </div>
    </div>
    """

def render_agent_message(role: str, action: str, style_class: str) -> str:
    action_html = markdown.markdown(action, extensions=["extra", "nl2br"])  # 转换 Markdown 为 HTML
    return f"""
    <div class="chat-bubble {style_class}">
        <div class="chat-icon"><strong>{role}</strong></div>
        <div style="margin-top: 8px; text-align: left;">{action_html}</div>
    </div>
    """

def extract_week_number(phase_name: str) -> int:
    """从 'Week 1–4'（含 en dash）中提取排序基准数字"""
    # 替换 en dash（–）和 em dash（—）为 ASCII dash（-）
    normalized = phase_name.replace("–", "-").replace("—", "-")
    match = re.search(r"Week (\d+)", normalized)
    return int(match.group(1)) if match else 0


def render_exercise_plan_return_html(plan: Dict) -> str:
    sorted_phases = sorted(plan.items(), key=lambda x: extract_week_number(x[0]))
    html_blocks = []

    for i, (phase, content) in enumerate(sorted_phases, start=1):
        goal = content.get("Goal", "")
        prescriptions = content.get("Prescription", [])

        markdown_text = f"<h4>Phase {i}: {phase}</h4>"
        markdown_text += f"<b>GOAL:</b> {goal}<br><br>"


        for item in prescriptions:
            category = item.get("Category", "Training")
            description = item.get("Description", "")
            markdown_text += f"<b>{category} Training:</b><br>"
            for part in description.split(", "):
                markdown_text += f"- {part.strip()}<br>"
            markdown_text += "<br>"

        html = render_agent_message_return_html(
            role="A. Exercise Prescriptionist Agent",
            action_html=markdown_text,
            style_class="exercise"
        )
        html_blocks.append(html)

    return "\n".join(html_blocks)


def render_surgical_pharma_plan_return_html(plan_data: Dict) -> List[str]:
    """
    返回 Surgical & Pharmacological Specialist Agent 的多个 HTML 块列表，
    每个块单独传入 st.markdown(..., unsafe_allow_html=True) 渲染。
    """
    html_blocks = []

    # Step 1: 渲染 Guideline Summary
    guideline_markdown = "#### Clinical Guideline Analysis\n\n### Matched Guidelines Summary\n\n"
    title_map = {
        "564": "Severe Functional Limitation",
        "225": "Moderate Functional Limitation with Mechanical Symptoms",
        "482": "Younger Patient with Single-Compartment Disease"
    }

    for item in plan_data.get("matched_guidelines", []):
        guideline_text = item.get("guideline", "")
        match = re.search(r"Scenario (\d+):", guideline_text)
        gid = match.group(1) if match else "Unknown"
        title = title_map.get(gid, "Clinical Scenario")

        guideline_markdown += f"**Guideline {gid}: {title}**\n"

        def extract_section(text, start_kw, end_kw=None):
            try:
                start = text.index(start_kw)
                end = text.index(end_kw, start) if end_kw else None
                return text[start + len(start_kw):end].strip()
            except ValueError:
                return ""

        clinical = extract_section(guideline_text, 'The patient reports', 'Demonstrates') or extract_section(guideline_text, 'Experiences', 'has limited') or ""
        physical = extract_section(guideline_text, 'Demonstrates', 'Shows') or extract_section(guideline_text, 'has limited', 'shows') or ""
        radio = extract_section(guideline_text, 'Shows', 'Total') or extract_section(guideline_text, 'exhibits', 'Total') or ""

        guideline_markdown += f"- **Clinical Presentation:** {clinical}\n"
        guideline_markdown += f"- **Physical Findings:** {physical}\n"
        guideline_markdown += f"- **Radiographic Features:** {radio}\n"
        guideline_markdown += f"**Recommendations:**\n"

        recos = re.findall(r"(Total knee arthroplasty|Unicompartmental knee arthroplasty.*?|Realignment Osteotomy.*?)\s*(Appropriate|May Be Appropriate|Rarely Appropriate)\s*(\d)", guideline_text)
        for rec in recos:
            guideline_markdown += f"- {rec[0]}: {rec[1]} ({rec[2]}/9)\n"
        guideline_markdown += "\n"

    guideline_html = render_agent_message_return_html(
        role="B. Surgical & Pharmacological Specialist Agent",
        action_html=markdown.markdown(guideline_markdown,extensions=["extra", "nl2br"]),
        style_class="surgical"
    )
    html_blocks.append(guideline_html)

    # Step 2: 药物推荐表格
    meds = plan_data.get("medication_plan", [])
    med_table_md = "#### Pharmacological Management Plan\n\n"
    med_table_md += "| Medication | Dosage | Administration Schedule | Notes |\n"
    med_table_md += "|------------|--------|-------------------------|-------|\n"

    for med in meds:
        name = med.get("name", "")
        dosage = med.get("dosage", "")
        freq = med.get("frequency", "")
        notes = ""
        if "Ibuprofen" in name:
            notes = "Monitor for GI effects; take with food"
        elif "Acetaminophen" in name:
            notes = "Not to exceed 3000 mg daily"
        elif "Corticosteroids" in name:
            notes = "Consider after failed oral analgesics"
        med_table_md += f"| {name} | {dosage} | {freq} | {notes} |\n"

    med_table_md += "\nNote: Medication regimen should be tailored based on patient comorbidities, concomitant medications, and individual response to therapy.\n"

    med_table_html = markdown.markdown(med_table_md, extensions=["extra", "nl2br"])
    wrapped_html = f"<div class='markdown-wrapper'>{med_table_html}</div>"

    pharma_html = render_agent_message_return_html(
        role="B. Surgical & Pharmacological Specialist Agent",
        action_html=wrapped_html,
        style_class="pharma"
    )
    html_blocks.append(pharma_html)

    return html_blocks


def render_nutrition_psychology_plan_return_html(plan_data: Dict) -> List[str]:
    """返回 Nutritional & Psychological Specialist Agent 的多个 HTML 气泡块"""
    html_blocks = []

    # ----------- Nutrition 部分 -----------
    nutrition = plan_data.get("nutrition", {})
    n_goal = nutrition.get("goal", "")
    n_duration = nutrition.get("duration", "")
    n_content = nutrition.get("content", [])

    nutrition_md = "#### Nutritional Intervention Plan\n"
    nutrition_md += f"**Goal:** {n_goal}\n\n"
    nutrition_md += f"**Delivery Method:** Personalized one-on-one counseling supplemented with mobile application reminders\n"
    nutrition_md += f"**Program Structure:**\n"
    nutrition_md += f"- **Initial Phase:** Weekly consultations (first 6 weeks)\n"
    nutrition_md += f"- **Maintenance Phase:** Bi-weekly check-ins\n"
    nutrition_md += f"- **Total Duration:** {n_duration} comprehensive program\n"

    # 分类策略
    strategies = {
        "Anti-inflammatory": [],
        "Macronutrient": [],
        "Weight": []
    }

    for item in n_content:
        if "Anti-inflammatory" in item or "Adequacy" in item:
            strategies["Anti-inflammatory"].append(item)
        elif "macronutrient" in item or "Balance" in item:
            strategies["Macronutrient"].append(item)
        elif "calorie" in item.lower() or "Calorie control" in item:
            strategies["Weight"].append(item)

    nutrition_md += "**Key Nutritional Strategies:**\n"
    if strategies["Anti-inflammatory"]:
        nutrition_md += "1. **Anti-inflammatory Focus**\n"
        nutrition_md += "   - Incorporate omega-3 rich foods (fatty fish, walnuts, flaxseeds)\n"
        nutrition_md += "   - Increase consumption of antioxidant-rich leafy greens\n"
        nutrition_md += "   - Integrate nuts and seeds for micronutrient support\n"
        nutrition_md += "   - Purpose: Reduce joint inflammation and support tissue repair\n"
    if strategies["Macronutrient"]:
        nutrition_md += "2. **Macronutrient Optimization**\n"
        nutrition_md += "   - Ensure adequate protein intake to support muscle maintenance\n"
        nutrition_md += "   - Balance complex carbohydrates for sustained energy\n"
        nutrition_md += "   - Include healthy fats to support joint lubrication\n"
        nutrition_md += "   - Purpose: Enhance musculoskeletal strength and joint function\n"
    if strategies["Weight"]:
        nutrition_md += "3. **Weight Management**\n"
        nutrition_md += "   - Implement portion awareness techniques\n"
        nutrition_md += "   - Monitor caloric balance through guided food journaling\n"
        nutrition_md += "   - Adjust intake based on activity levels and rehabilitation phases\n"
        nutrition_md += "   - Purpose: Reduce mechanical stress on knee joints\n"

    nutrition_html = render_agent_message_return_html(
        role="C. Nutritional & Psychological Specialist Agent",
        action_html=markdown.markdown(nutrition_md,extensions=["extra", "nl2br"]),
        style_class="nutrition"
    )
    html_blocks.append(nutrition_html)

    # ----------- Psychology 部分 -----------
    psych = plan_data.get("psychology", {})
    p_goal = psych.get("goal", "")
    p_duration = psych.get("duration", "")
    p_content = psych.get("content", [])

    psychology_md = "#### Psychological Support\n"
    psychology_md += f"**Goal:** {p_goal}\n\n"
    psychology_md += f"**Delivery Method:** Tele-health Cognitive Behavioral Therapy with structured daily practice components\n"
    psychology_md += f"**Program Structure:**\n"
    psychology_md += f"- **Intensive Phase:** Weekly sessions (first 8 weeks)\n"
    psychology_md += f"- **Consolidation Phase:** Bi-weekly sessions\n"
    psychology_md += f"- **Total Duration:** {p_duration} comprehensive program\n"

    psychology_md += "**Evidence-Based Psychological Approaches:**\n"
    for idx, item in enumerate(p_content, start=1):
        if "Motivational" in item:
            psychology_md += f"{idx}. **Motivational Interviewing**\n"
            psychology_md += "   - Explore personal values related to mobility and function\n"
            psychology_md += "   - Resolve ambivalence about rehabilitation commitment\n"
            psychology_md += "   - Develop intrinsic motivation for consistent exercise adherence\n"
            psychology_md += "   - Purpose: Strengthen commitment to rehabilitation protocols\n"
        elif "CBT" in item:
            psychology_md += f"{idx}. **Cognitive Restructuring**\n"
            psychology_md += "   - Identify and challenge maladaptive thoughts about pain and recovery\n"
            psychology_md += "   - Transform catastrophizing patterns into realistic perspectives\n"
            psychology_md += "   - Develop confidence in functional improvement\n"
            psychology_md += "   - Purpose: Reduce pain-related fear and enhance rehabilitation engagement\n"
        elif "mindfulness" in item.lower():
            psychology_md += f"{idx}. **Digital Mindfulness Integration**\n"
            psychology_md += "   - Implement scheduled mindfulness practice through mobile notifications\n"
            psychology_md += "   - Provide guided pain-specific meditation recordings\n"
            psychology_md += "   - Track stress levels in relation to symptom fluctuations\n"
            psychology_md += "   - Purpose: Enhance stress management and improve pain tolerance\n"

    psychology_md += "*Note: Both nutritional and psychological interventions will be coordinated with physical rehabilitation to ensure comprehensive care integration.*"

    psychology_html = render_agent_message_return_html(
        role="C. Nutritional & Psychological Specialist Agent",
        action_html=markdown.markdown(psychology_md, extensions=["extra", "nl2br"]),
        style_class="psychology"
    )
    html_blocks.append(psychology_html)

    return html_blocks


def render_clinical_decision_agent_return_html(plan_data: Dict) -> str:
    primary_goal = plan_data.get("Goals", {}).get("Primary", "")
    secondary_goal = plan_data.get("Goals", {}).get("Secondary", "")

    plan = plan_data.get("InterventionPlan", {})
    action_md = "#### Integrated Multimodal Intervention Plan\n\n"

    # Medication
    med_summary = plan.get("Medication", {}).get("Summary", "")
    action_md += "**🩺 Medication Strategy**\n"
    action_md += f"- {med_summary}\n\n"

    # Nutrition
    nutrition_desc = plan.get("NutritionPlan", {}).get("Description", "")
    framework = plan.get("NutritionPlan", {}).get("Framework", "")
    action_md += "**🥗 Nutrition Plan**\n"
    action_md += f"- **Framework:** {framework}\n"
    action_md += f"- {nutrition_desc}\n\n"

    # Exercise
    exercise = plan.get("ExercisePlan", {})
    framework = exercise.get("Framework", "")
    phases = exercise.get("Phases", {})
    action_md += "**🏃 Exercise Plan**\n"
    action_md += f"- **Framework:** {framework}\n"
    for week_range, content in phases.items():
        goal = content.get("Goal", "")
        prescription = content.get("Prescription", "")
        action_md += f"  - **{week_range}:** {goal}\n"
        action_md += f"    - {prescription}\n"
    action_md += "\n"

    # Psychology
    psych_summary = plan.get("PsychologicalSupport", {}).get("Summary", "")
    action_md += "**🧠 Psychological Support**\n"
    action_md += f"- {psych_summary}\n\n"

    # Surgical
    surgical_summary = plan.get("SurgicalOrInjectionConsiderations", {}).get("Summary", "")
    action_md += "**🛠️ Surgical or Injection Considerations**\n"
    action_md += f"- {surgical_summary}\n\n"

    # Safety
    safety_summary = plan.get("SafetyMonitoring", {}).get("Summary", "")
    action_md += "**🔍 Safety Monitoring Plan**\n"
    action_md += f"- {safety_summary}\n\n"

    # Personalization
    accessibility = plan_data.get("AccessibilityFeasibility", "")
    rationale = plan_data.get("PersonalizationRationale", "")
    evidence = plan_data.get("EvidenceCompliance", "")
    action_md += "#### Personalized Treatment Context\n"
    action_md += f"- **Accessibility & Feasibility:** {accessibility}\n"
    action_md += f"- **Personalization Rationale:** {rationale}\n"
    action_md += f"- **Evidence Compliance:** {evidence}\n"

    html = render_agent_message(
        role="🧩 Clinical Decision-Making Agent",
        action=action_md,
        style_class="decision"
    )

    return html


def render_progress_bar(step: int, total: int):
    '''进度条函数'''
    progress = step / total
    st.markdown(f"""
    <div style="background-color: #eee; height: 8px; width: 100%; border-radius: 4px; margin-bottom: 12px;">
        <div style="height: 100%; width: {progress*100:.1f}%; background-color: #4CAF50; border-radius: 4px;"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<small style='color: grey;'>Progress: {int(progress * 100)}%</small>", unsafe_allow_html=True)


def render_all_agents_auto():
    total_agents = 4
    progress_placeholder = st.empty()

    # ✅ Agent A: Exercise
    progress_placeholder.markdown(render_progress_bar_html(1, total_agents), unsafe_allow_html=True)
    exercise_plan = load_plan("exercise")
    with st.expander("A. Exercise Prescriptionist Agent", expanded=False):
        html = render_exercise_plan_return_html(exercise_plan)
        st.markdown(html, unsafe_allow_html=True)

    # ✅ Agent B: Surgical & Pharma
    progress_placeholder.markdown(render_progress_bar_html(2, total_agents), unsafe_allow_html=True)
    surgical_plan = load_plan("surgical_pharma")
    with st.expander("B. Surgical & Pharmacological Specialist Agent", expanded=False):
        # html = render_surgical_pharma_plan_return_html(surgical_plan)
        # st.markdown(html, unsafe_allow_html=True)
        html_blocks = render_surgical_pharma_plan_return_html(surgical_plan)
        for html in html_blocks:
            st.markdown(html, unsafe_allow_html=True)


    # ✅ Agent C: Nutrition & Psychology
    progress_placeholder.markdown(render_progress_bar_html(3, total_agents), unsafe_allow_html=True)
    nutrition_plan = load_plan("nutrition_psychology")
    with st.expander("C. Nutritional & Psychological Specialist Agent", expanded=False):
        # html = render_nutrition_psychology_plan_return_html(nutrition_plan)
        # st.markdown(html, unsafe_allow_html=True)
        html_blocks = render_nutrition_psychology_plan_return_html(nutrition_plan)
        for html in html_blocks:
            st.markdown(html, unsafe_allow_html=True)



    with st.spinner("Clinical Decision-Making Agent reasoning..."):
        time.sleep(3)

    progress_placeholder.markdown(render_progress_bar_html(4, total_agents), unsafe_allow_html=True)
    decision_plan = load_plan("clinical_integration")
    with st.expander("D. Clinical Decision-Making Agent", expanded=False):
        html = render_clinical_decision_agent_return_html(decision_plan)
        st.markdown(html, unsafe_allow_html=True)


# 进度条 HTML 渲染拆出来方便复用
def render_progress_bar_html(step: int, total: int) -> str:
    progress = step / total
    return f"""
    <div style="background-color: #eee; height: 8px; width: 100%; border-radius: 4px; margin-bottom: 12px;">
        <div style="height: 100%; width: {progress*100:.1f}%; background-color: #4CAF50; border-radius: 4px;"></div>
    </div>
    <small style='color: grey;'>Progress: {int(progress * 100)}%</small>
    """

def render_therapy_page():
    """渲染治疗推荐页面"""
    inject_agent_styles() 
    col1, col2 = st.columns([1.5, 1])

    with col2:
        safe_image_display(IMAGE_PATHS["recommendation_framework"], "Framework for personalizing treatment", use_container_width=True)
 
    with col1:
        st.subheader("🩺 Quick Therapy Demo")
    
        if "start_clicked" not in st.session_state:
            st.session_state.start_clicked = False
    
        case_data = load_case_data()
        if not case_data:
            st.warning("Case data cannot be loaded.")
            return
        
        case_names = list(case_data.keys())
        case_options = [""] + case_names
        
        selected_case = st.selectbox("Select a sample case：", case_options)
        
        st.markdown(get_chat_styles(), unsafe_allow_html=True)
        
        if selected_case != "":
            case = case_data[selected_case]
            
            def on_start_click():
                st.session_state.start_clicked = True
                st.query_params.update({"start": "1"})
            
            if not st.session_state.start_clicked:
                st.success(f"Selected：{selected_case}")
            
                st.markdown("### 📑 Case Reports")
                for report_name in case.get("reports", []):
                    st.markdown(f"📄 {report_name}")

                button_placeholder = st.empty()
                spacer_placeholder = st.empty()

                if button_placeholder.button(
                    "▶️ Start Multi-Agent Reasoning",
                    on_click=on_start_click,

                    key="start_reasoning_btn"
                ):

                    button_placeholder.empty()
                    spacer_placeholder.markdown(
                        "<div style='height: 48px;'></div>", 
                        unsafe_allow_html=True
                    )

                    render_all_agents_auto()
            else:

                render_all_agents_auto()

# =============================================================================
# 主程序
# =============================================================================
def main():

    st.set_page_config(**PAGE_CONFIG)

    render_navigation()

    page = st.query_params.get("page", "Home")

    page_routes = {
        "Home": render_home_page,
        "Assessing Current Status": render_assessment_page,
        "Predicting Progression Risk": render_prediction_page,
        "Tailored Therapy Recommendation": render_therapy_page
    }
    
    render_func = page_routes.get(page)
    if render_func:
        render_func()
    else:
        st.error(f"未知页面: {page}")

if __name__ == "__main__":
    main()
