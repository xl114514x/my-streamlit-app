import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
import docx2txt
import PyPDF2
import io
import time
import json

# 页面配置
st.set_page_config(
    page_title="动手学大模型应用开发", 
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-container {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 1rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
        
    /* 修改按钮组样式 - 放在左下角 */
    .message-actions {
        display: flex;
        gap: 5px;
        margin-top: 8px;
        align-items: center;
        justify-content: flex-start;
    }
        
    /* 简化的按钮样式 - 只显示图标，无边框 */
    .action-button {
        background: transparent;
        border: none;
        border-radius: 4px;
        padding: 4px;
        cursor: pointer;
        font-size: 16px;
        color: #666;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        text-decoration: none;
        font-family: inherit;
    }
        
    .action-button:hover {
        background: #f0f0f0;
        color: #333;
        transform: scale(1.1);
    }
        
    /* 修改后的复制按钮样式 - 添加白色边框 */
    .copy-button {
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 6px;
        cursor: pointer;
        font-size: 18px;
        color: #666;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
        
    .copy-button:hover {
        background: #f8f9fa;
        border-color: #adb5bd;
        color: #333;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
        
    .copy-button.copied {
        color: #28a745;
        background: #f0f8f0;
        border-color: #28a745;
    }
        
    /* 重新生成按钮样式 */
    .regenerate-button {
        color: #666;
    }
        
    .regenerate-button:hover {
        color: #333;
        background: #f0f0f0;
    }
        
    .regenerate-button.loading {
        color: #ffc107;
        background: #fff8e1;
    }
        
    /* 状态提示样式 */
    .status-message {
        font-size: 11px;
        color: #28a745;
        margin-left: 5px;
        display: none;
        align-items: center;
    }
        
    .status-message.show {
        display: inline-flex;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 初始化会话状态 ----------
def initialize_session_state():
    """初始化会话状态变量"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # 用于存储重新生成的请求
    if "regenerate_question" not in st.session_state:
        st.session_state.regenerate_question = None
        
    if "regenerate_index" not in st.session_state:
        st.session_state.regenerate_index = None

# ---------- 重新生成功能的Streamlit组件 ----------
# 修改后的 create_message_actions 函数
def create_message_actions(message_index, message_text, question=None):
    """创建消息操作按钮组"""
    col1, col2 = st.columns([1, 1])  # 创建两列，每列宽度相等

    with col1:  # 复制按钮
        if st.button("📋", key=f"copy_{message_index}", help="复制消息到剪贴板"):
            # 使用JavaScript复制功能
            copy_js = f"""
            <script>
            navigator.clipboard.writeText(`{message_text.replace('`', '\\`').replace('\\', '\\\\')}`).then(function() {{
                console.log('复制成功');
            }}).catch(function(err) {{
                console.error('复制失败:', err);
            }});
            </script>
            """
            st.components.v1.html(copy_js, height=0)
            st.success("已复制到剪贴板！", icon="✅")

    with col2:  # 重新生成按钮（仅对AI回答显示）
        if question is not None:
            if st.button("🔄", key=f"regen_{message_index}", help="重新生成回答"):
                # 设置重新生成的请求
                st.session_state.regenerate_question = question
                st.session_state.regenerate_index = message_index
                st.rerun()

# 修改后的 create_copy_button_html 函数 - 添加白色边框样式并修复显示问题
def create_copy_button_html(message_index, message_text):
    """创建带白色边框的复制按钮HTML"""
    # 转义文本中的特殊字符
    escaped_text = message_text.replace('\\', '\\\\').replace('`', '\\`').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

    copy_html = f'''
    <div style="display: flex; align-items: center; gap: 10px; margin: 5px 0; height: 40px;">
        <button onclick="copyToClipboard{message_index}()"
                class="copy-button"
                style="background: white; border: 1px solid #ddd; border-radius: 6px; padding: 6px; cursor: pointer; font-size: 18px; color: #666; transition: all 0.2s ease; display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex-shrink: 0;"
                onmouseover="this.style.background='#f8f9fa'; this.style.borderColor='#adb5bd'; this.style.color='#333'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 6px rgba(0,0,0,0.15)';"
                onmouseout="this.style.background='white'; this.style.borderColor='#ddd'; this.style.color='#666'; this.style.transform='translateY(0px)'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)';">
            📋
        </button>
        <span id="copy-status-{message_index}" style="color: #28a745; font-size: 12px; line-height: 1;"></span>
    </div>

    <script>
    function copyToClipboard{message_index}() {{
        const text = `{escaped_text}`;
        const statusElement = document.getElementById('copy-status-{message_index}');
        const button = event.target;

        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                statusElement.textContent = '✅ 已复制';
                button.style.color = '#28a745';
                button.style.background = '#f0f8f0';
                button.style.borderColor = '#28a745';
                setTimeout(() => {{
                    statusElement.textContent = '';
                    button.style.color = '#666';
                    button.style.background = 'white';
                    button.style.borderColor = '#ddd';
                }}, 2000);
            }}).catch(function(err) {{
                fallbackCopy{message_index}(text, statusElement, button);
            }});
        }} else {{
            fallbackCopy{message_index}(text, statusElement, button);
        }}
    }}

    function fallbackCopy{message_index}(text, statusElement, button) {{
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        try {{
            const successful = document.execCommand('copy');
            if (successful) {{
                statusElement.textContent = '✅ 已复制';
                button.style.color = '#28a745';
                button.style.background = '#f0f8f0';
                button.style.borderColor = '#28a745';
            }} else {{
                statusElement.textContent = '❌ 复制失败';
            }}
        }} catch (err) {{
            statusElement.textContent = '❌ 复制失败';
        }}

        document.body.removeChild(textArea);
        setTimeout(() => {{
            statusElement.textContent = '';
            button.style.color = '#666';
            button.style.background = 'white';
            button.style.borderColor = '#ddd';
        }}, 2000);
    }}
    </script>
    '''

    return copy_html

# ---------- 处理重新生成请求 ----------
def handle_regenerate_request():
    """处理重新生成回答的请求"""
    if st.session_state.regenerate_question is not None and st.session_state.regenerate_index is not None:
        question = st.session_state.regenerate_question
        message_index = st.session_state.regenerate_index
                
        try:
            # 找到对应的问题在messages中的位置
            # message_index是AI回答的索引，对应的问题应该在前一个位置
            if message_index > 0 and message_index < len(st.session_state.messages):
                # 移除要重新生成的AI回答
                st.session_state.messages = st.session_state.messages[:message_index]
                                
                # 同样调整对话历史
                # 每个用户问题对应一个 HumanMessage 和一个 AIMessage
                pairs_to_keep = message_index // 2
                st.session_state.chat_history = st.session_state.chat_history[:pairs_to_keep * 2]
                                
                # 添加问题（如果还没有）
                if not st.session_state.messages or st.session_state.messages[-1][0] != "user":
                    st.session_state.messages.append(("user", question))
                    st.session_state.chat_history.append(HumanMessage(content=question))
                                
                # 清除重新生成请求
                st.session_state.regenerate_question = None
                st.session_state.regenerate_index = None
                                
                return question  # 返回问题以便重新生成回答
                        
        except Exception as e:
            st.error(f"处理重新生成请求时出错: {e}")
            st.session_state.regenerate_question = None
            st.session_state.regenerate_index = None
        
    return None

# ---------- 从本地 Markdown 文件获取文档内容 ----------
def fetch_document_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"文件未找到: {file_path}")
        return ""
    except Exception as e:
        st.error(f"无法读取文件: {e}")
        return ""

# ---------- 处理上传文件 ----------
def process_uploaded_file(uploaded_file):
    """处理上传的文件并提取文本内容"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
                
        if file_extension == 'txt':
            content = str(uploaded_file.read(), "utf-8")
        elif file_extension == 'md':
            content = str(uploaded_file.read(), "utf-8")
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                content += f"\n[第{page_num+1}页]\n{page_text}"
        elif file_extension in ['docx', 'doc']:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
                        
            content = docx2txt.process(tmp_file_path)
            os.unlink(tmp_file_path)  # 删除临时文件
        else:
            st.error(f"不支持的文件格式: {file_extension}")
            return ""
                
        # 验证内容是否成功提取
        if not content or len(content.strip()) == 0:
            st.error(f"文件 {uploaded_file.name} 内容为空或无法提取")
            return ""
                
        st.success(f"成功提取 {uploaded_file.name}，内容长度: {len(content)} 字符")
        return content
            
    except Exception as e:
        st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
        return ""

# ---------- 测试检索器功能 ----------
def test_retriever(question="测试"):
    """测试检索器是否正常工作"""
    try:
        retriever = build_retriever()
        if retriever:
            docs = retriever.invoke(question)
            return docs
        return []
    except Exception as e:
        st.error(f"测试检索器时出错: {e}")
        return []

# ---------- 构建检索器 ----------
def build_retriever():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("请先设置环境变量 OPENAI_API_KEY")
        st.stop()
        
    all_docs = []
        
    # 从本地文件获取文档内容
    DOCUMENT_FILE_PATH = "测试.md"
    if os.path.exists(DOCUMENT_FILE_PATH):
        raw_docs = fetch_document_from_file(DOCUMENT_FILE_PATH)
        if raw_docs:
            all_docs.append(raw_docs)
        
    # 添加上传的文档内容
    if 'uploaded_docs' in st.session_state and st.session_state.uploaded_docs:
        all_docs.extend(st.session_state.uploaded_docs)
        st.info(f"已加载 {len(st.session_state.uploaded_docs)} 个上传文档到知识库")
        
    if not all_docs:
        st.warning("没有找到任何文档内容，AI将仅使用自身知识回答问题")
        return None
        
    # 切分长文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(all_docs)
        
    st.info(f"知识库已构建，包含 {len(docs)} 个文档片段")
        
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- 构建问答链（带记忆功能） ----------
def get_qa_chain_with_memory():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        
    # 改进的系统提示，允许模型在找不到相关信息时使用自身知识
    system = (
        "你是一个乐于助人的 AI 助手。\n"
        "请首先基于下面提供的上下文信息回答问题。如果上下文中包含相关信息，请优先使用这些信息。"
        "如果上下文中没有相关信息，请使用你的知识和经验来回答问题，并在回答开头说明'基于我的一般知识'。\n"
        "请保持回答的准确性和有用性。\n\n"
        "上下文信息:\n{context}\n\n"
        "请结合对话历史和上下文信息来回答用户的问题。"
    )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
        
    def format_docs(docs):
        if not docs:
            return "没有找到相关的上下文信息。"
        formatted = "\n\n".join(d.page_content for d in docs)
        return formatted
        
    def get_context_and_question(inputs):
        # 每次调用时重新构建检索器，确保使用最新的文档
        retriever = build_retriever()
        if retriever:
            try:
                context_docs = retriever.invoke(inputs["question"])
                context = format_docs(context_docs)
                if context != "没有找到相关的上下文信息。":
                    st.info(f"从知识库中找到 {len(context_docs)} 个相关文档片段")
                else:
                    st.info("未在知识库中找到相关信息，将使用AI一般知识回答")
            except Exception as e:
                st.warning(f"检索时出错: {e}")
                context = "检索出错，没有找到相关的上下文信息。"
        else:
            context = "没有找到相关的上下文信息。"
            st.info("知识库为空，将使用AI一般知识回答")
                
        return {
            "context": context,
            "question": inputs["question"],
            "chat_history": inputs["chat_history"]
        }
        
    chain = (
        get_context_and_question
        | prompt
        | llm
        | StrOutputParser()
    )
        
    return chain

# ---------- 侧边栏功能 ----------
def setup_sidebar():
    with st.sidebar:
        st.markdown("### 📁 文件上传")
                
        # 文件上传器
        uploaded_files = st.file_uploader(
            "上传文档文件",
            type=['txt', 'md', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="支持的格式：TXT, MD, PDF, DOCX, DOC"
        )
                
        if uploaded_files:
            # 初始化会话状态
            if 'uploaded_docs' not in st.session_state:
                st.session_state.uploaded_docs = []
            if 'uploaded_files_info' not in st.session_state:
                st.session_state.uploaded_files_info = []
                        
            # 处理新上传的文件
            existing_files = [info['name'] for info in st.session_state.uploaded_files_info]
            new_files_processed = 0
                        
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in existing_files:
                    with st.spinner(f"正在处理文件: {uploaded_file.name}"):
                        content = process_uploaded_file(uploaded_file)
                        if content:
                            # 添加文档内容
                            st.session_state.uploaded_docs.append(content)
                                                        
                            # 保存文件信息
                            st.session_state.uploaded_files_info.append({
                                'name': uploaded_file.name,
                                'size': uploaded_file.size,
                                'content_length': len(content)
                            })
                                                        
                            new_files_processed += 1
                            st.success(f"✅ {uploaded_file.name} 处理成功！")
                        else:
                            st.error(f"❌ {uploaded_file.name} 处理失败！")
                        
            if new_files_processed > 0:
                # 重新构建chain（每次都会重新构建检索器）
                st.session_state.chain = get_qa_chain_with_memory()
                st.success(f"🎉 成功处理 {new_files_processed} 个新文件！知识库已更新。")
                st.rerun()
                
        # 显示已上传的文件
        if 'uploaded_files_info' in st.session_state and st.session_state.uploaded_files_info:
            st.markdown("### 📋 已上传文件")
            for i, file_info in enumerate(st.session_state.uploaded_files_info):
                with st.expander(f"📄 {file_info['name']}", expanded=False):
                    st.write(f"**文件大小:** {file_info['size']} bytes")
                    st.write(f"**内容长度:** {file_info['content_length']} 字符")
                                        
                    # 显示文档内容预览
                    if 'uploaded_docs' in st.session_state and i < len(st.session_state.uploaded_docs):
                        preview = st.session_state.uploaded_docs[i][:200] + "..." if len(st.session_state.uploaded_docs[i]) > 200 else st.session_state.uploaded_docs[i]
                        st.text_area("内容预览:", preview, height=100, disabled=True)
                
        st.markdown("---")
                
        # 显示知识库状态
        if 'uploaded_docs' in st.session_state:
            total_chars = sum(len(doc) for doc in st.session_state.uploaded_docs)
            st.markdown("### 📊 知识库状态")
            st.metric("文档数量", len(st.session_state.uploaded_docs))
            st.metric("总字符数", f"{total_chars:,}")
                        
            # 测试检索功能
            if st.button("🔍 测试知识库检索", use_container_width=True):
                test_query = st.text_input("输入测试问题:", value="学生手册", key="test_query")
                if test_query:
                    with st.spinner("正在测试检索..."):
                        docs = test_retriever(test_query)
                        if docs:
                            st.success(f"✅ 检索成功！找到 {len(docs)} 个相关片段")
                            with st.expander("查看检索结果"):
                                for i, doc in enumerate(docs):
                                    st.write(f"**片段 {i+1}:**")
                                    st.write(doc.page_content[:300] + "...")
                        else:
                            st.warning("⚠️ 未找到相关内容")
                
        st.markdown("---")
                
        # 清除对话历史按钮
        if st.button("🗑️ 清除对话历史", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.regenerate_question = None
            st.session_state.regenerate_index = None
            st.success("对话历史已清除！")
            st.rerun()
                
        # 清除上传文件按钮
        if st.button("📁 清除上传文件", use_container_width=True):
            if 'uploaded_docs' in st.session_state:
                del st.session_state.uploaded_docs
            if 'uploaded_files_info' in st.session_state:
                del st.session_state.uploaded_files_info
            st.session_state.chain = get_qa_chain_with_memory()
            st.success("所有上传文件已清除！")
            st.rerun()
                
        st.markdown("---")
                
        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            **功能特点：**
            - 🔍 智能检索：从知识库中查找相关信息  
            - 🧠 知识融合：找不到时使用AI自身知识回答
            - 💭 对话记忆：记住之前的对话内容
            - 📁 文件上传：支持多种格式文档
            - 📋 一键复制：直接点击复制AI回答到剪贴板
            - 🔄 重新回答：不满意可重新生成回答
                        
            **使用方法：**
            1. 上传相关文档文件（会自动处理并加入知识库）
            2. 在下方输入框中提问
            3. AI会结合文档内容和对话历史回答
            4. 使用底部按钮进行复制或重新生成
                        
            **注意事项：**
            - 文件上传后会自动构建知识库
            - 大文件处理可能需要几秒钟时间
            - 支持同时上传多个文件
            - 复制功能支持现代浏览器的一键复制
            - 重新回答会基于相同问题生成新答案
            """)

# ---------- 生成AI回答的函数 ----------
def generate_ai_response(prompt, msgs):
    """生成AI回答"""
    try:
        # 准备输入数据，包含对话历史
        chain_input = {
            "question": prompt,
            "chat_history": st.session_state.chat_history
        }
                
        # 显示处理状态
        with st.spinner("正在思考中..."):
            # 流式输出回答
            response = st.write_stream(st.session_state.chain.stream(chain_input))
                
        # 保存消息到历史记录
        st.session_state.messages.append(("assistant", response))
                
        # 更新对话历史（用于记忆功能）
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])
                
        # 限制对话历史长度，避免token过多
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]
                
        # 添加复制按钮和重新生成按钮
        message_index = len(st.session_state.messages) - 1
                
        # 使用HTML按钮组（水平排列）
        st.markdown("---")  # 添加分隔线
        
        # 创建水平排列的按钮
        button_col1, button_col2, button_col3 = st.columns([1, 1, 8])
        
        with button_col1:
            copy_html = create_copy_button_html(message_index, response)
            st.components.v1.html(copy_html, height=50)
                
        with button_col2:
            # 重新生成按钮
            if st.button("🔄", key=f"regen_new_{message_index}", help="重新生成回答"):
                st.session_state.regenerate_question = prompt
                st.session_state.regenerate_index = message_index
                st.rerun()
                    
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append(("assistant", "抱歉，生成回答时出现了错误。请检查网络连接和API密钥。"))
        # 显示详细错误信息供调试
        with st.expander("错误详情"):
            st.code(str(e))

# ---------- Streamlit 主界面 ----------
def main():
    # 初始化会话状态
    initialize_session_state()
    
    # 页面标题（美化版）
    st.markdown("""
    <style>
    .custom-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        padding: 1rem;
        color: white;
        background: linear-gradient(to right, #667eea, #764ba2);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    </style>
    <div class="custom-title">🌐 重庆科技大学 · 智能问答系统</div>
    """, unsafe_allow_html=True)

    # 设置侧边栏
    setup_sidebar()

    # 初始化问答链
    if "chain" not in st.session_state:
        st.session_state.chain = get_qa_chain_with_memory()

    # 主聊天区域
    st.markdown("### 💬 智能问答")

    # 处理重新生成请求
    regenerate_question = handle_regenerate_request()

    # 聊天消息容器
    msgs = st.container(height=500)

    # 显示聊天历史（带头像 + 美化气泡）
    for i, (role, text) in enumerate(st.session_state.messages):
        avatar = "🧑‍💻" if role == "user" else "🚀"
        bubble_color = "#f0f2f6" if role == "user" else "#e6f0ff"

        with msgs.chat_message(role, avatar=avatar):
            st.markdown(f"""
            <div style="
                background-color: {bubble_color};
                padding: 1rem;
                border-radius: 12px;
                max-width: 90%;
                display: inline-block;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin: 5px 0;
                text-align: left;
            ">
            {text}
            </div>
            """, unsafe_allow_html=True)

            # 添加按钮（仅 assistant 有）
            if role == "assistant":
                question = st.session_state.messages[i-1][1] if i > 0 and st.session_state.messages[i-1][0] == "user" else None
                st.markdown("<div style='margin-top: 8px;'>", unsafe_allow_html=True)
                button_col1, button_col2, _ = st.columns([1, 1, 8])
                with button_col1:
                    copy_html = create_copy_button_html(i, text)
                    st.components.v1.html(copy_html, height=50)
                with button_col2:
                    if question:
                        if st.button("🔄", key=f"regen_history_{i}", help="重新生成回答"):
                            st.session_state.regenerate_question = question
                            st.session_state.regenerate_index = i
                            st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # 如果有重新生成请求
    if regenerate_question:
        with msgs.chat_message("assistant", avatar="🚀"):
            st.info("🔄 正在重新生成回答...")
            generate_ai_response(regenerate_question, msgs)
        st.rerun()

    # 用户输入框
    if prompt := st.chat_input("请输入你的问题..."):
        st.session_state.messages.append(("user", prompt))
        with msgs.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)
        with msgs.chat_message("assistant", avatar="🚀"):
            generate_ai_response(prompt, msgs)


# 程序入口
if __name__ == "__main__":
    main()

