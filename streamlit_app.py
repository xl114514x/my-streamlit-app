import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys

# 错误处理和替代方案
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError as e:
    st.error(f"ChromaDB导入失败: {e}")
    CHROMA_AVAILABLE = False

def get_retriever():
    if not CHROMA_AVAILABLE:
        st.error("ChromaDB不可用，请检查依赖安装")
        return None
    
    try:
        # 定义 Embeddings
        embedding = OpenAIEmbeddings(
            openai_api_key=st.secrets.get("OPENAI_API_KEY", "sk-bRG1dii8p3tpurPNhDXmyq1SIwd1DP4L2JwCkYyk5ltNqkVt")
        )
        
        # 向量数据库持久化路径
        persist_directory = 'data_base/vector_db/chroma'
        
        # 检查目录是否存在
        if not os.path.exists(persist_directory):
            st.error(f"向量数据库目录不存在: {persist_directory}")
            return None
        
        # 加载数据库
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        return vectordb.as_retriever()
    
    except Exception as e:
        st.error(f"创建检索器时出错: {e}")
        return None

def combine_docs(docs):
    if docs and "context" in docs:
        return "\n\n".join(doc.page_content for doc in docs["context"])
    return ""

def get_qa_history_chain():
    retriever = get_retriever()
    if not retriever:
        return None
    
    try:
        # 从 secrets 或环境变量获取 API 密钥
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("未找到 OpenAI API 密钥，请在 Streamlit secrets 中设置 OPENAI_API_KEY")
            return None
        
        llm = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0,
            openai_api_key=api_key
        )
        
        condense_question_system_template = (
            "请根据聊天记录总结用户最近的问题，"
            "如果没有多余的聊天记录则返回用户的问题。"
        )
        
        condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        
        retrieve_docs = RunnableBranch(
            (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
            condense_question_prompt | llm | StrOutputParser() | retriever,
        )
        
        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
        
        qa_chain = (
            RunnablePassthrough().assign(context=combine_docs)
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        
        qa_history_chain = RunnablePassthrough().assign(
            context=retrieve_docs, 
        ).assign(answer=qa_chain)
        
        return qa_history_chain
    
    except Exception as e:
        st.error(f"创建问答链时出错: {e}")
        return None

def gen_response(chain, input_text, chat_history):
    if not chain:
        yield "抱歉，系统初始化失败，无法回答问题。"
        return
    
    try:
        response = chain.stream({
            "input": input_text,
            "chat_history": chat_history
        })
        
        for res in response:
            if "answer" in res.keys():
                yield res["answer"]
    
    except Exception as e:
        yield f"生成回答时出错: {e}"

def main():
    st.set_page_config(
        page_title="问答助手",
        page_icon="🦜",
        layout="wide"
    )
    
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    
    # 检查必要的依赖
    if not CHROMA_AVAILABLE:
        st.error("系统依赖缺失，请联系管理员")
        st.stop()
    
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        with st.spinner("正在初始化系统..."):
            st.session_state.qa_history_chain = get_qa_history_chain()
    
    # 检查系统是否初始化成功
    if not st.session_state.qa_history_chain:
        st.error("系统初始化失败，请检查配置")
        st.stop()
    
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    
    # 显示整个对话历史
    for message in st.session_state.messages:  # 遍历对话历史
        with messages.chat_message(message[0]):  # messages指在容器下显示，chat_message显示用户及ai头像
            st.write(message[1])  # 打印内容
    
    if prompt := st.chat_input("请输入您的问题"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        
        # 生成回复
        answer_generator = gen_response(
            chain=st.session_state.qa_history_chain,
            input_text=prompt,
            chat_history=st.session_state.messages
        )
        
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer_generator)
        
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))

if __name__ == "__main__":
    main()
