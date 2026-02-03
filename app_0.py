import streamlit as st
import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# ⚙️ Configuration
# ==========================================
os.environ["GOOGLE_API_KEY"] = "Api key................" 

# ==========================================
# 1. RAG Backend Logic
# ==========================================
@st.cache_resource
def init_knowledge_base(data_folder="data"):
    files = glob.glob(f"{data_folder}/*.txt") + glob.glob(f"{data_folder}/*.pdf")
    
    if not files:
        st.error(f"⚠️ ไม่พบไฟล์ในโฟลเดอร์ {data_folder}")
        return None

    documents = []
    for file in files:
        try:
            if file.endswith('.pdf'): 
                loader = PyPDFLoader(file)
            else:
                loader = TextLoader(file, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"❌ อ่านไฟล์ {file} ไม่ได้: {e}")
            
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    with st.spinner("🔄 กำลังสร้างระบบเข้าใจภาษาไทย..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
    
    return vector_db

def ask_rag_bot(query, vector_db):
    prompt_template = """
    คุณคือผู้ช่วยการเงินของ Finansia (Finansia AI Assistant)
    ตอบคำถามโดยใช้ข้อมูลจาก Context ด้านล่างนี้เท่านั้น
    
    กฎเหล็ก:
    1. ถ้าหาคำตอบใน Context ไม่เจอ ให้บอกตรงๆ ว่า "ขออภัย ไม่พบข้อมูลในเอกสารอ้างอิง"
    2. ห้ามมั่วหรือแต่งเรื่องเองเด็ดขาด
    3. ตอบให้กระชับและเป็นประโยชน์

    ข้อมูลอ้างอิง (Context): {context}
    คำถาม (Question): {question}
    คำตอบ:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3,
            transport="rest"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        response = qa_chain.invoke({"query": query})
        return response

    except Exception as e:
        return {"result": f"เกิดข้อผิดพลาด: {e}", "source_documents": []}

# ==========================================
# 2. Streamlit UI (Frontend)
# ==========================================
def main():
    st.set_page_config(page_title="Finansia AI Assistant", layout="wide")
    st.title("💰 Finansia AI Assistant (RAG Chatbot)")

    if "vector_db" not in st.session_state:
        db = init_knowledge_base("data")
        if db:
            st.session_state.vector_db = db
            st.toast("ฐานข้อมูลพร้อมใช้งาน!", icon="✅")

    if "vector_db" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("❓ สอบถามเรื่องการลงทุน...")

        if query:
            st.chat_message("user").write(query)
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("assistant"):
                with st.spinner("🤖 AI กำลังค้นหาคำตอบ..."):
                    response = ask_rag_bot(query, st.session_state.vector_db)
                    answer_text = response["result"]
                    
                    unique_sources = {os.path.basename(doc.metadata['source']) for doc in response.get("source_documents", [])}
                    
                    st.write(answer_text)
                    
                    final_msg = answer_text
                    if unique_sources:
                        st.markdown("---")
                        st.caption("📚 **แหล่งอ้างอิง:**")
                        source_text = "\n\n📚 **แหล่งอ้างอิง:**"
                        for src in unique_sources:
                            st.caption(f"- 📄 {src}")
                            source_text += f"\n- 📄 {src}"
                        final_msg += source_text

            st.session_state.messages.append({"role": "assistant", "content": final_msg})

if __name__ == "__main__":
    main()