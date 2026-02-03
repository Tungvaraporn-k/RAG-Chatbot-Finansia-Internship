# Finansia AI Assistant: RAG Chatbot

โปรเจกต์นี้เป็น Chatbot ที่ปรึกษาการเงินที่พัฒนาขึ้นสำหรับโจทย์ AI/ML Intern Take-Home Assignment โดยใช้เทคนิค RAG (Retrieval-Augmented Generation) เพื่อให้คำตอบที่อ้างอิงจากเอกสารข้อมูลจริง ลดปัญหาการมั่วข้อมูล (Hallucination)

## 🛠️ Tech Stack
* **Framework:** Streamlit (Web UI)
* **LLM:** Google Gemini (model: `gemini-flash-latest`)
* **RAG Engine:** LangChain
* **Vector Store:** FAISS
* **Embeddings:** HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)

## ⚙️ วิธีการติดตั้งและรันโปรแกรม (Setup Instructions)

1. **Clone หรือดาวน์โหลดโปรเจกต์**
2. **ติดตั้ง Libraries ที่จำเป็น:**
   ```bash
   pip install streamlit langchain-google-genai langchain-community faiss-cpu sentence-transformers chromadb pypdf
