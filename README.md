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


## 📚 แหล่งข้อมูลและเอกสารอ้างอิง (Data Sources)
ระบบฐานความรู้ (Knowledge Base) นี้รวบรวมข้อมูลมาจากบทความทางการเงินที่น่าเชื่อถือจำนวน 13 ฉบับ ครอบคลุมเนื้อหาตั้งแต่พื้นฐานไปจนถึงเทคนิคการวิเคราะห์ โดยแบ่งเป็นหมวดหมู่ดังนี้:

**1. พื้นฐานการลงทุน (Fundamental Concepts)**
* [ความต่างระหว่าง การออม vs การลงทุน - ธนาคารออมสิน (GSB)](https://goodmoneybygsb.com/financial-knowledge/...)
* [พลังของดอกเบี้ยทบต้น - Chubb](https://www.chubb.com/th-th/articles/personal/power-of-compound-interest.html)
* [การจัดพอร์ตและการกระจายความเสี่ยง - ธนาคารกรุงไทย](https://krungthai.com/th/financial-partner/learn-financial/1755)
* [7 พฤติกรรมการลงทุนที่ควรเรียนรู้ - Finnomena](https://www.finnomena.com/techtoro/7-behaviors/)

**2. กลยุทธ์การลงทุนในหุ้น (Stock Strategies)**
* [การสร้าง Passive Income ด้วยเงินปันผล - Yuanta](https://www.yuanta.co.th/blog/5-way-to-make-passive-income-with-Stock-Dividend/)
* [กลยุทธ์การออมหุ้น (DCA) - Phillip Securities](https://www.phillip.co.th/th/stock-dca)
* [วิธีค้นหาหุ้นตัวแรกสำหรับมือใหม่ - Phillip Securities](https://www.phillip.co.th/th/update/advice/find-the-first-stock-for-beginner)
* [วิธีการประเมินมูลค่าหุ้น - Liberator](https://www.liberator.co.th/article/view/%E0%B8%A7%E0%B8%B4%E0%B8%98%E0%B8%B5%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B9%80%E0%B8%A1%E0%B8%B4%E0%B8%99%E0%B8%A1%E0%B8%B9%E0%B8%A5%E0%B8%84%E0%B9%88%E0%B8%B2%E0%B8%AB%E0%B8%B8)

**3. กองทุนรวมและ ETF (Mutual Funds & ETFs)**
* [ETF คืออะไร? - บล.บัวหลวง](https://www.bualuang.co.th/article/whatisetf)
* [มือใหม่เลือกกองทุนรวมยังไงดี - Kept by Krungsri](https://www.keptbykrungsri.com/kept-tips/investment/how-to-choose-mutual-fund)
* [การวิเคราะห์กองทุนรวม - Finnomena](https://www.finnomena.com/podcast/club-fund-day-ss2-ep-5/)

**4. สินทรัพย์ทางเลือก (Alternative Assets)**
* [ทองคำ: สินทรัพย์ปลอดภัย - ฮั่วเซ่งเฮง](https://www.huasengheng.com/news/safe-haven-investment-gold/)
* [พื้นฐานการเทรดคริปโต - Binance TH](https://www.binance.th/th/academy/trading/4dd273aa902242a1896fb4aef19430b6)
