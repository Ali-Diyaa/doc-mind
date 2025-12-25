import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import tempfile
import json
import re

# ------------------- Force CPU -------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ------------------- Page Config -------------------
st.set_page_config(page_title="Doc Mind – Your Study Partner", layout="wide")
st.title("📘 Doc Mind – Your Study Partner")
st.markdown("Upload your PDF and enjoy a perfectly working 5-question quiz every time!")

# ------------------- Model & Embeddings -------------------
@st.cache_resource
def get_llm(json_mode=False):
    return OllamaLLM(
        model="qwen2.5:3b",
        temperature=0.0 if json_mode else 0.3,
        format="json" if json_mode else ""
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm_normal = get_llm(json_mode=False)
llm_json = get_llm(json_mode=True)
embeddings = get_embeddings()

# ------------------- PDF Processing -------------------
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever

# ------------------- Fallback Questions  -------------------
def get_fallback_questions():
    return [
        {
            "question": "Which addressing mode transfers data directly between two CPU registers?",
            "options": {"A": "Immediate", "B": "Register", "C": "Direct", "D": "Displacement"},
            "correct_answer": "B",
            "explanation": "Register addressing mode transfers data between registers, e.g., MOV AX, BX."
        },
        {
            "question": "In immediate addressing, the data is:",
            "options": {"A": "Stored in memory", "B": "In a register", "C": "Part of the instruction", "D": "At an offset"},
            "correct_answer": "C",
            "explanation": "Immediate addressing uses a constant value embedded directly in the instruction, e.g., MOV AX, 1234H."
        },
        {
            "question": "Direct addressing uses square brackets like [1234H]. What does this access?",
            "options": {"A": "A register", "B": "An immediate value", "C": "A memory location", "D": "A segment register"},
            "correct_answer": "C",
            "explanation": "Direct addressing accesses a memory location whose address is given directly, e.g., MOV AX, [1234H]."
        },
        {
            "question": "Displacement addressing is also known as:",
            "options": {"A": "Base-relative", "B": "Immediate", "C": "Register indirect", "D": "Absolute"},
            "correct_answer": "A",
            "explanation": "Displacement (or base-relative) uses a base register plus an offset, e.g., MOV AX, [BX+4]."
        },
        {
            "question": "Which register is typically used as the base in displacement addressing within the data segment?",
            "options": {"A": "AX", "B": "CX", "C": "BX", "D": "SP"},
            "correct_answer": "C",
            "explanation": "BX (Base Register) is commonly used for displacement addressing in the data segment."
        }
    ]

# ------------------- Main UI -------------------
uploaded_file = st.file_uploader("📄 Upload your PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        retriever = process_pdf(uploaded_file)
    st.success("✅ PDF processed! All features ready.")

    tab1, tab2, tab3, tab4 = st.tabs(["📘 Summary", "❓ Questions", "💬 Chatbot", "🧠 Quiz"])

    with tab1:
        st.subheader("Detailed Summary")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                prompt = "Summarize the document in 8–12 detailed bullet points covering all addressing modes."
                chain = RetrievalQA.from_chain_type(llm=llm_normal, chain_type="stuff", retriever=retriever)
                st.markdown(chain.run(prompt))

    with tab2:
        st.subheader("Generate Study Questions")
        if st.button("Generate MCQs + Essay Questions"):
            with st.spinner("Creating..."):
                prompt = "Generate 5 MCQs (mark correct) and 5 open-ended questions."
                chain = RetrievalQA.from_chain_type(llm=llm_normal, chain_type="stuff", retriever=retriever)
                st.markdown(chain.run(prompt))

    with tab3:
        st.subheader("💬 Chat with Document")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if user_input := st.chat_input("Ask..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Thinking..."):
                chain = RetrievalQA.from_chain_type(llm=llm_normal, chain_type="stuff", retriever=retriever)
                answer = chain.run(user_input)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

    # ==================== QUIZ TAB ====================
    with tab4:
        st.header("🧠 Interactive Quiz (Always 5 Questions)")

        if st.button("Start New 5-Question Quiz"):
            with st.spinner("Generating quiz..."):
                quiz_prompt = """
OUTPUT ONLY A VALID JSON ARRAY WITH EXACTLY 5 OBJECTS. NO EXTRA TEXT.

Each object must have:
{
  "question": "string",
  "options": {"A": "text", "B": "text", "C": "text", "D": "text"},
  "correct_answer": "A" or "B" or "C" or "D",
  "explanation": "short explanation"
}

Example:
[{"question":"Test?","options":{"A":"No","B":"Yes","C":"No","D":"No"},"correct_answer":"B","explanation":"Because yes."}]
Generate 5 real questions from the document now.
"""

                chain = RetrievalQA.from_chain_type(llm=llm_json, chain_type="stuff", retriever=retriever)
                raw_quiz = chain.run(quiz_prompt)

                # Clean code blocks
                cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_quiz.strip(), flags=re.MULTILINE)

                valid_questions = []

                # Try normal JSON parse
                try:
                    data = json.loads(cleaned)
                    if isinstance(data, dict):
                        data = [data]
                    for q in data:
                        if (isinstance(q.get("options"), dict) and
                            q.get("correct_answer") in q["options"] and
                            all(k in q for k in ["question", "options", "correct_answer", "explanation"])):
                            valid_questions.append(q)
                except:
                    pass

                # Fallback: extract individual objects
                if len(valid_questions) < 5:
                    matches = re.findall(r'\{[^{}]*"question"[^{}]*"options"[^{}]*"correct_answer"[^{}]*"explanation"[^{}]*\}', cleaned, re.DOTALL)
                    for m in matches:
                        try:
                            q = json.loads(m)
                            if (isinstance(q.get("options"), dict) and
                                q.get("correct_answer") in q["options"]):
                                valid_questions.append(q)
                        except:
                            continue

                if len(valid_questions) < 3:  
                    st.warning("Model output was unreliable. Using built-in accurate questions instead.")
                    valid_questions = get_fallback_questions()

                # Always ensure exactly 5
                final_quiz = valid_questions[:5]
                while len(final_quiz) < 5 and final_quiz:
                    final_quiz.append(final_quiz[-1])

                st.session_state.quiz = final_quiz
                st.session_state.q_index = 0
                st.session_state.score = 0
                st.session_state.answered = False
                st.rerun()

        # =============== PLAY QUIZ ===============
        if "quiz" in st.session_state and st.session_state.q_index < 5:
            q = st.session_state.quiz[st.session_state.q_index]

            st.markdown(f"### Question {st.session_state.q_index + 1} / 5")
            st.write(q["question"])

            options = q["options"]
            option_keys = list(options.keys())

            if not st.session_state.get("answered", False):
                choice = st.radio(
                    "Choose:",
                    options=option_keys,
                    format_func=lambda x: f"{x}) {options[x]}",
                    key=f"q_{st.session_state.q_index}"
                )

                if st.button("Submit Answer"):
                    st.session_state.user_choice = choice
                    st.session_state.answered = True
                    st.rerun()

            else:
                if st.session_state.user_choice == q["correct_answer"]:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect")
                    st.warning(f"**Correct: {q['correct_answer']} - {options[q['correct_answer']]}**")

                st.info(f"**Explanation:** {q.get('explanation', 'N/A')}")

                st.markdown("### 📖 Read the explanation carefully!")

                if st.button("➜ Continue"):
                    if st.session_state.user_choice == q["correct_answer"]:
                        st.session_state.score += 1
                    st.session_state.q_index += 1
                    st.session_state.answered = False
                    st.rerun()

        elif "quiz" in st.session_state:
            st.success(f"🎉 Quiz Complete! Score: **{st.session_state.score} / 5**")
            if st.session_state.score == 5:
                st.balloons()
            if st.button("New Quiz"):
                for k in ["quiz", "q_index", "score", "answered", "user_choice"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

else:
    st.info("👆 Upload a PDF to start.")
    st.stop()