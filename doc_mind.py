#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from evaluate import load as load_metric
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# In[2]:


import os
from dotenv import load_dotenv

load_dotenv()

MODEL = "qwen2.5:3b"


# In[3]:


from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
model = OllamaLLM(model = MODEL)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

model.invoke("tell me a joke")

def clean_output(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



# In[4]:


from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = model | parser
chain.invoke("tell me a joke")


# In[5]:


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Lec03_Adressing Modes_1 (2).pdf")
pages = loader.load_and_split()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       
    chunk_overlap=100,   
    separators=["\n\n", "\n", ".", " "]  
)
docs = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={'normalize_embeddings': True}   
)

vectorstore = DocArrayInMemorySearch.from_documents(
    docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


# In[6]:


from langchain.prompts import PromptTemplate

template = """
Answer the question based on the Context below, If you can't 
answer the question, reply "I don't know"
Answer directly without showing your reasoning. Do not use <think> tags only show the Answer.
and if the Question was summarize the document so You are an expert teaching assistant. Summarize the document in a clear, structured way. 
Be concise but comprehensive and summarize it in 2-3 lines and please cover all the points explained in the document. 
Context: {context}

Question: {question}

"""
prompt = PromptTemplate.from_template(template)
print(prompt.format(context = "Context" , question = "This is a question"))


# In[7]:


chain = prompt | model | parser


# In[8]:


chain.invoke({
    "context":"my name is Ali",
    "question" : "What is my name"

})


# In[9]:


results = retriever.invoke("MOV instruction transfers data between what and what in Direct Adressing?")


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1) Embed the query
query ="MOV instruction transfers data between what and what in Direct Adressing?"
query_emb = embeddings.embed_query(query)
query_emb = np.array(query_emb).reshape(1, -1)

# 2) Embed returned docs
doc_embeddings = []
for doc in results:
    emb = embeddings.embed_query(doc.page_content)  
    doc_embeddings.append(emb)

doc_embeddings = np.array(doc_embeddings)

# 3) Compute cosine similarity
similarities = cosine_similarity(query_emb, doc_embeddings)[0]

# 4) Print results
for i, (doc, score) in enumerate(zip(results, similarities)):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Cosine similarity: {score:.4f}")
    print(doc.page_content[:200].replace("\n", " "), "...")


# In[11]:


from operator import itemgetter

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)
query = itemgetter("question")
response = chain.invoke({"question":"Summarize the whole document please"})
print(response)


# In[12]:


import re
from evaluate import load

bertscore = load("bertscore")

reference = """
The document discusses various addressing modes in computer architecture,
 including register addressing, immediate addressing, direct addressing,
 and displacement addressing.
 It explains how these modes transfer data between a memory location within the Data Segment
 (DS) or another segment like ES, and either registers or other memory locations,
 providing detailed examples for each mode to illustrate their use.
"""

cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

scores = bertscore.compute(predictions=[cleaned_response],
                           references=[reference],
                           lang="en")

best_score = scores["f1"][0] * 100
print(cleaned_response)
print(f"Best semantic similarity percentage: {best_score:.2f}%")


# In[13]:


import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained sentence embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

reference = """
The document discusses various addressing modes in computer architecture,
 including register addressing, immediate addressing, direct addressing,
 and displacement addressing.
 It explains how these modes transfer data between a memory location within the Data Segment
  or another segment like , and either registers or other memory locations,
"""


cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

ref_embedding = embedding_model.encode([reference])
resp_embedding = embedding_model.encode([cleaned_response])

similarity = cosine_similarity(resp_embedding, ref_embedding)[0][0] * 100

print(cleaned_response)
print(f"Best semantic similarity percentage (cosine): {similarity:.2f}%")


# In[14]:


def get_full_context(docs):
    return " ".join([doc.page_content for doc in docs])

full_context = get_full_context(docs)
print(full_context[:500])  



# In[15]:


qg_template = """
You are an expert teaching assistant.

Based on the following context, generate:
- 5 multiple-choice questions (MCQs) with 4 options each (A, B, C, D) and specify the correct answer clearly.
- 5 essay or open-ended questions that test understanding, analysis, or explanation.

Make sure all questions are clear, relevant, and cover the most important facts, concepts, and ideas in the context.
Avoid repeating questions and ensure variety, with giving the answer for each question


Context:
{context}
"""

qg_prompt = PromptTemplate.from_template(qg_template)


# In[16]:


qg_chain = (qg_prompt | model | parser)


# In[17]:


generated_questions = qg_chain.invoke({"context": full_context})


# In[18]:


print("===== Generated Questions =====\n")
print(generated_questions)


# In[19]:


import os
os.environ["GROQ_API_KEY"] = "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"


# In[20]:


from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# 1. Build vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

pages = loader.load_and_split()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents(pages)

vectorstore = DocArrayInMemorySearch.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# 2. Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# 3. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and concise assistant."),
    ("system", "If the answer is not found in context, respond: 'I don't know from the documents.'"),
    ("human", "Conversation History:\n{history}\n\nRelevant context:\n{context}\n\nUser Question:\n{question}")
])


# In[21]:


# 4. Build RAG chain 
rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),  
        "history": lambda x: x["history"],
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Chat loop
history = ""

print("Chatbot ready! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    answer = rag_chain.invoke({
        "question": user_input,
        "history": history
    })

    print("AI:", answer)

    history += f"\nUser: {user_input}\nAI: {answer}\n"


# # Quiz

# In[22]:


quiz_prompt = PromptTemplate(
    template="""
You are an expert examiner.

Using ONLY the context below, generate {num_questions} multiple-choice questions.

Rules:
- Each question has exactly 4 options (A, B, C, D)
- One correct answer
- Based strictly on the context
- Difficulty: university exam level

Format EXACTLY like this:

[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct_answer": "B",
    "concept": "Short concept name"
  }}
]

Context:
{context}
""",
    input_variables=["context", "num_questions"]
)


# In[23]:


quiz_chain = (
    {
        "context": lambda x: " ".join(
            d.page_content for d in retriever.invoke("Generate exam questions")
        ),
        "num_questions": lambda x: x["num_questions"]
    }
    | quiz_prompt
    | llm
    | StrOutputParser()
)


# In[24]:


import json

def start_quiz(num_questions=5):
    raw = quiz_chain.invoke({"num_questions": num_questions})
    quiz = json.loads(raw)

    st.session_state.quiz = quiz
    st.session_state.q_index = 0
    st.session_state.score = 0
    st.session_state.mistakes = []
    st.session_state.quiz_active = True


# In[25]:


def show_question():
    q = st.session_state.quiz[st.session_state.q_index]

    st.subheader(f"Question {st.session_state.q_index + 1}")
    st.write(q["question"])

    choice = st.radio(
        "Choose an answer:",
        list(q["options"].keys()),
        format_func=lambda x: f"{x}) {q['options'][x]}"
    )

    if st.button("Submit Answer"):
        check_answer(choice, q)


# In[26]:


def check_answer(user_choice, q):
    correct = q["correct_answer"]

    if user_choice == correct:
        st.success("Correct!")
        st.session_state.score += 1
    else:
        st.error(f"Wrong! Correct answer: {correct}")

        st.session_state.mistakes.append(q["concept"])

        explanation = explain_answer(q, user_choice)
        st.info("📘 Explanation from document:")
        st.write(explanation)

    st.session_state.q_index += 1


# In[27]:


explain_prompt = PromptTemplate(
    template="""
You are a teaching assistant.

Explain why the correct answer is correct and the student's answer is wrong.
Use ONLY the context.

Question:
{question}

Correct Answer:
{correct}

Student Answer:
{student}

Context:
{context}
""",
    input_variables=["question", "correct", "student", "context"]
)

def explain_answer(q, student_choice):
    context_docs = retriever.invoke(q["question"])
    context = " ".join(d.page_content for d in context_docs)

    return (
        explain_prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "question": q["question"],
        "correct": q["options"][q["correct_answer"]],
        "student": q["options"][student_choice],
        "context": context
    })


# In[28]:


def show_results():
    st.subheader("Quiz Finished")
    st.write(f"Score: {st.session_state.score} / {len(st.session_state.quiz)}")

    if st.session_state.mistakes:
        st.warning(" Weak Topics:")
        for c in set(st.session_state.mistakes):
            st.write(f"- {c}")
    else:
        st.success("Perfect score")


# # Streamlit

# In[23]:


get_ipython().system('pip install streamlit')


# In[29]:


import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import os

# ------------------- Streamlit UI Setup -------------------
st.set_page_config(page_title="Doc Mind – Your Study Partner")
st.title("📘 Doc Mind – Your Study Partner")

# ------------------- Load Model & Embeddings -------------------
@st.cache_resource
def get_llm():
    return OllamaLLM(model="deepseek-r1:1.5b")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

llm = get_llm()
embeddings = get_embeddings()

# ------------------- PDF Handling -------------------
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Create FAISS vector store for local semantic search
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever, docs

# ------------------- Summarization -------------------
def summarize_content(retriever):
    summary_template = """
    You are an expert teaching assistant.
    Based on the following context, summarize it in simple, clear English.

    Context:
    {context}
    """
    prompt = PromptTemplate(template=summary_template, input_variables=["context"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain.run("Summarize the document")

# ------------------- Question Generation -------------------
def generate_questions(retriever):
    question_template = """
    You are an expert educator.
    Using the following context, create:
    - 5 Multiple Choice Questions (MCQs) with 4 options each and correct answers.
    - 5 open-ended or essay questions.

    Context:
    {context}
    """
    prompt = PromptTemplate(template=question_template, input_variables=["context"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa_chain.run("Generate questions")

# ------------------- UI Workflow -------------------
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        retriever, docs = process_pdf(uploaded_file)

    if st.button("🧠 Summarize Document"):
        with st.spinner("Generating summary..."):
            summary = summarize_content(retriever)
            st.subheader("📘 Summary:")
            st.write(summary)

    if st.button("❓ Generate Questions"):
        with st.spinner("Generating questions..."):
            questions = generate_questions(retriever)
            st.subheader("📝 Questions:")
            st.write(questions)
# ------------------- Chatbot -------------------
def chat_with_pdf(retriever, user_input):
    template = """
    You are a helpful assistant.

    Use ONLY the document context to answer.

    Context:
    {context}

    Question:
    {question}

    If the answer is not in the document, say:
    "I don't know from the document."
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain.run(user_input)


# ------------------- UI Workflow -------------------
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if uploaded_file:
    retriever, docs = process_pdf(uploaded_file)

    st.success("PDF processed successfully!")

    tab1, tab2, tab3, tab4 = st.tabs(["📘 Summary", "📝 Questions", "💬 Chatbot", "Quiz"])

    # --------- TAB 1: Summary ---------
    with tab1:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                output = summarize_content(retriever)
                st.subheader("📘 Summary:")
                st.write(output)

    # --------- TAB 2: Questions ---------
    with tab2:
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                output = generate_questions(retriever)
                st.subheader("📝 Questions:")
                st.write(output)

    # --------- TAB 3: Chatbot ---------
    with tab3:
        st.subheader("💬 Ask anything from the PDF")
        user_query = st.text_input("Your question:")
        if st.button("Ask"):
            if user_query.strip():
                with st.spinner("Searching document..."):
                    output = chat_with_pdf(retriever, user_query)
                st.write("📌 **Answer:**")
                st.write(output)
    
    with tab4:
        st.header("🧠 Quiz Mode")

        if "quiz_active" not in st.session_state:
            if st.button("Start Quiz"):
                start_quiz(num_questions=5)

        if st.session_state.get("quiz_active"):
            if st.session_state.q_index < len(st.session_state.quiz):
                show_question()
            else:
                show_results()
                st.session_state.quiz_active = False
else:
    st.info("👆 Upload a PDF to start.")


# In[30]:





# In[ ]:




