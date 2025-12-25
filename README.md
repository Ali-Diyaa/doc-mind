#  Doc Mind – Your Study Partner

**Doc Mind** is an AI-powered RAG (Retrieval-Augmented Generation) application that helps students interact with their PDF study materials. It provides **summarization, question generation, interactive quizzes**, and a **document-based chatbot**, all powered by large language models.

---

## Features

### PDF Upload & Processing
- Upload your PDF files.
- Split documents into chunks for efficient semantic search.
- Build a local **FAISS vector store** for fast retrieval.

### Document Summarization
- Generate concise summaries of uploaded PDFs.
- Summaries are 2–3 lines, covering all key points.

### Question Generation
- Generate **multiple-choice questions (MCQs)** with 4 options (A–D).
- Generate **essay/open-ended questions** to test comprehension.
- Correct answers are clearly provided for MCQs.

### Interactive Quiz Mode
- Test your knowledge using automatically generated MCQs.
- Track your score in real-time.
- Provides feedback and shows correct answers.

### Chatbot
- Ask questions about the PDF content.
- Answers are retrieved **only from the uploaded document**.
- Responds with `"I don't know from the document."` if the answer is not available.

---

## Technologies Used
- **Python 3.10+**
- **Streamlit** – Interactive web interface
- **LangChain** – LLM chaining, prompts, and RAG pipelines
- **Ollama LLM** – Large language model backend
- **HuggingFace Sentence Transformers** – For embeddings
- **FAISS** – Vector search for document retrieval
- **dotenv** – Load environment variables

---

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/doc-mind.git
cd doc-mind
