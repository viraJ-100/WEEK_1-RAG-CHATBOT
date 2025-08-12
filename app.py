import streamlit as st
from groq import Groq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import json
load_dotenv()

embedding_model = OllamaEmbeddings(model="nomic-embed-text") 


def clear_db():
    Chroma(
        collection_name="study",
        persist_directory="./study_db",
        embedding_function=embedding_model
    ).delete_collection()


# --- Page Config ---
st.set_page_config(page_title="RAG STUDY BOT", page_icon="./images/icon.png", layout="wide")

# Default page setting
if "page" not in st.session_state:
    st.session_state.page = "chat"

# --- Global Styles (make sidebar buttons same size) ---
st.markdown(
    """
    <style>
    /* Make all sidebar buttons equal width and height */
    [data-testid="stSidebar"] .stButton { width: 100% !important; }
    [data-testid="stSidebar"] .stButton button {
        width: 100% !important;
        height: 48px !important;
        display: block !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Sidebar ---
# Button at the top
with st.sidebar:
    
    if st.button("✏️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.page = "chat"

    
    if st.button("Quiz Time", use_container_width=True):
        st.session_state.page = "quiz"

    
    if st.button("Summarization", use_container_width=True):
        st.session_state.page = "summarization"

    if st.button("Clear DB", use_container_width=True):
        st.session_state.messages = []
        vectorstore = None
        retriever = None
        clear_db()
        if retriever is None:
            st.warning("⚠️ No files to clear.")
        else:
            st.success("✅ Database has been cleared!", icon="🗑️")


# Upload section
uploaded_files = st.sidebar.file_uploader(
    "Drag and drop your PDFs and txts",
    type=None,
    accept_multiple_files=True,
    label_visibility = "collapsed"
)


# --- Title ---
st.markdown("<h1 style='font-size: 60px;'>DocMind</h1>", unsafe_allow_html=True)

# --- API Key ---
groq_api_key = os.getenv("groq_api_key")


#--------------------------------------------------------RETRIVAL---------------------------------------------------------------
# --- Initialize Vectorstore ---
embedding_model = OllamaEmbeddings(model="nomic-embed-text") 
vectorstore = None

# Load existing DB if available
if os.path.exists("./study_db") and os.listdir("./study_db"):
    vectorstore = Chroma(
        collection_name="study",
        persist_directory="./study_db",
        embedding_function=embedding_model
    )

def calculate_chunk_ids(chunks):
    
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Increment the chunk index if it's the same page
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Assign a unique ID to each chunk
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks):
    # Load the existing vectorstore
    db = Chroma(
        collection_name="study",
        persist_directory="./study_db",
        embedding_function=embedding_model
    )

    # Assign unique IDs to chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    # print(f"Number of existing docs: {len(existing_ids)}")

    # Filter only new chunks
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        # print(f"Adding {len(new_chunks)} new documents...")
        new_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        db.persist()
    else:
        print(".")


# If new files uploaded, process them and add to DB
if groq_api_key and uploaded_files:
    documents = []

    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            # Handle PDF files
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file.name, "page": page_num}
                    ))

        elif file.name.lower().endswith(".txt"):
            # Handle TXT files
            text = file.read().decode("utf-8")  # read & decode
            documents.append(Document(
                page_content=text,
                metadata={"source": file.name, "page": 1}
            ))

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
        
    chunks = text_splitter.split_documents(documents)

    if vectorstore:
        # vectorstore.add_documents(chunks)
        add_to_chroma(chunks)
    else:
        chunks_with_ids = calculate_chunk_ids(chunks)
        ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        vectorstore = Chroma.from_documents(
            chunks_with_ids,
            embedding_model,
            ids=ids,
            collection_name="study",
            persist_directory="./study_db"
        )

    vectorstore.persist()

## Create retriever if DB exists
retriever = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.page == "chat":
    # --- Display Chat History ---
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # --- User Input ---
    user_query = st.chat_input("Your question about the PDFs...")
        
    if user_query:
        if retriever is None:
            st.warning("⚠️ Please upload files first before asking questions.")
        else:
            # Show user message
            st.chat_message("user").markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.spinner("Thinking..."):
                docs = retriever.get_relevant_documents(user_query)
                context = "\n\n".join([doc.page_content for doc in docs])

                llm = Groq(api_key=groq_api_key)
                #-----------------------------------------------------------------AUGMENTED-------------------------------------------------------
                system_message = """
                You are an assistant who answers user queries based on business process manuals, training guides, and procedural documentation.
                The user input will include the context you need to answer the question.

                The context will begin with the token: ###Context.
                This context contains references to specific sections of one or more documents relevant to the query.

                The user question will begin with the token: ###Question.

                Instructions:
                1. Answer only using the information provided in the context.
                2. Do not mention or refer to the context in your answer.
                3. If the answer cannot be found in the context, respond exactly with: "I don't know".
                4. Keep answers clear, concise, and aligned with the terminology and process descriptions typically found in business manuals and procedural guides.
                """

                prompt = f"###Context:\n{context}\n\n###Question: {user_query}"

                # Clear existing content and write the new prompt
                with open("prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)

                model_name = os.getenv("model_name")

                response = llm.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                #-----------------------------------------------------------------GENERATION-------------------------------------------------------
                ai_reply = response.choices[0].message.content

                # --- 📌 ADD COMPACT CITATIONS ---
                citations_dict = {}
                for doc in docs:
                    src = doc.metadata.get("source", "Unknown Source")
                    page = doc.metadata.get("page", "N/A")
                    if src not in citations_dict:
                        citations_dict[src] = set()
                    citations_dict[src].add(str(page))

                citations_str = " | ".join([
                    f"{src} : {','.join(sorted(pages, key=lambda x: int(x) if x.isdigit() else 999))}"
                    for src, pages in citations_dict.items()
                ])

                if citations_str:
                    ai_reply += f"\n\n[Sources: {citations_str}]"

            # Show assistant message
            st.chat_message("assistant").markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    if not st.session_state.messages:
        st.markdown(
            "<h1 style='font-size: 20px;'>Ask. Explore. Understand your documents!!!</h1>",
            unsafe_allow_html=True
        )


elif st.session_state.page == "quiz":
    st.title("📝 Quiz Time")

    topic = st.text_input("Enter a topic")
    num_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    if st.button("Generate Quiz"):
        if not topic.strip():
            st.error("⚠️ Please enter a topic before generating the quiz.")
            st.stop()
        if vectorstore is None:
            st.warning("Vectorstore is not initialized. Please upload and process documents first.")
            st.stop()

        with st.spinner("Generating quiz..."):
            llm = Groq(api_key=groq_api_key)

            # Retrieve relevant chunks using semantic search
            retrieval_k = min(50, max(5, int(num_questions) * 2))
            relevant_docs = vectorstore.similarity_search(topic, k=retrieval_k)
            if not relevant_docs:
                st.error(f"❌ No relevant content found for '{topic}'.")
                st.stop()
            db_content = " ".join([doc.page_content for doc in relevant_docs])

            # Build quiz prompt (request strict JSON)
            quiz_prompt = f"""
            You are to create a multiple-choice quiz strictly from the provided content. Do NOT use any outside knowledge.

            Content:
            {db_content}

            Requirements:
            - Generate exactly {int(num_questions)} questions.
            - Each question must have exactly 4 options.
            - Provide the correct option letter (A, B, C, or D).
            - Provide a brief explanation of why the correct answer is correct based only on the content.

            Output:
            - Return ONLY a strict JSON array (no backticks, no markdown) of objects with this schema:
              [
                {{
                  "question": string,
                  "options": [string, string, string, string],
                  "correct": "A" | "B" | "C" | "D",
                  "explanation": string
                }}
              ]
            """

            # Call LLM
            response = llm.chat.completions.create(
                model=os.getenv("model_name"),
                messages=[{"role": "user", "content": quiz_prompt}],
                temperature=0.7
            )

            raw_output = response.choices[0].message.content.strip()

            # Try parse JSON directly; fallback to extracting JSON array
            quiz_items = None
            try:
                quiz_items = json.loads(raw_output)
            except Exception:
                try:
                    start = raw_output.find("[")
                    end = raw_output.rfind("]")
                    if start != -1 and end != -1 and end > start:
                        quiz_items = json.loads(raw_output[start:end+1])
                except Exception:
                    quiz_items = None

        if not isinstance(quiz_items, list):
            st.error("Failed to parse quiz. Please try again.")
        else:
            st.markdown("### Your Quiz")
            option_labels = ["A", "B", "C", "D"]
            for idx, item in enumerate(quiz_items, start=1):
                question = item.get("question", "").strip()
                options = item.get("options", [])
                correct = item.get("correct", "").strip()
                explanation = item.get("explanation", "").strip()

                st.markdown(f"**Q{idx}. {question}**")
                # Ensure exactly 4 options
                if not isinstance(options, list) or len(options) != 4:
                    st.warning("This question did not return 4 options.")
                for i, opt in enumerate(options[:4]):
                    label = option_labels[i] if i < 4 else ""
                    st.markdown(f"{label}) {opt}")

                with st.expander("Show answer and explanation"):
                    st.markdown(f"**Correct:** {correct}")
                    if explanation:
                        st.write(explanation)


elif st.session_state.page == "summarization":
    st.title("📄 Summarization")

    if vectorstore is None:
        st.warning("Vectorstore is not initialized. Please upload and process documents first.")
    else:
        # Build list of available PDF sources from the DB
        data_for_sources = vectorstore.get(include=["metadatas"]) or {}
        metadatas = data_for_sources.get("metadatas", []) or []
        all_sources = [m.get("source") for m in metadatas if isinstance(m, dict) and m.get("source")]
        pdf_sources = sorted({s for s in all_sources if str(s).lower().endswith(".pdf")})

        if not pdf_sources:
            st.warning("No PDFs found in the database. Please upload PDF files first.")
        else:
            col1, col2 = st.columns([2, 1])
            selected_pdf = col1.selectbox("Choose a PDF", options=pdf_sources, index=0)
            target_words = col2.number_input(
                "Number of words",
                min_value=50,
                max_value=2000,
                value=200,
                step=50,
            )

            if st.button("Generate Summary"):
                with st.spinner(f"Summarizing '{selected_pdf}'..."):
                    # Pull only the selected PDF's documents from the DB
                    docs_data = vectorstore.get(where={"source": selected_pdf}, include=["documents"]) or {}
                    db_content_list = docs_data.get("documents") if docs_data else None

                    if db_content_list:
                        llm = Groq(api_key=groq_api_key)

                        # --- Token-based chunking ---
                        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            chunk_size=2000,  # tokens
                            chunk_overlap=200,
                        )
                        chunks = text_splitter.split_text(" ".join(db_content_list))

                        summaries = []
                        for idx, chunk in enumerate(chunks, 1):
                            summarize_prompt = f"""
                            You are a summarization assistant. Summarize ONLY the provided chunk of text from the document '{selected_pdf}'.
                            Do NOT use outside information.

                            Keep key facts, numbers, and terminology intact.

                            Text chunk ({idx}/{len(chunks)}):
                            {chunk}
                            """
                            response = llm.chat.completions.create(
                                model=os.getenv("model_name"),
                                messages=[{"role": "user", "content": summarize_prompt}],
                                temperature=0.2,
                            )
                            summaries.append(response.choices[0].message.content)

                        # --- Combine chunk summaries with word target ---
                        final_prompt = f"""
                        Combine the following chunk summaries into a single cohesive summary for the document '{selected_pdf}'.
                        Write approximately {int(target_words)} words. Be concise but preserve important facts and structure.

                        Chunk summaries:
                        {" ".join(summaries)}
                        """
                        final_response = llm.chat.completions.create(
                            model=os.getenv("model_name"),
                            messages=[{"role": "user", "content": final_prompt}],
                            temperature=0.2,
                        )
                        final_summary = final_response.choices[0].message.content

                        st.subheader("📌 Summary")
                        st.write(final_summary)
                    else:
                        st.warning("No content found in the database for the selected PDF.")
