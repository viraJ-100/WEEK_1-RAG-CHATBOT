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


# --- Sidebar ---
# Button at the top
with st.sidebar:
    
    if st.button("✏️ New Conversation"):
        st.session_state.messages = []
        st.session_state.page = "chat"

    
    if st.button("Quiz Time"):
        st.session_state.page = "quiz"

    
    if st.button("Summarization"):
        st.session_state.page = "summarization"

    if st.button("Clear DB"):
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

    if st.button("Generate Quiz"):
        if not topic.strip():
            st.error("⚠️ Please enter a topic before generating the quiz.")
            st.stop()

        with st.spinner("Generating quiz..."):
            llm = Groq(api_key=groq_api_key)

            # Retrieve relevant chunks using semantic search
            relevant_docs = vectorstore.similarity_search(topic, k=5)  # Adjust k as needed
            if not relevant_docs:
                st.error(f"❌ No relevant content found for '{topic}'.")
                st.stop()
            db_content = " ".join([doc.page_content for doc in relevant_docs])

            # Build quiz prompt
            quiz_prompt = f"""
            Create a multiple-choice quiz based strictly on the following document content:
            {db_content}

            Do NOT use any outside knowledge.

            Output format:
            Q: <question>
            A) <option1>
            B) <option2>
            C) <option3>
            D) <option4>
            Correct Answer: <letter>
            Generate 10 questions.
            """

            # Call LLM
            response = llm.chat.completions.create(
                model=os.getenv("model_name"),
                messages=[{"role": "user", "content": quiz_prompt}],
                temperature=0.7
            )

            quiz_text = response.choices[0].message.content

        st.markdown("### Your Quiz")
        st.text_area("Generated Quiz", quiz_text, height=400)


elif st.session_state.page == "summarization":
    st.title("📄 Summarization")

    if st.button("Generate Summary"):
        if vectorstore:
            with st.spinner("Summarizing the document database..."):
                # Pull all documents from the DB
                docs_data = vectorstore.get(include=["documents"])
                db_content_list = docs_data["documents"] if docs_data and docs_data["documents"] else None

                if db_content_list:
                    llm = Groq(api_key=groq_api_key)

                    # --- Token-based chunking ---
                    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        chunk_size=2000,  # tokens
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_text(" ".join(db_content_list))

                    summaries = []
                    for idx, chunk in enumerate(chunks, 1):
                        summarize_prompt = f"""
                        You are a summarization assistant. Summarize ONLY the provided chunk of text.
                        Do NOT use outside information.

                        Formatting guidelines:
                        - Organize by **topic** where possible.
                        - Include short paragraphs and bullet points.
                        - Keep key facts, numbers, and terminology intact.

                        Text chunk ({idx}/{len(chunks)}):
                        {chunk}
                        """
                        response = llm.chat.completions.create(
                            model=os.getenv("model_name"),
                            messages=[{"role": "user", "content": summarize_prompt}],
                            temperature=0.2
                        )
                        summaries.append(response.choices[0].message.content)

                    # --- Combine chunk summaries ---
                    final_prompt = f"""
                    Combine the following chunk summaries into a single cohesive, topic-wise summary.
                    Keep it concise but detailed. Preserve important facts and structure.

                    Chunk summaries:
                    {" ".join(summaries)}
                    """
                    final_response = llm.chat.completions.create(
                        model=os.getenv("model_name"),
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=0.2
                    )
                    final_summary = final_response.choices[0].message.content

                    st.subheader("📌 Topic-wise Summary")
                    st.write(final_summary)
                else:
                    st.warning("No documents found in the database.")
        else:
            st.warning("Vectorstore is not initialized. Please upload and process documents first.")
