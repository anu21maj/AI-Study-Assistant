from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Locate the project directory and study notes
BASE_DIR = Path(__file__).resolve().parent.parent
NOTES_FILE = BASE_DIR / "data" / "study_notes.txt"

SIMILARITY_THRESHOLD = 0.35


@st.cache_data
def load_study_notes():
    """Load study notes from a text file."""

    with open(NOTES_FILE, "r", encoding="utf-8") as file:
        content = file.read()

    notes = [
        note.strip()
        for note in content.split("\n\n")
        if note.strip()
    ]

    return notes


@st.cache_resource
def load_model():
    """Load the sentence embedding model once."""

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def create_embeddings(notes):
    """Convert study notes into embeddings."""

    model = load_model()
    return model.encode(list(notes))


# Page configuration
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered",
)

# Application heading
st.title("📚 AI Study Assistant")

st.write(
    "Ask questions about artificial intelligence and machine learning. "
    "I'll search your study notes for the most relevant information."
)

# Load study notes
try:
    notes = load_study_notes()

    if not notes:
        st.warning(
            "No study notes found. Please add notes to "
            "data/study_notes.txt."
        )
        st.stop()

    model = load_model()
    note_embeddings = create_embeddings(tuple(notes))

except Exception as error:
    st.error(f"Could not load the study assistant: {error}")
    st.stop()

st.success(f"Knowledge base loaded: {len(notes)} study notes.")

# Question input
question = st.text_input(
    "Enter your question",
    placeholder="Example: What is machine learning?",
)

# Search button
if st.button("Ask Question", type="primary"):

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        with st.spinner("Searching your study notes..."):

            # Convert the question into an embedding
            question_embedding = model.encode([question.strip()])

            # Calculate similarity scores
            similarities = cosine_similarity(
                question_embedding,
                note_embeddings,
            )[0]

            # Get the top three matching notes
            top_matches = similarities.argsort()[::-1][:3]

            # Keep only notes meeting the threshold
            relevant_matches = [
                (index, float(similarities[index]))
                for index in top_matches
                if similarities[index] >= SIMILARITY_THRESHOLD
            ]

        if not relevant_matches:
            st.warning(
                "Sorry, I couldn't find relevant information "
                "in your study notes."
            )

        else:
            st.subheader("🔎 Matching Study Notes")

            for rank, (index, score) in enumerate(
                relevant_matches, start=1
            ):
                with st.container(border=True):
                    st.markdown(f"**Result {rank}**")
                    st.write(notes[index])
                    st.caption(f"Similarity score: {score:.4f}")

st.divider()

st.caption(
    "Powered by Sentence Transformers and cosine similarity. "
    "Results are retrieved from your existing study notes."
)

