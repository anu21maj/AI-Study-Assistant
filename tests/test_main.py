from app.main import load_study_notes
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def test_load_study_notes():
    notes = load_study_notes()

    assert len(notes) > 0
    assert all(note.strip() for note in notes)


def test_study_notes_contain_machine_learning():
    notes = load_study_notes()

    assert any(
        "machine learning" in note.lower()
        for note in notes
    )


def test_semantic_search_finds_relevant_note():
    notes = load_study_notes()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    note_embeddings = model.encode(notes)
    question_embedding = model.encode(
        ["What is machine learning?"]
    )

    similarities = cosine_similarity(
        question_embedding,
        note_embeddings
    )[0]

    best_match = similarities.argmax()
    best_note = notes[best_match]

    assert "machine learning" in best_note.lower()

