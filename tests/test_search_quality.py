from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.main import load_study_notes

def test_semantic_search_retrieves_expected_topics():
    notes = load_study_notes()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    note_embeddings = model.encode(notes)

    test_questions = [
        ("What is machine learning?", "machine learning"),
        ("Explain precision and recall.", "precision"),
        ("What is overfitting?", "overfitting"),
        ("How does a random forest work?", "random forest"),
        ("What is data leakage?", "data leakage"),
        ("Explain stemming and lemmatization.", "lemmatization"),
        ("What is semantic search?", "semantic search"),
        ("What is object detection?", "object detection"),
        ("What does MAE measure?", "mean absolute error"),
        (
            "What is a convolutional neural network?",
            "convolutional neural network",
        ),
    ]

    for question, expected_topic in test_questions:
        question_embedding = model.encode([question])

        similarities = cosine_similarity(
            question_embedding,
            note_embeddings,
        )[0]

        top_indices = similarities.argsort()[::-1][:3]

        top_notes = [
            notes[index].lower()
            for index in top_indices
        ]

        assert any(
            expected_topic in note
            for note in top_notes
        ), (
            f"Expected topic '{expected_topic}' was not retrieved "
            f"for question '{question}'."
        )

