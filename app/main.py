from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the study notes file
NOTES_FILE = BASE_DIR / "data" / "study_notes.txt"

# Minimum similarity score for displaying a note
SIMILARITY_THRESHOLD = 0.35


def load_study_notes():
    """Load study notes from a text file."""

    with open(NOTES_FILE, "r", encoding="utf-8") as file:
        content = file.read()

    # Each non-empty paragraph becomes a separate study note
    notes = [
        note.strip()
        for note in content.split("\n\n")
        if note.strip()
    ]

    return notes


def main():
    # Load notes from the file
    documents = load_study_notes()

    if not documents:
        print("No study notes found. Please add notes to study_notes.txt.")
        return

    print(f"Loaded {len(documents)} study notes.")

    # Load the sentence embedding model
    print("Loading the semantic search model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert each study note into an embedding
    document_embeddings = model.encode(documents)

    print("\nAI Study Assistant is ready!")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk a question: ").strip()

        # Exit the application
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Handle empty input
        if not question:
            print("Please enter a question.")
            continue

        # Convert the question into an embedding
        question_embedding = model.encode([question])

        # Compare the question with all study notes
        similarities = cosine_similarity(
            question_embedding,
            document_embeddings
        )[0]

        # Find the most relevant study note
        best_match = similarities.argmax()

        # Get the similarity score of the best match
        best_score = similarities[best_match]

        # Check whether the best match is relevant enough
        if best_score < SIMILARITY_THRESHOLD:
            print(
                "\nSorry, I couldn't find relevant information "
                "in your study notes."
            )
            continue

        # Find the indices of the top 3 most relevant notes
        top_matches = similarities.argsort()[::-1][:3]

        print("\nTop matching study notes:")

        rank = 1

        for index in top_matches:
            score = similarities[index]

            # Display only notes that meet the threshold
            if score >= SIMILARITY_THRESHOLD:
                print(f"\n{rank}. Similarity score: {score:.4f}")
                print(documents[index])
                rank += 1

        # Handle the case where no notes pass the threshold
        if rank == 1:
            print("No sufficiently relevant study notes were found.")


if __name__ == "__main__":
    main()

