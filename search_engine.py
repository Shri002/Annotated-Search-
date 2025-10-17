import os
import math
import re
import pytest

# ----------------------------
# Utility: clean()
# ----------------------------
def clean(token: str, pattern: re.Pattern[str] = re.compile(r"\W+")) -> str:
    """
    Returns all the characters in the token lowercased and without matches to the given pattern.
    Example:
    >>> clean("Hello!")
    'hello'
    """
    return pattern.sub("", token.lower())


# ----------------------------
# Class: Document
# ----------------------------
class Document:
    """
    Represents a text file as a document object and computes term frequency for each word.
    """

    def __init__(self, path: str) -> None:
        """Initializes a new Document object from the given path."""
        self.path = path
        self.term_freq: dict[str, float] = {}
        self.words: list[str] = []

        with open(self.path, encoding="utf-8") as f:
            content = f.read()
            self.words = content.split()

        for word in self.words:
            clean_word = clean(word)
            if clean_word:
                self.term_freq[clean_word] = self.term_freq.get(clean_word, 0) + 1

        # Normalize frequencies
        total_words = len(self.words) if self.words else 1
        for key in self.term_freq:
            self.term_freq[key] /= total_words

    def term_frequency(self, term: str) -> float:
        """Return the term frequency of a word, or 0 if not found."""
        term = clean(term)
        return self.term_freq.get(term, 0.0)

    def get_path(self) -> str:
        """Return the document's file path."""
        return self.path

    def get_words(self) -> set[str]:
        """Return a set of unique cleaned words in the document."""
        return set(self.term_freq)

    def __repr__(self) -> str:
        """Return a string representation of this document."""
        return f"Document('{self.path}')"


# ----------------------------
# Class: SearchEngine
# ----------------------------
class SearchEngine:
    """
    Represents a corpus of Document objects.
    Computes tf–idf statistics and searches for matching documents.
    """

    def __init__(self, path: str, extension: str = ".txt") -> None:
        """Construct an inverted index from the files in the given directory."""
        self.path = path
        self.count = 0
        self.all_index: dict[str, list[Document]] = {}

        for filename in os.listdir(path):
            if filename.endswith(extension):
                full_path = os.path.join(path, filename)
                new_doc = Document(full_path)
                self.count += 1
                for word in new_doc.get_words():
                    self.all_index.setdefault(word, []).append(new_doc)

    def _calculate_idf(self, term: str) -> float:
        """Return inverse document frequency (idf) for a term."""
        if term in self.all_index:
            return math.log(self.count / len(self.all_index[term]))
        return 0.0

    def search(self, query: str) -> list[str]:
        """Return document paths matching the query, sorted by tf–idf score."""
        big_list = {}
        query_terms = [clean(word) for word in query.split()]
        matching_docs = set()

        for term in query_terms:
            if term in self.all_index:
                matching_docs.update(self.all_index[term])

        for doc in matching_docs:
            total_score = sum(
                self._calculate_idf(term) * doc.term_frequency(term)
                for term in query_terms
            )
            big_list[doc] = total_score

        # Sort by descending tf–idf score
        sorted_docs = sorted(big_list.items(), key=lambda x: x[1], reverse=True)
        return [doc.get_path() for doc, _ in sorted_docs]

    def __repr__(self) -> str:
        """Return a string representation of the search engine."""
        return f"SearchEngine('{self.path}')"


# ----------------------------
# Pytest Unit Tests
# ----------------------------
class TestDocument:
    @pytest.fixture(scope="class")
    def sample_docs(self, tmp_path_factory):
        # Create temporary test folder
        folder = tmp_path_factory.mktemp("docs")
        d1 = folder / "doc1.txt"
        d2 = folder / "doc2.txt"
        d3 = folder / "doc3.txt"

        d1.write_text("dogs are the greatest pets")
        d2.write_text("i believe in manicures")
        d3.write_text("dont judge a polish by its first coat")

        return {
            "doc1": Document(str(d1)),
            "doc2": Document(str(d2)),
            "doc3": Document(str(d3)),
        }

    def test_term_frequency(self, sample_docs):
        doc1, doc2, doc3 = sample_docs["doc1"], sample_docs["doc2"], sample_docs["doc3"]
        assert doc1.term_frequency("dogs") == pytest.approx(1 / 5)
        assert doc2.term_frequency("manicures") == pytest.approx(1 / 4)
        assert doc3.term_frequency("coat") == pytest.approx(1 / 7)

    def test_get_words(self, sample_docs):
        doc1, doc2, doc3 = sample_docs["doc1"], sample_docs["doc2"], sample_docs["doc3"]
        assert doc1.get_words() == {"dogs", "are", "the", "greatest", "pets"}
        assert "manicures" in doc2.get_words()
        assert "coat" in doc3.get_words()

    def test_repr(self, sample_docs):
        doc1 = sample_docs["doc1"]
        assert repr(doc1).startswith("Document(")
        assert "doc1.txt" in repr(doc1)


class TestSearchEngine:
    @pytest.fixture(scope="class")
    def search_setup(self, tmp_path_factory):
        folder = tmp_path_factory.mktemp("doggos")
        (folder / "doc1.txt").write_text("dogs are the greatest pets")
        (folder / "doc2.txt").write_text("cats seem pretty okay")
        (folder / "doc3.txt").write_text("i love dogs")
        return SearchEngine(str(folder))

    def test_calculate_idf(self, search_setup):
        engine = search_setup
        assert engine._calculate_idf("dogs") > 0
        assert engine._calculate_idf("nonexistent") == 0

    def test_search_single_term(self, search_setup):
        results = search_setup.search("dogs")
        assert any("doc1.txt" in r for r in results)
        assert any("doc3.txt" in r for r in results)

    def test_search_multi_term(self, search_setup):
        results = search_setup.search("love dogs")
        # doc3 should come before doc1 (higher tf–idf)
        assert results[0].endswith("doc3.txt")
        assert results[1].endswith("doc1.txt")


# ----------------------------
# Automatic Demo Corpus Setup
# ----------------------------
def setup_demo_folder(folder_name="doggos"):
    """Create a sample folder with example text files if it doesn’t exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        with open(os.path.join(folder_name, "doc1.txt"), "w") as f:
            f.write("dogs are the greatest pets")
        with open(os.path.join(folder_name, "doc2.txt"), "w") as f:
            f.write("cats seem pretty okay")
        with open(os.path.join(folder_name, "doc3.txt"), "w") as f:
            f.write("i love dogs")
        print(f"✅ Created demo folder '{folder_name}' with example files.")
    else:
        print(f"📁 Folder '{folder_name}' already exists — skipping creation.")


# ----------------------------
# Example Run
# ----------------------------
if __name__ == "__main__":
    setup_demo_folder()
    engine = SearchEngine("doggos")
    results = engine.search("love dogs")
    print("\n🔍 Search results for 'love dogs':")
    for i, path in enumerate(results, 1):
        print(f"{i}. {path}")
