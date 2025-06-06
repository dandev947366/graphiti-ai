import re
import nltk
from typing import List, Optional, Generator, Union
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


@dataclass
class ChunkingConfig:
    max_tokens: int = 512
    min_tokens: int = 50
    stride: int = 50
    language: str = "english"
    remove_stopwords: bool = False
    clean_special_chars: bool = True
    header_patterns: List[str] = None
    footer_patterns: List[str] = None
    special_chars_pattern: str = r"[^A-Za-z0-9\s\.,;:\'\"\?\!\-]"


class EmbeddingPreprocessor:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config if config else ChunkingConfig()
        self.stopwords = (
            set(nltk.corpus.stopwords.words(self.config.language))
            if self.config.remove_stopwords
            else None
        )

        if self.config.header_patterns is None:
            self.config.header_patterns = [
                r"^\s*Copyright.*$",
                r"^\s*Confidential.*$",
                r"^\s*Page\s*\d+.*$",
                r"^\s*©.*$",
            ]
        if self.config.footer_patterns is None:
            self.config.footer_patterns = [
                r"^\s*Page\s*\d+.*$",
                r"^\s*©.*$",
                r"^\s*http[s]?://.*$",
                r"^\s*All rights reserved.*$",
            ]

    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning in one pass"""
        # Remove headers and footers
        for pattern in self.config.header_patterns + self.config.footer_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove special characters (keeping basic punctuation)
        if self.config.clean_special_chars:
            text = re.sub(self.config.special_chars_pattern, "", text)

        # Normalize whitespace and formatting
        text = re.sub(r"\s+", " ", text)  # Convert all whitespace to single spaces
        text = re.sub(r"\s([.,;:?!])", r"\1", text)  # Remove space before punctuation
        text = re.sub(r"([.,;:?!])\s+", r"\1 ", text)  # Single space after punctuation
        text = re.sub(r"\.{2,}", ".", text)  # Fix multiple periods
        text = re.sub(r"\s*-\s*", "-", text)  # Fix hyphen spacing

        # Remove leading/trailing spaces and line breaks
        text = text.strip()

        # Remove stopwords if enabled
        if self.config.remove_stopwords and self.stopwords:
            words = word_tokenize(text)
            words = [w for w in words if w.lower() not in self.stopwords]
            text = " ".join(words)

        return text

    def chunk_text(self, text: str) -> Generator[str, None, None]:
        """Generate clean, optimized chunks"""
        text = self.clean_text(text)
        sentences = sent_tokenize(text, language=self.config.language)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            token_count = len(tokens)

            if token_count > self.config.max_tokens:
                if current_chunk:
                    yield " ".join(current_chunk)
                    current_chunk = []
                    current_length = 0

                # Split oversized sentences
                for i in range(
                    0, token_count, self.config.max_tokens - self.config.stride
                ):
                    chunk = " ".join(tokens[i : i + self.config.max_tokens])
                    yield chunk
                continue

            if current_length + token_count <= self.config.max_tokens:
                current_chunk.append(sentence)
                current_length += token_count
            else:
                if current_chunk:
                    yield " ".join(current_chunk)
                current_chunk = [sentence]
                current_length = token_count

        if current_chunk and current_length >= self.config.min_tokens:
            yield " ".join(current_chunk)


def main():
    sample_text = """
    COPYRIGHT 2023. CONFIDENTIAL DOCUMENT.
    
    In 1939, Robert Oppenheimer and others predicted that neutron stars above another limit, 
    the Tolman-Oppenheimer-Volkoff limit, would collapse further for the reasons presented by Chandrasekhar, 
    and concluded that no law of physics was likely to intervene and stop at least some stars from collapsing to black holes.
    
    Their original calculations, based on the Pauli exclusion principle, gave it as 0.7 M☉. 
    Subsequent consideration of neutron-neutron repulsion mediated by the strong force raised the estimate to approximately 1.5 M☉ to 3.0 M☉.
    
    Page 2 of 3
    Observations of the neutron star merger GW170817, which is thought to have generated a black hole shortly afterward, 
    have refined the TOV limit estimate to ~2.17 M☉.
    
    © All rights reserved.
    """

    config = ChunkingConfig(
        max_tokens=100,
        stride=20,
        clean_special_chars=True,
        header_patterns=[r"Copyright.*", r"Confidential.*", r"Page\s*\d+.*"],
        footer_patterns=[r"Page\s*\d+.*", r"©.*", r"All rights reserved.*"],
    )

    processor = EmbeddingPreprocessor(config)

    print("Cleaned and Chunked Text:\n" + "=" * 50)
    for chunk in processor.chunk_text(sample_text):
        print(chunk)
        print("-" * 50)


if __name__ == "__main__":
    main()
