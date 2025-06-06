import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Union, List, Literal, Dict, Optional
import logging

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)  # For newer NLTK versions
except Exception as e:
    logging.warning(f"NLTK download warning: {e}")

# Define document type configurations
DOCUMENT_CONFIGS: Dict[str, Dict] = {
    "full": {
        "special_chars_pattern": r"[^\w\s.,;:?!\-']",
        "citation_pattern": r"\[[\w\d\s,-]+\]|\([\w\s.,-]+(?:\d{4})\)",
        "preserve_patterns": [],
    },
    "scientific": {
        "special_chars_pattern": r"[^\w\s.,;:?!\-☉~=<>≤≥±→←↑↓°µ]",
        "citation_pattern": r"\[[\w\d\s,-]+\]|\([\w\s.,-]+(?:\d{4})\)",
        "preserve_patterns": [
            r"\d+\.\d+",  # Decimal numbers
            r"[A-Z][a-z]+\d*",  # Chemical formulas (like H2O, NaCl)
            r"[\d]+°[CF]?",  # Temperature/angles like 45°C
            r"\d+%",  # Percentages like 15%
        ],
    },
    "social": {
        "special_chars_pattern": r"[^\w\s#@.,;:?!\-']",
        "citation_pattern": r"",
        "preserve_patterns": [
            r"#[a-zA-Z0-9_]+",  # Hashtags
            r"@[a-zA-Z0-9_]+",  # Mentions
        ],
    },
    "legal": {
        "special_chars_pattern": r"[^\w\s.,;:?!\-§¶']",
        "citation_pattern": r"\((?:\d{1,4}\s\w+\s\d{1,4})\)|\[\d+\]",
        "preserve_patterns": [
            r"§\d+",  # Section references like §123
            r"\d+\sU\.S\.C\.",  # US Code citations
            r"\d+\sCFR",  # Code of Federal Regulations
        ],
    },
    "technical": {
        "special_chars_pattern": r"[^\w\s.,;:?!\-_=<>{}[\]()°]",
        "citation_pattern": r"\[[\w-]+\]",
        "preserve_patterns": [
            r"[a-z0-9_]+\(\)",  # Function calls like func()
            r"\{[^}]+\}",  # Code blocks like {some_code}
            r"\[[^\]]+\]",  # Arrays/references like [1, 2, 3]
            r"\d+°",  # Angles like 90°
            r"[A-Z]{2,}",  # Acronyms like API, CPU
        ],
    },
    "medical": {
        "special_chars_pattern": r"[^\w\s.,;:?!\-']",
        "citation_pattern": r"\[[\w\d\s,-]+\]|\([\w\s.,-]+(?:\d{4})\)",
        "preserve_patterns": [
            r"[A-Z][a-z]+(?:-[A-Z][a-z]+)*",  # Drug names like Acetaminophen-APAP
            r"\d+(?:\.\d+)?%",  # Percentages
            r"\d+mg|\d+ml|\d+g",  # Dosages
        ],
    },
    "financial": {
        "special_chars_pattern": r"[^\w\s.,;:?!\-$%']",
        "citation_pattern": r"\[[\w\d\s,-]+\]|\([\w\s.,-]+(?:\d{4})\)",
        "preserve_patterns": [
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # Currency like $1,000.00
            r"\d+(?:\.\d{2})?%",  # Percentages
            r"\d+(?:,\d{3})*",  # Large numbers like 1,000,000
        ],
    },
    "news": {
        "special_chars_pattern": r"[^\w\s.,;:?!\-']",
        "citation_pattern": r"\[[\w\d\s,-]+\]|\([\w\s.,-]+(?:\d{4})\)",
        "preserve_patterns": [
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",  # Proper names like John Doe
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # Dates like 12/31/2023
        ],
    },
}


def clean_chunk(
    text: str,
    mode: Literal[
        "full",
        "scientific",
        "social",
        "legal",
        "technical",
        "medical",
        "financial",
        "news",
    ] = "full",
    return_type: Literal["text", "tokens"] = "text",
    remove_citations: bool = True,
    remove_special_chars: bool = True,
    normalize_whitespace: bool = True,
    min_token_length: int = 2,
    custom_stopwords: Optional[List[str]] = None,
    preserve_numbers: bool = True,
    preserve_case: bool = False,
    preserve_urls: bool = True,
    preserve_emails: bool = True,
    remove_markdown: bool = True,
) -> Union[str, List[str]]:
    """
    Enhanced universal text cleaner for all document types with multiple processing modes.

    Args:
        text: Input text to clean
        mode: Processing profile for different document types
        return_type: 'text' returns string, 'tokens' returns processed tokens
        remove_citations: Remove citation markers
        remove_special_chars: Remove non-alphanumeric characters
        normalize_whitespace: Normalize all whitespace
        min_token_length: Minimum length for tokens to keep
        custom_stopwords: Additional stopwords to remove
        preserve_numbers: Whether to preserve numerical values
        preserve_case: Whether to preserve original case
        preserve_urls: Whether to preserve URLs
        preserve_emails: Whether to preserve email addresses
        remove_markdown: Whether to remove markdown formatting

    Returns:
        Cleaned text or tokens based on return_type
    """
    if not text or not isinstance(text, str):
        return "" if return_type == "text" else []

    # Initialize components
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
    except Exception as e:
        logging.warning(f"NLTK initialization warning: {e}")
        lemmatizer = None
        stop_words = set()

    if custom_stopwords:
        stop_words.update(custom_stopwords)

    # Get configuration for the selected mode
    config = DOCUMENT_CONFIGS.get(mode, DOCUMENT_CONFIGS["full"])

    # Remove markdown formatting if requested
    if remove_markdown:
        # Remove markdown headers
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        # Remove markdown bold/italic
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
        # Remove markdown links
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove markdown code blocks
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove markdown lists (unordered and ordered)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Preserve URLs and emails if requested
    url_placeholders = {}
    email_placeholders = {}

    if preserve_urls:
        urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", text)
        for i, url in enumerate(urls):
            placeholder = f"__URL_PLACEHOLDER_{i}__"
            url_placeholders[placeholder] = url
            text = text.replace(url, placeholder, 1)

    if preserve_emails:
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
        for i, email in enumerate(emails):
            placeholder = f"__EMAIL_PLACEHOLDER_{i}__"
            email_placeholders[placeholder] = email
            text = text
