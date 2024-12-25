import spacy
from spacy.cli import download

# Download the 'en_core_web_sm' model if not already installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

