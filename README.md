# O Guarani Retrieval Chatbot

A retrieval-based question-answering chatbot for José de Alencar's
**O Guarani**, built with classical NLP techniques.

## Pipeline

The recovered notebook:

- loads and cleans the book text;
- tokenizes it into overlapping chunks;
- lemmatizes text and removes stopwords with spaCy;
- represents chunks using **TF-IDF**;
- detects simple question intent (`quem`, `onde`, `quando`, `como`, `o que`);
- extracts named entities;
- retrieves the top text passages with cosine similarity;
- formats a response from the retrieved context.

This is intentionally presented as a **retrieval/NLP project**, not as an LLM
or generative model.

## Source text

The original notebook clones the text repository:

`gustavoFraiz/GuaraniTXT`

The book text itself is not duplicated in this repository package.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── retrieval.py
└── notebooks/
    └── o_guarani_chatbot.ipynb
```


