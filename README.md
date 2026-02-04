#  Multilingual Semantic Search using Sentence Embeddings

##  Overview
This project demonstrates a **multilingual semantic search system** using transformer-based sentence embeddings.  
It maps sentences from different languages into a **shared semantic vector space**, enabling meaning-based similarity search across languages.

The project is especially relevant for **Indian language NLP applications**, including multilingual search, translation alignment, and language understanding systems.

---

##  Motivation
Keyword-based search systems fail when:
- the query and documents are in **different languages**
- wording differs while the **meaning remains the same**

This project addresses these challenges by using **semantic embeddings**, where sentences with similar meanings are located close to each other in vector space regardless of language.

---

##  Core Idea
- Convert sentences into dense numerical vectors (embeddings)
- Store embeddings for multilingual text
- Compute similarity between query and stored embeddings using **cosine similarity**
- Retrieve semantically similar sentences across languages

---

---

##  Technologies Used
- Python
- Sentence Transformers
- Hugging Face Transformers
- NumPy
- scikit-learn

---

##  How It Works

###  Generate Sentence Embeddings
```bash
python embeddings.py


## 🏗️ Project Structure
