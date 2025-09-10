# 📚 Book Matcher

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://your-app-url.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

---

##  Goal

The goal of **Book Matcher** is simple:  
Take either:

- An **excerpt** from a book (100–1500 words), or  
- A **book title and author name**

and return **similar book recommendations**.

This project started as a fun way for me to get back into reading after an unintended gap. My recent reads made me curious about exploring similar works. Googling and Reddit threads were helpful, but I wanted something interactive: a tool where I could paste an excerpt or type a title and see what comes back.  

The lists here may not be “universal truths”—but they reflect a **defined similarity metric** and gave me a way to learn new tools, build a small pipeline, and personalize recommendations.

---

##  Tools & Libraries

- [**Streamlit**](https://streamlit.io/) — web app framework for rapid prototyping.
- [**Sentence-Transformers**](https://www.sbert.net/) — semantic embeddings (we use `all-MiniLM-L6-v2`, a lightweight 384-dim model).
- [**scikit-learn**](https://scikit-learn.org/stable/) — stopword lists for stylometry.
- [**Textstat**](https://pypi.org/project/textstat/) — readability metrics.
- [**Open Library API**](https://openlibrary.org/developers/api) — metadata source for titles, authors, subjects, descriptions.
- [**NumPy**](https://numpy.org/) — cosine similarity & vector math.
- [**Requests**](https://docs.python-requests.org/en/latest/) — HTTP client for APIs.

---

## ⚙️ Logic & Workflow

### 1. Input
- **By Excerpt:** extract keywords (capitalized words, long nouns) → query Open Library → gather candidate books.  
- **By Title/Author:** lookup the work → fetch its subjects → pull related works.

### 2. Candidate Retrieval
- Queries Open Library’s `/search.json` and `/subjects/<name>.json`.
- Deduplicates by `work_key` or normalized title + author.
- Caps results and enforces author diversity.

### 3. Feature Extraction
- **Semantic embeddings:**  
  Encode synopsis (or fallback: title + genre). Compare to query with cosine similarity.  
- **Style features:**  
  From excerpt only: sentence length variance, type–token ratio, stopword density, punctuation rhythm, readability.

### 4. Scoring
Final score is a weighted sum:

final_score = α * semantic_similarity
+ β * style_alignment
+ γ * personal_boost
– δ * duplication_penalty


- `semantic_similarity`: cosine between embeddings.  
- `style_alignment`: query-side stylometry.  
- `personal_boost`: nudges for your preferences (see below).  
- `duplication_penalty`: ensures variety.  

### 5. Output
- **Title + Author**  
- **Year**  
- **Genres/Subjects** (from work metadata)  
- **Why this recommendation** (semantic score, keyword overlaps, tags like *Paris setting* or *memoir vibe*).  
- **Summary** (short blurb inline; expandable full description).

---

##  Personalization

One of the most fun parts is **personalization**. In the sidebar you can:

- Tilt results toward **fiction vs non-fiction**.  
- Prefer certain **eras** (`<1900`, `1900–1950`, etc.).  
- Add **boost subjects/keywords** (e.g., “paris, memoir, travel”).  
- Add **downweight subjects** (e.g., “biography”).  
- Boost specific **authors** you like.  
- Adjust the **personalization weight (γ)** with a slider.  

Each recommendation shows if a personal boost contributed (`personalized +0.12`).  
Profiles can be exported/imported as JSON, so you can carry your preferences across devices.

---

##  Example

**Input:** Excerpt from *A Moveable Feast* (Ernest Hemingway).  
**Recommendations:**  
- *The Sun Also Rises* — Ernest Hemingway  
  *Why: semantic match 0.71 • Paris setting • same author*  
- *Good Morning, Midnight* — Jean Rhys  
  *Why: semantic match 0.65 • memoir vibe • overlap: paris*  
- *Down and Out in Paris and London* — George Orwell  
  *Why: semantic match 0.63 • memoir vibe • urban poverty*  

---

##  Deployment

- Hosted on **Streamlit Community Cloud**.  
- `requirements.txt` pins dependencies.  
- `runtime.txt` sets Python 3.11.  
- First run downloads the embedding model (~90 MB). Cached after that.  

---

##  Getting Started (Local)

```bash
# Clone repo
git clone https://github.com/your-username/book-matcher.git
cd book-matcher
# Create environment
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
# Install deps
pip install -r requirements.txt
# Run app
streamlit run app.py
```
---

##  Acknowledgements

Open Library
 for free metadata.

Hugging Face
 for Sentence Transformers.

Streamlit
 for an effortless app framework.