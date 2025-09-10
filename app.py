# ----------- Book Matcher (Streamlit) -----------
# pip install streamlit requests sentence-transformers numpy scikit-learn textstat

import re
import json
import difflib
import urllib.parse as up
import requests
import numpy as np
import textstat
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Light Book Matcher", page_icon="📚", layout="wide")


# ============================================================
# Embeddings
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-d


EMB = load_embedder()


def embed_texts(texts):
    vec = EMB.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float16)


def cosine(a, B):
    a = a / (np.linalg.norm(a) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (Bn @ a).astype(np.float32)


# ============================================================
# Tiny stylometry (query-side only)
# ============================================================
def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def style_feats(t: str):
    t = (t or "").strip()
    if not t:
        return np.zeros(8, dtype=np.float32)
    sents = split_sents(t)
    lens = [len(s.split()) for s in sents] or [0]
    words = re.findall(r"[A-Za-z']+", t.lower())
    tokens = len(words) or 1
    types = len(set(words))
    ttr = types / tokens
    comma = t.count(",") / max(len(sents), 1)
    dash = (t.count("—") + t.count("-")) / max(len(sents), 1)
    ell = t.count("...") / max(len(sents), 1)
    stop = sum(w in ENGLISH_STOP_WORDS for w in words) / tokens
    read = textstat.flesch_reading_ease(t) if tokens > 50 else 0.0
    v = np.array(
        [np.mean(lens), np.var(lens), ttr, comma, dash + ell, stop, read, len(sents)],
        np.float32,
    )
    return (v - v.mean()) / (v.std() + 1e-6)


# ============================================================
# Open Library helpers
# ============================================================
def ol_search(q, limit=50):
    r = requests.get(
        "https://openlibrary.org/search.json",
        params={"q": q, "limit": limit, "language": "eng"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("docs", [])


def ol_work(work_key):
    r = requests.get(f"https://openlibrary.org{work_key}.json", timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def ol_work_description(work_key: str) -> str:
    try:
        w = ol_work(work_key)
    except Exception:
        return ""
    d = w.get("description")
    if isinstance(d, dict):
        d = d.get("value")
    return (d or w.get("subtitle") or "").strip()


@st.cache_data(show_spinner=False)
def ol_subjects_for_work(work_key: str) -> str:
    """
    Return a ';'-joined genre/subject string for a work.
    Pulls subjects + places/people/times as a fallback.
    """
    if not work_key:
        return ""
    try:
        w = ol_work(work_key)
    except Exception:
        return ""
    parts = []
    parts += w.get("subjects", []) or []
    parts += w.get("subject_places", []) or []
    parts += w.get("subject_people", []) or []
    parts += w.get("subject_times", []) or []
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return ";".join(out[:6])


def _uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _capitalized_keywords(text):
    caps = re.findall(r"(?<!\.)\s([A-Z][a-zA-Z\-]{2,})", text)
    return _uniq([c for c in caps if c.lower() not in ENGLISH_STOP_WORDS])[:6]


def _longword_keywords(text):
    words = [
        w
        for w in re.findall(r"[A-Za-z']+", text.lower())
        if w not in ENGLISH_STOP_WORDS and len(w) >= 5
    ]
    return _uniq(words)[:6]


def _ol_subject(subject, limit=50):
    url = f"https://openlibrary.org/subjects/{up.quote(subject)}.json"
    r = requests.get(url, params={"limit": limit}, timeout=20)
    if r.status_code != 200:
        return []
    data = r.json()
    works = data.get("works", [])
    docs = []
    for w in works:
        docs.append(
            {
                "key": w.get("key"),  # /works/...
                "title": w.get("title", ""),
                "author_name": [a.get("name", "") for a in w.get("authors", [])],
                "first_publish_year": w.get("first_publish_year"),
                "subject": w.get("subject", []),
                "subtitle": (
                    w.get("description")
                    if isinstance(w.get("description"), str)
                    else ""
                ),
            }
        )
    return docs


def _normalize_title(t):
    t = re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
    return t


def _docs_to_rows(docs, cap):
    rows = []
    for d in docs:
        rows.append(
            {
                "book_id": d.get("key")
                or d.get("cover_edition_key")
                or (d.get("edition_key", [None])[0] if d.get("edition_key") else None),
                "title": d.get("title", ""),
                "title_norm": _normalize_title(d.get("title", "")),
                "author": ", ".join(d.get("author_name", [])[:3]),
                "first_author": (d.get("author_name", [None])[0] or "").strip(),
                "year": d.get("first_publish_year"),
                "genre": ";".join((d.get("subject") or [])[:6]),
                "style_text": "",
                "synopsis": d.get("subtitle", "") or "",
                "work_key": (
                    d.get("key") if str(d.get("key", "")).startswith("/works/") else ""
                ),
            }
        )
        if len(rows) >= cap:
            break
    # De-dupe strongly: by work_key if present else by (title_norm, first_author)
    seen, out = set(), []
    for r in rows:
        key = r["work_key"] or (r["title_norm"], r["first_author"].lower())
        if r["title"] and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def find_seed_work_key(title: str, author: str | None):
    """Return (work_key, norm_title, first_author_lower) for the best-matched seed book."""
    docs = ol_search(f"{title} {author or ''}", limit=10) or ol_search(title, limit=10)
    if not docs:
        return None, _normalize_title(title), (author or "").lower().strip()
    d0 = _best_match(docs, title, author) or docs[0]
    wk = d0.get("key") or (
        d0.get("work_key", [None])[0] if d0.get("work_key") else None
    )
    norm_title = _normalize_title(d0.get("title", ""))
    first_author = (d0.get("author_name", [author or ""])[0] or "").lower().strip()
    return wk, norm_title, first_author


# -------- Excerpt → Candidates --------
def candidates_by_excerpt(excerpt, cap=200):
    caps = _capitalized_keywords(excerpt)
    longs = _longword_keywords(excerpt)
    queries = []
    if caps:
        queries.append(" ".join(caps))
    if longs:
        queries.append(" ".join(longs))
    if caps and longs:
        queries.append(" ".join(caps[:3] + longs[:3]))
    queries += ["paris memoir", "paris city winter", "urban literary fiction"]
    for q in _uniq(queries):
        docs = ol_search(q, limit=min(80, cap))
        rows = _docs_to_rows(docs, cap)
        if rows:
            return rows
    for s in _uniq(
        [c.lower() for c in caps] + ["paris", "cities", "memoirs", "fiction"]
    ):
        docs = _ol_subject(s, limit=min(80, cap))
        rows = _docs_to_rows(docs, cap)
        if rows:
            return rows
    docs = ol_search("literary fiction", limit=cap)
    return _docs_to_rows(docs, cap)


# -------- Title/Author → Candidates --------
def _best_match(docs, title, author=None):
    target = f"{title} {author or ''}".lower().strip()
    names = [
        f"{d.get('title','')} {', '.join(d.get('author_name',[])[:1])}".lower().strip()
        for d in docs
    ]
    if not names:
        return None
    i = max(
        range(len(names)),
        key=lambda j: difflib.SequenceMatcher(None, target, names[j]).ratio(),
    )
    return docs[i]


def candidates_by_title_author(title, author=None, cap=200):
    docs = ol_search(f"{title} {author or ''}", limit=10) or ol_search(title, limit=10)
    if not docs:
        return []
    d0 = _best_match(docs, title, author) or docs[0]
    wk = d0.get("key") or (
        d0.get("work_key", [None])[0] if d0.get("work_key") else None
    )
    if not wk:
        return []
    w = ol_work(wk)
    subs = (w.get("subjects") or [])[:8]
    cands = []
    for s in subs:
        docs_s = ol_search(f'subject:"{s}"', limit=40)
        cands += _docs_to_rows(docs_s, cap - len(cands))
        if len(cands) >= cap:
            break
    return cands


# ============================================================
# Rerank / diversify
# ============================================================
def diversify(order_idx, items, scores, max_per_author=1):
    """Keep highest per author; drop further ones until cap, preserving order."""
    kept = []
    seen_count = {}
    for i in order_idx:
        author = (items[i]["first_author"] or items[i]["author"]).lower().strip()
        if author:
            if seen_count.get(author, 0) >= max_per_author:
                continue
            seen_count[author] = seen_count.get(author, 0) + 1
        kept.append(i)
        if len(kept) >= len(order_idx):
            break
    return kept


# ============================================================
# Personalization helpers
# ============================================================
def _year_bucket(y):
    try:
        y = int(y)
    except Exception:
        return None
    if y < 1900:
        return "<1900"
    if y <= 1950:
        return "1900–1950"
    if y <= 2000:
        return "1950–2000"
    return ">2000"


def personal_boost(
    cand: dict,
    pref_fiction: str,
    pref_eras: list[str],
    boost_subjects: list[str],
    avoid_subjects: list[str],
    fav_authors: list[str],
) -> float:
    """
    Returns a small boost in [0, 0.35] based on user preferences.
    """
    text = " ".join(
        [
            (cand.get("genre") or ""),
            (cand.get("synopsis") or ""),
            (cand.get("title") or ""),
        ]
    ).lower()

    boost = 0.0

    # subjects
    for kw in boost_subjects:
        k = kw.lower().strip()
        if k and k in text:
            boost += 0.06
    for kw in avoid_subjects:
        k = kw.lower().strip()
        if k and k in text:
            boost -= 0.06

    # authors
    if cand.get("author"):
        for a in fav_authors:
            if a.lower().strip() and a.lower().strip() in cand["author"].lower():
                boost += 0.08

    # eras
    b = _year_bucket(cand.get("year"))
    if b and (b in set(pref_eras)):
        boost += 0.05

    # fiction vs non-fiction (heuristic)
    if pref_fiction != "No preference":
        is_nonfiction = any(
            w in text
            for w in [
                "biography",
                "autobiography",
                "memoir",
                "history",
                "essays",
                "reportage",
                "nonfiction",
                "non-fiction",
            ]
        )
        if pref_fiction == "Fiction" and not is_nonfiction:
            boost += 0.04
        if pref_fiction == "Non-fiction" and is_nonfiction:
            boost += 0.04

    return max(0.0, min(0.35, boost))


# ============================================================
# Explanation / UI helpers
# ============================================================
def _truncate(txt: str, n_chars=360) -> str:
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt if len(txt) <= n_chars else txt[: n_chars - 1].rsplit(" ", 1)[0] + "…"


def _why_line(
    score: float, query_terms: list[str], cand: dict, query_author: str | None = None
):
    hay = " ".join(
        [cand.get("title", ""), cand.get("genre", ""), cand.get("synopsis", "")]
    ).lower()
    hits = [w for w in query_terms if w and w.lower() in hay][:3]
    bits = []
    if query_author and query_author.lower() in (cand.get("author", "").lower()):
        bits.append("same author")
    if "paris" in hay:
        bits.append("Paris setting")
    if "memoir" in hay or "autobiograph" in hay:
        bits.append("memoir vibe")
    if "friend" in hay or "group" in hay:
        bits.append("ensemble cast")
    if "spain" in hay:
        bits.append("Spain travel")

    reason_core = f"semantic match {score:.2f}"
    if bits:
        reason_core += " • " + ", ".join(bits)
    if hits:
        reason_core += " • overlap: " + ", ".join(hits)
    return reason_core


# ============================================================
# UI
# ============================================================
st.title("📚 Book Matcher")

with st.sidebar:
    k = st.slider("Results (k)", 3, 20, 8)
    alpha = st.slider("Content weight α", 0.0, 1.0, 0.65, 0.05)
    max_per_author = st.select_slider("Max per author", options=[1, 2, 3], value=1)

    st.markdown("### 👤 Your profile (optional)")
    pref_fiction = st.selectbox(
        "Fiction vs Non-fiction tilt",
        ["No preference", "Fiction", "Non-fiction"],
        index=0,
    )
    pref_eras = st.multiselect(
        "Prefer eras",
        ["<1900", "1900–1950", "1950–2000", ">2000"],
        default=["1900–1950"],
    )
    pref_subjects = st.text_input(
        "Boost subjects/keywords (comma-sep)", value="paris, memoir, travel"
    )
    avoid_subjects = st.text_input("Downweight subjects (comma-sep)", value="biography")
    fav_authors = st.text_input(
        "Boost authors (comma-sep)", value="Ernest Hemingway, Jean Rhys"
    )
    w_personal = st.slider("Personalization weight γ", 0.0, 0.5, 0.15, 0.05)

    # profile import/export
    profile = {
        "pref_fiction": pref_fiction,
        "pref_eras": pref_eras,
        "pref_subjects": [s.strip() for s in pref_subjects.split(",") if s.strip()],
        "avoid_subjects": [s.strip() for s in avoid_subjects.split(",") if s.strip()],
        "fav_authors": [s.strip() for s in fav_authors.split(",") if s.strip()],
        "w_personal": float(w_personal),
    }
    st.download_button(
        "⬇️ Download profile JSON",
        data=json.dumps(profile, indent=2),
        file_name="bookmatch_profile.json",
        mime="application/json",
    )
    uploaded = st.file_uploader("⬆️ Load profile JSON", type=["json"])
    if uploaded:
        try:
            prof = json.load(uploaded)
            pref_fiction = prof.get("pref_fiction", pref_fiction)
            pref_eras = prof.get("pref_eras", pref_eras)
            pref_subjects = ", ".join(prof.get("pref_subjects", []))
            avoid_subjects = ", ".join(prof.get("avoid_subjects", []))
            fav_authors = ", ".join(prof.get("fav_authors", []))
            w_personal = float(prof.get("w_personal", w_personal))
            st.success("Profile loaded. Adjust sliders/fields if needed.")
        except Exception:
            st.error("Could not parse profile JSON.")

    st.caption(
        "We collapse duplicate editions and limit how many picks come from the same author for variety."
    )

tab1, tab2 = st.tabs(["By Excerpt", "By Title/Author"])

# -------- By Excerpt --------
with tab1:
    txt = st.text_area(
        "Paste an excerpt (100–1500 words ideal):",
        height=220,
        placeholder="Paste a paragraph here…",
    )
    if st.button("Find similar (excerpt)"):
        if not txt.strip():
            st.warning("Please paste an excerpt.")
            st.stop()

        cands = candidates_by_excerpt(txt, cap=220)
        if not cands:
            st.error("No candidates found.")
            st.stop()

        syns = [c["synopsis"] or (c["title"] + " " + c["genre"]) for c in cands]
        V = embed_texts(syns)
        q_c = embed_texts([txt]).astype(np.float16)[0]
        scores = cosine(q_c.astype(np.float32), V.astype(np.float32))

        # --- personalization: compute per-candidate boosts ---
        pb = np.array(
            [
                personal_boost(
                    r,
                    pref_fiction=pref_fiction,
                    pref_eras=pref_eras,
                    boost_subjects=[
                        s.strip() for s in pref_subjects.split(",") if s.strip()
                    ],
                    avoid_subjects=[
                        s.strip() for s in avoid_subjects.split(",") if s.strip()
                    ],
                    fav_authors=[
                        s.strip() for s in fav_authors.split(",") if s.strip()
                    ],
                )
                for r in cands
            ],
            dtype=np.float32,
        )

        final_scores = (alpha * scores) + (w_personal * pb)

        order = np.argsort(-final_scores)
        order = diversify(order, cands, final_scores, max_per_author=max_per_author)[:k]

        query_terms = _capitalized_keywords(txt) + _longword_keywords(txt)

        st.subheader("Recommendations")
        st.markdown(
            "ℹ️ **How scores are calculated:** "
            "We embed your input and each candidate’s synopsis and compute cosine similarity (semantic match). "
            "We add a small *personalization boost* based on your sidebar preferences (authors, subjects, eras, fiction tilt), "
            "then diversify authors to avoid repeats."
        )

        for idx in order:
            r = cands[idx]
            score = final_scores[idx]

            desc = r["synopsis"]
            if (not desc) and r.get("work_key"):
                desc = ol_work_description(r["work_key"])

            why = _why_line(float(score), query_terms, r, query_author=None)
            if pb[idx] > 1e-3:
                why += f" • personalized +{pb[idx]:.2f}"

            genre = r.get("genre") or (
                ol_subjects_for_work(r.get("work_key", "")) if r.get("work_key") else ""
            )
            genre_line = f"\n\nGenres: {genre}" if genre else ""

            st.markdown(
                f"**{r['title']}** — {r['author']}\n\n"
                f"*{r['year']}*"
                f"{genre_line}\n\n"
                f"**Why:** {why}"
            )

            if desc:
                short = _truncate(desc, 320)
                st.markdown(f"> {short}")
                if short != desc:
                    with st.expander("Show full summary"):
                        st.write(desc)


# -------- By Title/Author --------
with tab2:
    t = st.text_input("Title", value="", placeholder="A Moveable Feast")
    a = st.text_input("Author", value="", placeholder="Ernest Hemingway")
    if st.button("Find similar (title/author)"):
        if not t.strip():
            st.warning("Enter a title.")
            st.stop()

        cands = candidates_by_title_author(t, a or None, cap=220)
        if not cands:
            st.error("No candidates found.")
            st.stop()

        # Drop the input book itself (same work or same title+author)
        seed_wk, seed_title_norm, seed_author = find_seed_work_key(t, a or None)
        filtered = []
        for c in cands:
            same_work = seed_wk and c.get("work_key") == seed_wk
            same_title = _normalize_title(c.get("title", "")) == seed_title_norm
            same_author = seed_author and (seed_author in (c.get("author", "").lower()))
            if not (same_work or (same_title and same_author)):
                filtered.append(c)
        cands = filtered
        if not cands:
            st.warning("Only the input book matched; broaden your query.")
            st.stop()

        query_text = (t + " " + a).strip()
        V = embed_texts(
            [(c["synopsis"] or (c["title"] + " " + c["genre"])) for c in cands]
        )
        q = embed_texts([query_text]).astype(np.float16)[0]
        scores = cosine(q.astype(np.float32), V.astype(np.float32))

        # --- personalization: compute per-candidate boosts ---
        pb = np.array(
            [
                personal_boost(
                    r,
                    pref_fiction=pref_fiction,
                    pref_eras=pref_eras,
                    boost_subjects=[
                        s.strip() for s in pref_subjects.split(",") if s.strip()
                    ],
                    avoid_subjects=[
                        s.strip() for s in avoid_subjects.split(",") if s.strip()
                    ],
                    fav_authors=[
                        s.strip() for s in fav_authors.split(",") if s.strip()
                    ],
                )
                for r in cands
            ],
            dtype=np.float32,
        )

        final_scores = (alpha * scores) + (w_personal * pb)

        order = np.argsort(-final_scores)
        order = diversify(order, cands, final_scores, max_per_author=max_per_author)[:k]

        query_terms = _longword_keywords(query_text) + _capitalized_keywords(query_text)

        st.subheader("Recommendations")
        st.markdown(
            "ℹ️ **How scores are calculated:** "
            "We embed your input and each candidate’s synopsis and compute cosine similarity (semantic match). "
            "We add a small *personalization boost* based on your sidebar preferences (authors, subjects, eras, fiction tilt), "
            "then diversify authors to avoid repeats."
        )
        for idx in order:
            r = cands[idx]
            score = final_scores[idx]
            desc = r["synopsis"]
            if (not desc) and r.get("work_key"):
                desc = ol_work_description(r["work_key"])

            why = _why_line(
                float(score), query_terms, r, query_author=(a or "").strip() or None
            )
            if pb[idx] > 1e-3:
                why += f" • personalized +{pb[idx]:.2f}"

            genre = r.get("genre") or (
                ol_subjects_for_work(r.get("work_key", "")) if r.get("work_key") else ""
            )
            genre_line = f"\n\nGenres: {genre}" if genre else ""

            st.markdown(
                f"**{r['title']}** — {r['author']}\n\n"
                f"*{r['year']}*"
                f"{genre_line}\n\n"
                f"**Why:** {why}"
            )

            if desc:
                short = _truncate(desc, 320)
                st.markdown(f"> {short}")
                if short != desc:
                    with st.expander("Show full summary"):
                        st.write(desc)
