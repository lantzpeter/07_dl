import streamlit as st
import json
import numpy as np
import random
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

#Loading key for Google AI
api_key = st.secrets["API_KEY"]

#Initiating GenAi model from Google.
client = genai.Client(api_key=api_key)

#Loading model
model = SentenceTransformer("model/my_model")

#Loading embeddings
with open("embeddings/embeddings_stat.json") as f:
    data_stat = json.load(f)
texts_stat = [d["text"] for d in data_stat]
embeddings_stat = np.array([d["embedding"] for d in data_stat])

with open("embeddings/embeddings_homl.json") as f:
    data_homl = json.load(f)
texts_homl = [d["text"] for d in data_homl]
embeddings_homl = np.array([d["embedding"] for d in data_homl])

with open("embeddings/embeddings_lotr.json") as f:
    data_lotr = json.load(f)
texts_lotr = [d["text"] for d in data_lotr]
embeddings_lotr = np.array([d["embedding"] for d in data_lotr])

#Tagging texts so i can trace them
texts_stat_tagged = [f"[STAT] {t}" for t in texts_stat]
texts_homl_tagged = [f"[HOML] {t}" for t in texts_homl]
#Combining both stat and homl so the model can search in both.
texts_combined = texts_stat_tagged + texts_homl_tagged
embeddings_combined = np.vstack([embeddings_stat, embeddings_homl])

#Setting prompt
system_prompt = """You are a helpful assistant who explains statistical and machine learning concepts clearly.

You will receive a question along with some context. You must base your answer **only on the provided context**, and not on any external knowledge.

If the context does not contain enough information to answer the question, simply reply:  
**"I don't know."** Do not attempt to guess or invent details.

Please write your answer in clear, simple language, and structure it into well-formed paragraphs.

Use a slightly whimsical tone, as if you were Gandalf explaining it over a campfire.

---

Context:
{context}

Question:
{query}

To make it more entertaining, include a Lord of the Rings quote that fits the context. You may **tweak the quote slightly** to make it funnier or more related to the topic.

At the end of your answer, include the quote on its own line, followed by the name of the character who said it.
For example:
"There is always hope." – Aragorn

Quote:
{lotr_quote}" – {lotr_author}
"""

#
def generate_user_prompt(query, texts, embeddings, model, k=5):
    context_chunks = semantic_search(query, texts, embeddings, model, k)
    context = "\n\n".join([chunk for chunk, score in context_chunks])
    return f"Question: {query}\n\nContext:\n{context}"

def semantic_search(query, texts, embeddings, model, k=5):
    query_emb = model.encode([query])
    similarity_scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = similarity_scores.argsort()[-k:][::-1]
    return [(texts[i], similarity_scores[i]) for i in top_indices]

def extract_lotr_author(quote): #I want to present the author of the quote later on so thats why I extract it seperatly.
    if "–" in quote:
        return quote.split("–")[-1].strip()
    elif "-" in quote:
        return quote.split("-")[-1].strip()
    else:
        return "Unknown"

def generate_response(system_prompt, query, texts, embeddings, model, client, lotr_quotes, gemini_model="gemini-2.0-flash"):
    lotr_quote = random.choice(lotr_quotes)
    lotr_author = extract_lotr_author(lotr_quote)
    user_prompt = generate_user_prompt(query, texts, embeddings, model)

    filled_prompt = system_prompt.format(
        context=user_prompt.split("Context:\n")[1],
        query=query,
        lotr_quote=lotr_quote,
        lotr_author=lotr_author
    )

    response = client.models.generate_content(
        model=gemini_model,
        contents=filled_prompt
    )

    return response.text

st.image("image/hero_gandalf.png")
st.title("GandalfBot – Ask about ML & Stats")
st.text("GandalfBot has watched The Lord of the Rings too many times and now believes he's actually Gandalf. " \
"He loves answering questions about machine learning and statistics in a slightly mysterious tone..." \
" often quoting the films as if Frodo himself asked about linear regression.")
st.text("He's studied two mighty tomes: ISLR and Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.")
st.text("Be prepared to learn, young hobbit. Ask your question below 👇")

query = st.text_input("Ask your question:")

if query:
    st.markdown("---")
    st.markdown("### Gandalf says:")
    response = generate_response(
        system_prompt,
        query,
        texts_combined,
        embeddings_combined,
        model,
        client,
        texts_lotr
    )
    st.write(response)