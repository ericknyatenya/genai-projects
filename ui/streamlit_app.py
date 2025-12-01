import streamlit as st
import requests

API_URL = "http://localhost:8000"


def main():
    st.title("genai-projects — RAG demo")
    q = st.text_input("Query")
    provider = st.selectbox("Provider", ["hf", "openai"])
    if st.button("Ask") and q:
        try:
            r = requests.post(API_URL + "/rag", json={"query": q, "provider": provider}, timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
