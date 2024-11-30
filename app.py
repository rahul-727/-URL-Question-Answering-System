import streamlit as st
import requests

st.title("URL-Based Query System")

url_input = st.text_input("Enter a URL:")
query_input = st.text_input("Ask a question about the URL:")

if st.button("Submit"):
    if url_input and query_input:
        # Step 1: Send the URL to the FastAPI endpoint for parsing
        url_response = requests.post("http://127.0.0.1:8000/url-parser", json={"url": url_input})

        if url_response.status_code == 200:
            st.success("URL processed successfully.")
            
            # Step 2: Send the query to the FastAPI endpoint to get the answer
            query_response = requests.post("http://127.0.0.1:8000/query", json={"query": query_input})

            if query_response.status_code == 200:
                # Extract the generated answer from the FastAPI response
                answer = query_response.json().get('answer', 'No answer found.')
                
                st.write("### Answer:")
                st.write(f"- {answer}")
            else:
                st.error("Error retrieving answers from the FastAPI query endpoint.")
        else:
            st.error("Error processing the URL with the FastAPI URL-parser endpoint.")
    else:
        st.warning("Please provide both a URL and a query.")
