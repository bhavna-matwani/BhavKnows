# BhavKnows - Application for Answering questions about Bhavna

## Overview

This Streamlit application serves as a chatbot, providing information about Bhavna's professional background and qualifications. The chatbot is powered by large language models and can answer questions based on the provided context, which includes Bhavna's resume and additional data.

## Features

- Provides information about Bhavna's educational background, professional experience, and skills
- Allows users to ask questions and receive relevant responses
- Stores conversation history in a Cloud Firebase database for further analysis
- Supports both CSV and PDF (resume) data sources

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Cloud Firebase project

## Configuration

- The application uses a FAISS index to store the CSV and PDF data embeddings. If the index file (`faiss_index`) does not exist, it will be created automatically.
- The CSV data file path is set in the `data_source` variable, and the PDF resume file path is set in the `pdf_source` variable.
- The Cloud FireStore connection details are set using the environment secrets `firebase_key`.

## Acknowledgements

This is inspired from [HariGPT](https://github.com/harikrishnad1997/HariGPT) and is dependent on the following libraries and tools:

- [Streamlit](https://streamlit.io/) for building the web application
- [LangChain](https://langchain.com/) for integrating the language model and retrieval chain
- [OpenAI API](https://openai.com/) for the language model
- [FAISS](https://github.com/facebookresearch/faiss) for the vector database
- [Firebase](https://firebase.google.com/) for storing the conversation history

## License

This project is licensed under the [MIT License](LICENSE).
