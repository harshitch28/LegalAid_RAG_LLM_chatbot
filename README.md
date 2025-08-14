# 🇮🇳 LegalAid — Conversational RAG Chatbot with Memory

## Overview

LegalAid is a conversational chatbot designed for the legal domain, leveraging Retrieval-Augmented Generation (RAG) and persistent memory to provide context-aware, personalized responses. The system maintains multi-session conversation history, enabling progressive and adaptive interactions across time.

This project was developed as part of an assignment focusing on building a domain-specific RAG system with long-term memory and session management.

## Problem Statement

Create a chatbot that maintains conversation history while using RAG to provide contextual responses. The system should remember previous interactions from multiple chat sessions and build upon them while retrieving relevant information.

## Key Features

- **Multi-session conversation memory management:** Each user can have multiple chat sessions, with history tracked and retrievable.
- **Context-aware response generation:** Responses are generated using both retrieved knowledge base (KB) snippets and relevant user memory.
- **Historical interaction tracking:** All previous messages are stored and considered for future responses.
- **Progressive conversation building:** The chatbot builds upon prior exchanges for deeper, more relevant answers.
- **Personalized response adaptation:** Responses are tailored to the user's session and history.

## Technical Challenges Addressed

- Long-term memory storage and retrieval
- Context window management for prompt construction
- Session continuity across time
- Memory relevance scoring for retrieval
- Privacy and data retention considerations

## Architecture

- **Embedding Model:** Uses Gemini or HuggingFace Sentence Transformers for semantic search.
- **Vector Database:** Chroma is used for storing and retrieving KB chunks.
- **Chunking Strategy:** Legal documents are chunked for efficient retrieval.
- **Frontend:** Built with Gradio for an interactive web UI.
- **Backend:** Python-based, modular code for RAG pipeline and memory management.

## Demo

A fully working demo is deployed via Gradio.  
**[Live Demo Link]   =>   https://huggingface.co/spaces/harshit-chauhan-28/LegalAid-Chatbot-RAG

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harshitch28/LegalAid_RAG_LLM_chatbot.git
   cd LegalAid_RAG_LLM_chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   - Copy `.env.example` to `.env` and set your API keys and configuration.

4. **Run the app:**
   ```bash
   python app.py
   ```
   The Gradio UI will launch in your browser.

## Usage

- Enter your User ID and start a new session or select an existing one.
- Ask legal questions; the chatbot will use both your conversation history and legal KB for answers.
- View citations and memory snippets used in each response.
- Export conversations or clear local history as needed.

## Evaluation

- **Retrieval accuracy:** Assessed by relevance of KB/memory snippets.
- **Latency:** Response generation time is displayed.
- **RAGAS or basic metrics:** Can be extended for further evaluation.

## Privacy & Data Retention

- User memory and session data are stored for continuity but can be cleared.
- Data is retained only for demonstration purposes; see code for details.

## Project Scope

- **Domain:** Law (can be adapted for healthcare, finance, etc.)
- **Extensible:** Modular design allows for easy adaptation to other domains or data types.

## Submission Requirements

- **GitHub repo link:** =>    https://github.com/harshitch28/LegalAid_RAG_LLM_chatbot
- **Deployed app link:**   =>   https://huggingface.co/spaces/harshit-chauhan-28/LegalAid-Chatbot-RAG

## License

This project is for educational purposes only.  
_Disclaimer: The chatbot does not provide legal advice. See disclaimer in the app UI._

---

**Developed by:** Harshit Chauhan  
**Contact:** harshit.cr@gmail.com

