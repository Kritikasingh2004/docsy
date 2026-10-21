# Todo

## 1. Document Ingestion

- [ ] Create a persistent local ChromaDB client and a collection for document chunks.
- [ ] Use an Ollama embedding model such as `nomic-embed-text` to turn chunks into vectors.
- [ ] Store chunk text, document ID, filename, page number, and chunk index, along with embeddings.
- [ ] Generate a document ID, process the PDF, store chunks in ChromaDB, and return a summary.

## 2. Document Retrieval

- [ ] Embed a user's question and query ChromaDB for the most relevant chunks.
- [ ] Allow retrieval across all uploaded documents or restrict it to a selected document ID.
- [ ] Include filenames, page numbers, and relevant chunk text in the retrieved results.
- [ ] Verify that questions retrieve the correct sections.
- [ ] Ensure unrelated questions do not retrieve misleading context.

## 3. Question Answering with Ollama

- [ ] Pull a chat model suitable for your hardware and verify the local API works.
- [ ] Pass the question and retrieved context to Ollama through its local API.
- [ ] Tell the model to answer from the supplied document context and cite pages.
  - [ ] Make the model admit when the documents do not contain the answer.
- [ ] Create `POST /chat/ask`.
  - [ ] Accept a question and optional document ID, then retrieve context and generate an answer.
- [ ] Return the answer, source filenames, page numbers, and optionally the retrieved excerpts.

## 4. Reliability and Frontend Integration

- [ ] Decide whether re-uploading a file replaces an existing document or creates a new document.
  - [ ] Add document deletion.
- [ ] Handle Ollama being offline, missing models, empty retrieval results, and embedding failures.
- [ ] Upload a test PDF, ask a question whose answer is in it, and verify the answer's source page.
- [ ] Connect the frontend to document upload, document listing, and question-answering endpoints.
