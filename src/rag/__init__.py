"""
Real-RAG evaluation track (Phase 2b / 5b).

Provides a ChromaDB-backed retrieval pipeline that injects the existing
`tool_output_misaligned` attack documents as `tool` / `ipython` role messages
before model inference, exercising the trained System > User > Tool privilege
boundary at the chat-template level rather than via inline string injection.
"""
