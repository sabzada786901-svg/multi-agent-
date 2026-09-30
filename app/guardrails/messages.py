"""Fixed user-facing messages used by the guardrails (single source of truth)."""

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded PDF."

UNSUPPORTED_MESSAGE = (
    "I couldn't provide a reliable answer because the required information "
    "was not sufficiently supported by the uploaded PDF."
)

OUT_OF_SCOPE_MESSAGE = (
    "I understand your question, but I can only answer questions related to the information "
    "available in the uploaded PDF. I can't provide a related answer to this question. "
    "I can also help with your GitHub repositories, Google Calendar and email."
)

UNSAFE_MESSAGE = (
    "I can't help with that request. I won't reveal system instructions, credentials or secrets, "
    "or follow instructions that try to override my safety rules."
)

NO_INDEX_MESSAGE = (
    "No PDF has been indexed yet. Put a PDF in data\\pdfs and run:\n"
    '    python -m app.ingest "data\\pdfs\\your-file.pdf"'
)
