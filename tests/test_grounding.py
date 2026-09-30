from pathlib import Path

from app.agents.rag_agent import answer_pdf_question
from app.guardrails.grounding import GroundingCheck, check_grounding, lexical_support
from app.guardrails.messages import UNSUPPORTED_MESSAGE
from app.guardrails.output_guardrail import apply_output_guardrail
from app.rag.loader import load_pdf

CTX = ["Authentication is handled through OAuth 2.0 tokens issued by the identity service."]


async def test_hallucinated_answer_is_rejected_end_to_end(store, fake_llm):
    store([(3, CTX[0])])
    fake_llm.answer = "According to the uploaded PDF, authentication uses biometric retina scanning. [S1]"
    fake_llm.grounding = GroundingCheck(grounded=False, supported=False, reason="not in context")
    out = await answer_pdf_question("What does the PDF say about authentication?")
    assert out.text == UNSUPPORTED_MESSAGE
    assert "retina" not in out.text


async def test_validator_cannot_wave_through_zero_overlap(fake_llm):
    fake_llm.grounding = GroundingCheck(grounded=True, supported=True, reason="trust me")
    r = await check_grounding("q", CTX, "Quantum entanglement enables faster-than-light chess strategies.")
    assert not (r.grounded and r.supported)


async def test_validator_outage_falls_back_to_lexical(fake_llm):
    fake_llm.grounding = RuntimeError("429 rate limit")
    good = await check_grounding("q", CTX, "Authentication is handled through OAuth 2.0 tokens.")
    bad = await check_grounding("q", CTX, "Passwords are stored in plain text on a shared spreadsheet.")
    assert good.supported and not bad.supported


async def test_validator_outage_rejects_wrong_number_from_policy_pdf(fake_llm):
    fake_llm.grounding = RuntimeError("429 rate limit")
    pdf_path = Path(__file__).resolve().parents[1] / "data" / "pdfs" / "company_leave_policy.pdf"
    context = [page.text for page in load_pdf(pdf_path)]
    question = "How many annual leave days are employees entitled to?"
    correct = "Full-time employees are entitled to 20 paid annual leave days per calendar year."
    correct_sick = "Employees are entitled to 10 paid sick leave days per year."
    incorrect_number = "Full-time employees are entitled to 200 paid annual leave days per calendar year."
    incorrect_category = "Full-time employees are entitled to 10 paid annual leave days per calendar year."
    incorrect_association = "Male employees are entitled to 90 days of paid paternity leave."

    supported = await check_grounding(question, context, correct)
    supported_sick = await check_grounding(question, context, correct_sick)
    unsupported_number = await check_grounding(question, context, incorrect_number)
    unsupported_category = await check_grounding(question, context, incorrect_category)
    unsupported_association = await check_grounding(question, context, incorrect_association)

    assert supported.supported
    assert supported_sick.supported
    assert not unsupported_number.supported
    assert not unsupported_category.supported
    assert not unsupported_association.supported


def test_lexical_support():
    assert lexical_support("OAuth 2.0 tokens issued by the identity service", CTX)
    assert not lexical_support("completely unrelated statement about volcanoes", CTX)


def test_output_guardrail_enforces_failed_grounding():
    failed = {"grounded": False, "supported": False, "reason": "x"}
    assert apply_output_guardrail("Some unsupported claim", failed) == UNSUPPORTED_MESSAGE
