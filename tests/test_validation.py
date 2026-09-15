"""Corpus intake rules.

Every rule here exists because a real document failed in a way that the obvious
check — "does this PDF contain text?" — waved through. The thresholds are set from
measurements over the working corpus, recorded alongside each constant.
"""

from pathlib import Path

import pytest

from policyiq.ingest.pdf import extract_pages
from policyiq.ingest.validation import check_document

HEALTHY = [
    (
        1,
        "1. The Company shall indemnify the Insured Person for Medical Expenses "
        "incurred towards Hospitalisation during the Policy Period.\n"
        "2. The premium payable under this Policy is shown in the Schedule.",
    ),
    (
        2,
        "3. Pre-Existing Diseases are covered after a waiting period of 36 months.\n"
        "4. A claim must be notified to the insurer within 48 hours of admission.",
    ),
]


def rules(pages) -> set[str]:
    return {rejection.rule for rejection in check_document(pages)}


def test_a_healthy_policy_is_accepted():
    assert check_document(HEALTHY) == []


def test_a_document_with_no_extractable_text_is_rejected():
    assert "no_text" in rules([])


def test_a_document_with_a_broken_font_map_is_rejected():
    """One corpus PDF extracted as '3olicy' and 'uniTue' - the embedded font's
    character map was wrong, so pypdf returned the wrong letters plus raw control
    bytes. It contains plenty of text, so a text-presence check passes it."""
    broken = [
        (1, "\x0b3olicy \x1dProKiEitioQ of 5eEDtes\x1d uniTue bene\xbfcial \x0cterms"),
        (2, "\x0bIQsured \x1dHosSitDl \x1dSremium\x1d cover\x0c provisioQs here"),
    ]
    assert "control_characters" in rules(broken)


def test_a_document_of_unrelated_prose_is_rejected():
    """Vocabulary that carries none of the domain's terms is either the wrong
    document or an extraction that produced plausible-looking rubbish."""
    prose = [
        (1, "The quick brown fox jumps over the lazy dog again and again today."),
        (2, "Sunlight fell across the garden wall while the kettle boiled slowly."),
    ]
    assert "low_domain_vocabulary" in rules(prose)


def test_a_document_whose_pages_repeat_is_rejected():
    """A previously sourced PDF had the entire policy text duplicated onto every
    page. Ingested, it floods retrieval with near-duplicates and makes a citation's
    page number meaningless."""
    body = (
        "1. The Company shall indemnify the Insured Person for Medical Expenses "
        "incurred towards Hospitalisation. 2. The premium and claim procedure "
        "are set out in the Schedule to this Policy issued by the insurer."
    )
    repeated = [(number, body) for number in range(1, 9)]
    assert "duplicate_pages" in rules(repeated)


def test_whitespace_differences_do_not_hide_duplicate_pages():
    """Page furniture varies the spacing between otherwise identical pages."""
    body = (
        "1. The Company shall indemnify the Insured Person for Medical Expenses "
        "incurred towards Hospitalisation. 2. The premium and claim procedure "
        "are set out in the Schedule to this Policy issued by the insurer."
    )
    repeated = [(number, body.replace(" ", "  " if number % 2 else " ")) for number in range(1, 9)]
    assert "duplicate_pages" in rules(repeated)


def test_every_document_in_the_working_corpus_passes(sample_pdf: Path):
    """The thresholds must not reject documents already judged good by hand."""
    assert check_document(extract_pages(sample_pdf)) == []


@pytest.mark.parametrize("pdf", sorted(Path("data/policies").glob("*.pdf")))
def test_each_real_corpus_document_passes(pdf: Path):
    """Skipped in CI, where the corpus is absent; locally this is the check that
    stops a threshold being tightened into rejecting good documents."""
    assert check_document(extract_pages(pdf)) == []
