"""Chunks must say which section they belong to.

A policy's exclusion list runs for pages under a single heading. Chunked on clause
boundaries, the items pages after that heading arrive with no statement of what they
are, and a model asked "is maternity covered?" and shown a list containing maternity
answered yes - the opposite of the policy. In the real corpus the heading is three
pages before the item (niva-bupa-reassure-2: heading page 14, item page 17; Arogya
Sanjeevani: 10 and 12; star-health-assure: 27 and 31).

The pages below mirror those three documents' layouts, shortened. Each one ends the
standard list with the heading of the next section, because two of the three real
documents do exactly that, and a heading at the bottom of a chunk describes what
follows it, not what precedes it.

These assert only that the section's name reaches the chunk's text, since that text is
what gets embedded and what the model reads. How it gets there is the chunker's
business.

Marked xfail(strict=True): they describe behaviour the chunker does not have yet. When
it does, they pass, strict turns that into a failure, and the marker must be removed -
so the fix cannot land without the test being switched on.
"""

import pytest

from policyiq.ingest.chunking import chunk_pages

TARGET, OVERLAP = 300, 50

FILLER = (
    "Expenses related to the treatment of the listed condition shall be excluded until "
    "the expiry of the stated months of continuous coverage after the date of inception. "
)

pending = pytest.mark.xfail(
    strict=True, reason="chunks do not yet carry their section heading"
)


def niva_layout() -> list[tuple[int, str]]:
    return [
        (14, "Claims are settled on the agreed tariff.\n"
             "5. Exclusions\n"
             "5.1. Standard Exclusions\n"
             "5.1.1. Pre-existing Diseases (Code-Excl01):\n" + FILLER * 3),
        (15, "5.1.2. Specified disease waiting period (Code-Excl02):\n" + FILLER * 3),
        (16, "5.1.9. Hazardous or adventure sports (Code-Excl09):\n" + FILLER * 3),
        (17, "5.1.14. Unproven treatments (Code-Excl16):\n" + FILLER * 2 +
             "\n5.1.15. Sterility and Infertility (Code-Excl17)\n"
             "Expenses related to sterility and infertility, including IVF.\n"
             "5.1.16. Maternity Expenses (Code-Excl18)\n"
             "Medical treatment expenses traceable to childbirth.\n"
             "5.2. Specific Exclusions\n"),
        (18, "5.2.1. Personal Waiting Period\n"
             "Conditions specified for an Insured Person under the personal waiting "
             "period are payable only after it ends.\n" + FILLER * 2),
    ]


def arogya_layout() -> list[tuple[int, str]]:
    return [
        (10, "Any awarded cumulative bonus shall be withdrawn.\n\n"
             "E Exclusion\n"
             "The Company shall not be liable to make any payment under the policy in "
             "connection with or in respect of following expenses:\n\n"
             "E.i. Standard Exclusions\n\n"
             "• Pre-Existing Diseases (Code- Excl01)\n"
             "a) " + FILLER * 3),
        (11, "• Hazardous or Adventure sports: (Code- Excl09)\n" + FILLER * 3),
        (12, "• Unproven Treatments: (Code- Excl16)\n" + FILLER * 2 +
             "\n• Sterility and Infertility: (Code- Excl17)\n"
             "Expenses related to sterility and infertility, including IVF.\n"
             "• Maternity Expenses (Code:Excl 18):\n"
             "i. medical treatment expenses traceable to childbirth.\n\n"
             "ii. Specific Exclusions\n"),
        (13, "• Personal Waiting Period\n"
             "Conditions specified for an Insured Person under the personal waiting "
             "period are payable only after it ends.\n" + FILLER * 2),
    ]


def star_layout() -> list[tuple[int, str]]:
    return [
        (27, "Claims are settled within thirty days.\n"
             "C. EXCLUSIONS\n"
             "STANDARD EXCLUSIONS\n"
             "1. Pre-Existing Diseases - Code Excl 01:\n" + FILLER * 3),
        (28, "9.  Hazardous or Adventure sports - Code Excl 09:\n" + FILLER * 3),
        (31, "16.  Unproven Treatments - Code Excl 16:\n" + FILLER * 2 +
             "\n17.  Sterility and Infertility (Except to the extent covered under "
             "Coverage 17) - Code Excl 17 : Expenses related to sterility and "
             "infertility, including IVF.\n"
             "18.  Maternity - Code Excl 18 (Except to the extent covered under "
             "Coverage 15)\n"
             "SPECIFIC EXCLUSIONS\n"),
        (32, "1.  Personal Waiting Period\n"
             "Conditions specified for an Insured Person under the personal waiting "
             "period are payable only after it ends.\n" + FILLER * 2),
    ]


LAYOUTS = [
    pytest.param(niva_layout, id="numbered-headings"),
    pytest.param(arogya_layout, id="lettered-headings"),
    pytest.param(star_layout, id="capitalised-headings"),
]


def chunk_containing(chunks, text: str):
    found = [c for c in chunks if text in c.content]
    assert found, f"no chunk contains {text!r}"
    return found[0]


@pending
@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_item_pages_after_its_heading_still_names_its_section(layout):
    chunks = chunk_pages(layout(), TARGET, OVERLAP)
    item = chunk_containing(chunks, "Sterility and Infertility")

    assert "standard exclusions" in item.content.lower(), item.content


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_fixture_actually_separates_the_item_from_its_heading(layout):
    """Guards the test above against passing for the wrong reason. If the item and its
    heading already landed in one chunk, that test would pass with no fix at all."""
    chunks = chunk_pages(layout(), TARGET, OVERLAP)
    item = chunk_containing(chunks, "Sterility and Infertility")
    heading_page = layout()[0][0]

    assert item.page_number != heading_page
    assert "standard exclusions" not in item.content.lower()


@pending
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_heading_at_the_end_of_a_page_belongs_to_what_follows(layout):
    """"Specific Exclusions" closes the standard list and opens the next one. The items
    after it are specific exclusions, not standard ones."""
    chunks = chunk_pages(layout(), TARGET, OVERLAP)
    following = chunk_containing(chunks, "Personal Waiting Period")

    assert "specific exclusions" in following.content.lower(), following.content
    assert "standard exclusions" not in following.content.lower(), following.content


@pytest.mark.parametrize("layout", LAYOUTS)
def test_carrying_the_heading_carries_no_text_from_the_heading_page(layout):
    """The section's name may cross a page; its text may not. A citation says the
    chunk came from the page it names - clause text from another page breaks that,
    the same rule overlap already follows. Passes today; must keep passing."""
    chunks = chunk_pages(layout(), TARGET, OVERLAP)
    heading_page = layout()[0][0]

    for chunk in chunks:
        if chunk.page_number != heading_page:
            assert "Pre-existing Diseases" not in chunk.content
            assert "Pre-Existing Diseases" not in chunk.content
