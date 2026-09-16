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

The chunker puts the section on the chunk's first line and also keeps it in
`Chunk.section`; `body` below is the chunk's text without it, which is what the
guards need to inspect.
"""

import pytest

from policyiq.ingest.chunking import chunk_pages

TARGET, OVERLAP = 300, 50

FILLER = (
    "Expenses related to the treatment of the listed condition shall be excluded until "
    "the expiry of the stated months of continuous coverage after the date of inception. "
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


def body(chunk) -> str:
    """The chunk's own text, without the section line the chunker adds."""
    return chunk.content.removeprefix(chunk.section).lstrip("\n")


def chunk_containing(chunks, text: str):
    found = [c for c in chunks if text in c.content]
    assert found, f"no chunk contains {text!r}"
    return found[0]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_item_pages_after_its_heading_still_names_its_section(layout):
    chunks = chunk_pages(layout(), TARGET, OVERLAP)
    item = chunk_containing(chunks, "Sterility and Infertility")

    assert "standard exclusions" in item.content.lower(), item.content


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_fixture_actually_separates_the_item_from_its_heading(layout):
    """Guards the test above against passing for the wrong reason. If the item and its
    heading already landed in one chunk's text, that test would pass with no fix at
    all - so this checks the body, without the section line the fix adds."""
    chunks = chunk_pages(layout(), TARGET, OVERLAP)
    item = chunk_containing(chunks, "Sterility and Infertility")
    heading_page = layout()[0][0]

    assert item.page_number != heading_page
    assert "standard exclusions" not in body(item).lower()


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
    the same rule overlap already follows."""
    chunks = chunk_pages(layout(), TARGET, OVERLAP)
    heading_page = layout()[0][0]

    for chunk in chunks:
        if chunk.page_number != heading_page:
            assert "Pre-existing Diseases" not in chunk.content
            assert "Pre-Existing Diseases" not in chunk.content


# --- Defences against headings the PDF reader damaged ------------------------------
# A missed heading is not a neutral failure: the section before it keeps its label, and
# exclusion items end up labelled as coverage. All three cases below came from the
# real corpus.


def only_chunk_containing(pages, text: str):
    return chunk_containing(chunk_pages(pages, TARGET, OVERLAP), text)


def test_a_digit_printed_for_a_letter_does_not_hide_a_heading():
    """Policy_Document_Arogya_Sanjeevani prints its heading as "7. EXCLUS1ONS"."""
    pages = [
        (7, "4. COVERAGE\n" + FILLER * 3),
        (11, "7. EXCLUS1ONS\n" + FILLER * 3),
        (12, "Sterility and Infertility is not payable.\n" + FILLER * 2),
    ]
    item = only_chunk_containing(pages, "Sterility and Infertility")

    assert "EXCLUSIONS" in item.section
    assert "COVERAGE" not in item.section


def test_a_line_the_reader_cut_short_is_not_a_heading():
    """"3.9. Condition Pr" is a definition whose title was truncated. Read as a heading,
    it relabelled every definition after it as terms and conditions."""
    pages = [
        (1, "3. DEFINITIONS\n" + FILLER * 3),
        (2, "3.9. Condition Pr\n" + FILLER * 3),
        (3, "3.10. Congenital Anomaly means a condition present since birth.\n" + FILLER),
    ]
    item = only_chunk_containing(pages, "Congenital Anomaly")

    assert item.section == "Section: DEFINITIONS"


def test_a_coded_exclusion_item_overrides_a_missed_heading():
    """IRDAI assigns Excl codes to exclusions only. If the heading was lost entirely,
    the codes still identify the section, rather than leaving COVERAGE in place."""
    pages = [
        (7, "4. COVERAGE\n" + FILLER * 3),
        (11, "7. [heading lost]\n7.1. Pre-Existing Diseases (Code-Excl01)\n" + FILLER * 3),
        (12, "Sterility and Infertility is not payable.\n" + FILLER * 2),
    ]
    item = only_chunk_containing(pages, "Sterility and Infertility")

    assert "exclusions" in item.section.lower()
    assert "COVERAGE" not in item.section


def test_a_mid_sentence_reference_to_an_exclusion_code_is_not_an_exclusion():
    """star-health-assure's benefits say Exclusion no.1 (Code-Excl 01) does not apply to
    a benefit. That refers to an exclusion; the text around it is still coverage."""
    pages = [
        (15, "4. COVERAGE\n"
             "iii.  Exclusion no.1, (Code-Excl 01), Exclusion no.2 (Code-Excl 02) shall\n"
             "not apply to this cover.\n" + FILLER * 3),
        (16, "Home care treatment is payable up to the limit.\n" + FILLER * 2),
    ]
    item = only_chunk_containing(pages, "Home care treatment")

    assert item.section == "Section: COVERAGE"


def test_a_heading_closing_a_short_clause_does_not_label_that_clause():
    """The layouts above reach the chunker as long clauses and go through the hard
    split. This one arrives as short clauses that get merged, the other route through
    the chunker: the clause holding the maternity item ends with the next section's
    heading, and still has to be labelled as the section it belongs to."""
    pages = [
        (14, "5. Exclusions\n5.1. Standard Exclusions\n" + FILLER),
        (17, FILLER + "\n\n" + FILLER + "\n\n"
             "5.1.16. Maternity Expenses (Code-Excl18)\n"
             "Medical treatment expenses traceable to childbirth.\n"
             "5.2. Specific Exclusions"),
    ]
    chunks = chunk_pages(pages, TARGET, OVERLAP)
    item = chunk_containing(chunks, "Maternity Expenses")
    first_on_page = next(c for c in chunks if c.page_number == 17)

    assert item != first_on_page, "fixture must make the item start a new chunk"
    assert item.section == "Section: Exclusions > Standard Exclusions"


def test_a_numbered_heading_led_by_its_section_word_may_end_in_a_full_stop():
    """niva-bupa-reassure-2: "4. Benefits available under the policy." Rejected, eight
    pages of benefits were labelled as definitions."""
    pages = [
        (5, "2.2. Specific Definitions\n" + FILLER * 2 +
            "\n4. Benefits available under the policy.\n4.1. Expenses in reaching a Hospital"),
        (6, "Road ambulance expenses are payable up to the limit.\n" + FILLER * 2),
    ]
    item = only_chunk_containing(pages, "Road ambulance")

    assert item.section == "Section: Benefits available under the policy"


def test_a_sentence_ending_in_a_section_word_and_a_full_stop_is_not_a_heading():
    pages = [
        (1, "5. Exclusions\n" + FILLER * 2 + "\nFraud and Permanent Exclusions.\n"),
        (2, "War and nuclear risks are not payable.\n" + FILLER * 2),
    ]
    item = only_chunk_containing(pages, "War and nuclear")

    assert item.section == "Section: Exclusions"
