"""Synthetic policy documents for tests.

The real corpus in data/policies/ is gitignored — insurer PDFs are not ours to
redistribute, and CI runners will never have them. Without a synthetic fixture every
PDF-dependent test would skip in the pipeline, making the whole suite decorative.

This generator produces text with the structural features the chunker cares about:
numbered clauses, lettered sub-clauses, and capitalised section headings.
"""

from pathlib import Path

from fpdf import FPDF

PAGES: list[tuple[str, list[str]]] = [
    (
        "DEFINITIONS",
        [
            "1. In this Policy, the following words shall have the meanings assigned "
            "to them below, unless the context otherwise requires.",
            "(a) Accident means a sudden, unforeseen and involuntary event caused by "
            "external, visible and violent means.",
            "(b) Hospital means any institution established for in-patient care and "
            "day care treatment of illness or injuries.",
            "(c) Pre-Existing Disease means any condition, ailment, injury or disease "
            "that is diagnosed by a physician within 48 months prior to the effective "
            "date of the policy issued by the insurer.",
        ],
    ),
    (
        "COVERAGE AND BENEFITS",
        [
            "2. The Company shall indemnify the Insured Person for Medical Expenses "
            "incurred towards Hospitalisation during the Policy Period, up to the Sum "
            "Insured specified in the Schedule.",
            "(a) Room rent, boarding and nursing expenses as provided by the Hospital.",
            "(b) Intensive Care Unit charges, subject to the sub-limits stated in the "
            "Schedule of Benefits.",
            "(c) Surgeon, anaesthetist, medical practitioner and specialist fees, "
            "whether paid directly to the treating doctor or to the Hospital.",
            "3. Pre-hospitalisation Medical Expenses incurred for a period of 30 days "
            "immediately before the date of admission are payable.",
        ],
    ),
    (
        "WAITING PERIODS",
        [
            "4. Pre-Existing Diseases shall be covered only after a waiting period of "
            "36 months of continuous coverage has elapsed since the inception of the "
            "first policy with the Company.",
            "5. A waiting period of 30 days shall apply from the inception of the "
            "Policy, during which no claim shall be payable except those arising from "
            "an Accident.",
            "6. Specified surgical procedures including cataract, hernia and joint "
            "replacement shall be subject to a waiting period of 24 months.",
        ],
    ),
    (
        "EXCLUSIONS",
        [
            "7. The Company shall not be liable to make any payment under this Policy "
            "in respect of any expenses incurred in connection with the following.",
            "(a) Any treatment arising from or traceable to war, invasion, act of "
            "foreign enemy, or warlike operations.",
            "(b) Cosmetic or plastic surgery, unless necessitated by an Accident, "
            "burns, or cancer, and certified as medically necessary.",
            "(c) Expenses related to any admission primarily for enforced bed rest, "
            "evaluation or diagnostic purposes not followed by active treatment.",
            "(d) Dental treatment or surgery of any kind, unless requiring "
            "Hospitalisation and arising from an Accident.",
        ],
    ),
    (
        "CLAIMS PROCEDURE",
        [
            "8. Notice of a claim must be given to the Company within 48 hours of "
            "admission in the case of emergency Hospitalisation, and 72 hours prior to "
            "admission in the case of planned Hospitalisation.",
            "9. All supporting documents relating to the claim must be submitted to "
            "the Company within 15 days from the date of discharge from the Hospital.",
            "10. The Company shall settle or reject a claim within 30 days of receipt "
            "of the last necessary document.",
        ],
    ),
]


def write_synthetic_policy(path: Path, title: str = "SPECIMEN HEALTH INSURANCE POLICY") -> Path:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for heading, paragraphs in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, f"{title}\n{heading}")
        pdf.ln(3)
        pdf.set_font("Helvetica", size=11)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path
