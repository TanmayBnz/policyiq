# ADR 0001: Clause-aware chunking

## Status
Accepted — 2026-09-15

## Context
Insurance policy documents are structured as numbered clauses, lettered sub-clauses
and capitalised section headings. Retrieval operates on chunks, so how a document is
cut determines what can be found and what can be cited.

Fixed-width chunking splits mid-clause. In a policy that is not a cosmetic problem:
separating an obligation from the exclusions that qualify it can produce an answer
stating the opposite of the contract. "We cover hospitalisation expenses" is true
only until the exclusion list attached to it is read.

## Decision
Split on clause and paragraph boundaries, detected by a regex matching numbered
clauses (`1. `), lettered sub-clauses (`(a) `), capitalised headings, and blank-line
paragraph breaks. Merge consecutive segments greedily toward a 1200-character target.
Hard-split only those segments that exceed the target on their own. Carry 150
characters of overlap between chunks within a page, but never across a page boundary.

Overlap stops at page boundaries because chunks inherit the page number they came
from, and that number is surfaced to users as a citation. Carrying text across would
attach one page's words to another page's number — every citation after the boundary
would be subtly wrong, and wrong in a way that only manual checking reveals.

## Consequences
Chunk sizes vary rather than being uniform, which is the intended trade: a chunk
boundary that respects meaning is worth more than one that respects arithmetic.

Page attribution stays exact, which is what makes citations verifiable.

A clause spanning a page break loses the overlap that would otherwise make it
retrievable from both sides, so it may rank slightly worse. Accepted for now, and
revisited if the evaluation on days 7-9 shows it matters.

## Measured
Corpus: 10 documents, 268 pages produces 874 chunks, 3.3 per page.
Chunk size: minimum 95, median 998, maximum 1346 characters against a 1200 target.
No chunk exceeds target plus overlap.

## Review findings, 2026-09-15
Three defects were found in the first implementation while all five of its tests
passed — which says more about the tests than about the code:

1. `text[-overlap_chars:]` returns the entire string when overlap is zero, because
   Python has no negative zero. Zero overlap therefore produced total overlap, and
   each chunk contained all of its predecessors.
2. Segments were merged without a separator, welding clauses together as
   `withdrawn.6. WAITING PERIOD` and corrupting the text the embedder sees.
3. The oversized-clause hard split sat in an `else` branch, so it ran only when no
   chunk was accumulating. A long clause arriving after a short one bypassed it and
   was emitted whole — 16% of chunks exceeded the target, the largest at 2.4x.

The tests now assert properties rather than examples: no chunk exceeds the ceiling,
no chunk contains its predecessor, and merged clauses carry whitespace between them.
Example-based tests passed a broken implementation; property-based ones did not.
