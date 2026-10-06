from policyiq.db import get_pool
from policyiq.retrieval.vector import RetrievedChunk

# DECISION: match ANY of the question's words, not all of them.
#
# The obvious call, websearch_to_tsquery, requires EVERY word. That suits a search box,
# where people type "co-payment arogya", and fails on a full question: "Is there a
# co-payment on claims under Arogya Sanjeevani, and how much is it?" needs one chunk
# containing all of co-payment, claim, Arogya, Sanjeevani and much. Measured on the
# golden set, it put a right page in the top 5 for 7 of 31 questions. Matching any word,
# and letting the ranking in keyword_search sort the many matches, managed 24 of 31.
#
# How the OR is built: to_tsvector reduces the question to its stemmed words with stop
# words removed, unnest lists them, and string_agg joins them with |, the OR operator.
#
# BUGFIX (prototype): this first used to_tsquery('simple', ...) to turn that string into
# a query. to_tsquery parses its input again, and the English parser had produced
# compound words such as 'co-pay', which a second parse turns into a phrase -
# 'co-pay' <-> 'co' <-> 'pay' - demanding three adjacent words where one was meant.
# Casting with ::tsquery reads the string as finished query syntax and leaves the words
# alone.
#
# quote_literal wraps each word in quotes, so a word that looks like query syntax
# (&, !, :) is taken as text. The question itself is a bound parameter and never becomes
# part of the SQL.
#
# A question made only of stop words ("what is it that they were?") has no words left.
# string_agg over nothing is NULL, NULL @@ anything is NULL, and the WHERE clause drops
# every row: no results, not an error.
#
# A module constant so the tests run this exact expression rather than a copy of it.
QUESTION_TO_QUERY = (
    "SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery"
    " FROM unnest(to_tsvector('english', %s))"
)


def keyword_search(question: str, top_k: int) -> list[RetrievedChunk]:
    """Rank chunks by the words they share with the question.

    This exists to catch what vector search misses: exact terms. An embedding captures
    what a passage is about, so "Secure Benefit" and "sum insured restoration" can look
    alike, and a product name or a date barely moves the vector at all. Matching words
    gets those right and gets paraphrase wrong - the two fail differently, which is why
    they are fused (see fusion.py) rather than one replacing the other.

    Scores are Postgres ts_rank values. They are only comparable within one call, not
    with vector similarities - which is exactly why fusion uses ranks and ignores them.
    """
    with get_pool().connection() as conn:
        rows = conn.execute(
            f"""
            WITH q AS (SELECT ({QUESTION_TO_QUERY}) AS query)
            SELECT c.id, c.document_id, d.filename, c.page_number, c.chunk_index,
                   c.content,
                   -- DECISION: ts_rank with normalisation 1, which divides the score
                   -- by 1 + log(length of the text). Without it, long chunks win by
                   -- containing more words. Measured: 24/31 either way, and MRR 0.583
                   -- against 0.579, so this is a principled choice, not a proven one.
                   --
                   -- ts_rank is NOT BM25, the ranking most search engines use. BM25
                   -- weights a rare word above a common one (inverse document
                   -- frequency); ts_rank does not, so "policy" counts as much as
                   -- "cataract". Dropping words that appear in many chunks was tried
                   -- as a stand-in and made things worse at every cut-off tried (a
                   -- tenth, a quarter, half of all chunks): the words it dropped, such
                   -- as "waiting" and "hospitalisation", were exactly the ones the
                   -- questions were about. (No percent sign in this SQL except the two
                   -- placeholders, comments included: psycopg reads every one as a
                   -- placeholder, and an earlier draft of this comment failed every
                   -- query.)
                   --
                   -- ts_rank_cd, which rewards the words appearing close together,
                   -- scored 14/31 and was dropped.
                   ts_rank(c.content_tsv || d.filename_tsv, q.query, 1) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            CROSS JOIN q
            -- The filename's words count as part of every chunk of that document; see
            -- sql/002_keyword_search.sql for why, and why this cannot use an index.
            WHERE (c.content_tsv || d.filename_tsv) @@ q.query
            -- Same total-order tiebreaker as vector search. Ties are common here:
            -- nine documents share standard clauses, so identical text scores
            -- identically.
            ORDER BY score DESC, c.id
            LIMIT %s
            """,
            (question, top_k),
        ).fetchall()

    return [
        RetrievedChunk(
            chunk_id=r[0], document_id=r[1], filename=r[2], page_number=r[3],
            chunk_index=r[4], content=r[5], score=float(r[6]),
        )
        for r in rows
    ]
