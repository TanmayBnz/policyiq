"""python -m policyiq.evaluation

  run          score the golden set against the ingested corpus
  gold build   find each answer's pages from its evidence pattern (needs the PDFs)
  gold check   report cases whose recorded pages no longer match their evidence

Run from the host with DATABASE_URL and OLLAMA_BASE_URL pointing at localhost; the
committed .env addresses the containers by their compose names.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import policyiq.answer as answer_module
from policyiq.config import settings
from policyiq.evaluation import gold
from policyiq.evaluation.cases import DEFAULT_QUESTIONS, load_cases, save_cases
from policyiq.evaluation.runner import CaseResult, run

DEFAULT_RESULTS = Path("evaluation/results")


def _print_row(result: CaseResult) -> None:
    if result.expect == "answer":
        rank = next((i for i, c in enumerate(result.retrieved, 1) if c["gold"]), None)
        retrieval = f"rank {rank}" if rank else "MISS  "
    else:
        retrieval = "  -   "
    verdict = {None: "", True: "pass", False: "FAIL"}[result.passed]
    reason = "; ".join(result.failures)
    print(f"  {result.id:34} {retrieval:7} {verdict:5} {reason}", flush=True)


def _use_prompt(path: Path) -> None:
    """Swap the prompt for this run only - the way to compare wordings on the whole set
    instead of on the handful of questions that prompted the change."""
    template = path.read_text()
    for field in ("{context}", "{question}"):
        if field not in template:
            sys.exit(f"{path} has no {field} placeholder")
    answer_module.PROMPT_TEMPLATE = template


def cmd_run(args) -> None:
    cases = load_cases(args.questions)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    if args.prompt:
        _use_prompt(args.prompt)

    mode = "retrieval only" if args.retrieval_only else (
        "answers, warm model (not repeatable)" if args.warm else "answers, cold model per question")
    print(f"{len(cases)} cases, top_k={args.top_k}, {mode}")
    report = run(cases, args.top_k, generate=not args.retrieval_only, cold=not args.warm,
                 progress=_print_row)
    report["configuration"]["label"] = args.label
    report["configuration"]["prompt_file"] = str(args.prompt) if args.prompt else None

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{stamp}-{args.label}.json" if args.label else f"{stamp}.json"
    path = args.out / name
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n" + json.dumps(report["summary"], indent=2))
    print(f"\n{report['seconds']}s, report written to {path}")


def cmd_gold(args) -> None:
    cases = load_cases(args.questions)
    corpus = gold.load_corpus_pages(args.corpus)
    if not corpus:
        # Loud, not a skip: checking a golden set against nothing would report success.
        sys.exit(f"no PDFs in {args.corpus}")

    if args.action == "build":
        built = gold.build(cases, corpus)
        save_cases(built, args.questions)
        for case in built:
            pages = sum(len(s.pages) for s in case.sources)
            print(f"  {case.id:34} {len(case.sources):2} documents, {pages:3} pages")
        return

    problems = gold.check(cases, corpus)
    for problem in problems:
        print(problem)
    print(f"{len(cases)} cases checked, {len(problems)} problems")
    sys.exit(1 if problems else 0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m policyiq.evaluation")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    run_parser.add_argument("--retrieval-only", action="store_true",
                            help="skip generation: seconds instead of minutes")
    run_parser.add_argument("--warm", action="store_true",
                            help="keep the model loaded: ~3.5x faster, answers not repeatable")
    run_parser.add_argument("--prompt", type=Path, help="alternative prompt template")
    run_parser.add_argument("--label", default="", help="suffix for the report filename")
    run_parser.add_argument("--only", nargs="+", metavar="ID", help="run just these cases")
    run_parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    run_parser.set_defaults(func=cmd_run)

    gold_parser = sub.add_parser("gold")
    gold_parser.add_argument("action", choices=["build", "check"])
    gold_parser.add_argument("--corpus", type=Path, default=Path("data/policies"))
    gold_parser.set_defaults(func=cmd_gold)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
