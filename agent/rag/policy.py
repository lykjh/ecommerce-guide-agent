"""Small, version-pinned policy corpus; local retrieval never reads the legacy product DB."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


POLICY_DIR = Path(__file__).with_name("policies")


class PolicyUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class Clause:
    id: str
    title: str
    text: str


def load_policy(directory=POLICY_DIR):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "published" or manifest["policy_id"] != "MALL-SERVICE-001"
            or manifest["version"] != "v1.0"):
        raise PolicyUnavailable("Policy is withdrawn or has an unsupported publication state")
    effective_at = datetime.fromisoformat(manifest["effective_at"])
    if effective_at.tzinfo is None or effective_at > datetime.now(timezone.utc):
        raise PolicyUnavailable("Policy is not yet effective")
    raw = (directory / "mall-service-v1.md").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise PolicyUnavailable("Policy source changed without a version update")
    text = raw.decode("utf-8").split("## 编制依据与版本说明", 1)[0]
    clauses = []
    chapter = 0
    chapter_title = ""
    title, clause_id, lines = "", "", []

    def flush():
        body = "\n".join(lines).strip().removesuffix("---").strip()
        if body:
            clauses.append(Clause(clause_id, title, body))

    for line in text.splitlines():
        if line.startswith("## 第"):
            flush()
            chapter += 1
            clause_id, title, lines = str(chapter), line[3:], []
            chapter_title = title
        elif line.startswith("### "):
            flush()
            clause_id, title = line[4:].split(" ", 1)
            title = chapter_title + " / " + title
            lines = []
        elif chapter:
            lines.append(line)
    flush()
    if chapter != 16 or not clauses:
        raise PolicyUnavailable("Incomplete policy document")
    return manifest, clauses


def _terms(text):
    text = re.sub(r"\s+", "", text.lower())
    return Counter(text[i:i + 2] for i in range(len(text) - 1)
                   if all(c.isalnum() for c in text[i:i + 2]))


def retrieve(clauses, query, k=10):
    """Character-bigram TF-IDF retrieval for the single Chinese policy document."""
    documents = [_terms(c.title + c.text) for c in clauses]
    frequency = Counter(term for terms in documents for term in terms)
    weights = {term: math.log((len(documents) + 1) / (count + 1)) + 1
               for term, count in frequency.items()}
    query_terms = _terms(query)
    ranked = []
    for clause, terms in zip(clauses, documents):
        overlap = query_terms.keys() & terms.keys()
        if not overlap:
            continue
        score = sum(query_terms[t] * terms[t] * weights[t] ** 2 for t in overlap)
        norm = math.sqrt(sum((n * weights[t]) ** 2 for t, n in terms.items()))
        ranked.append((score / norm, clause))
    return [clause for _, clause in sorted(ranked, key=lambda item: item[0], reverse=True)[:k]]
