from governance.retrieval import BM25Index, chunk


def test_chunking_groups_sentences_and_ids_are_stable():
    passages = chunk("doc", "Doc", "A one. B two. C three.\n\nD four.", sentences_per_passage=2)
    assert [p.id for p in passages] == ["doc#0", "doc#1", "doc#2"]
    assert passages[0].text == "A one. B two."


def test_corpus_loads_titles(index):
    assert {p.title for p in index.passages} >= {"Leave Policy", "Information Security Policy"}


def test_search_ranks_relevant_document_first(index):
    assert index.search("paternity leave days", k=1)[0].passage.doc_id == "leave_policy"
    assert index.search("USB storage laptop", k=1)[0].passage.doc_id == "security"
    assert index.search("hotel limit per night", k=1)[0].passage.doc_id == "expenses"


def test_search_respects_k_and_is_sorted(index):
    hits = index.search("leave days policy", k=3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_unknown_terms_return_nothing(index):
    assert index.search("zzzz qqqq") == []


def test_from_texts():
    idx = BM25Index.from_texts({"a": "Cats sleep a lot.", "b": "Dogs bark loudly."})
    assert idx.search("dogs")[0].passage.doc_id == "b"
