from scripts.build_corpus_v4_openculture import SOURCES, get_document_id, metadata_text


def test_openculture_allowlist_contains_only_explicit_public_domain_books():
    assert len(SOURCES) == 1
    source = SOURCES[0]
    assert source["repo"] == "PleIAs/Spanish-PD-Books"
    assert source["revision"] == "001eaf13681f483069361dd82195ce279e12ed63"
    assert source["license"] == "public_domain"
    assert source["license_url"].startswith("https://huggingface.co/datasets/PleIAs/Spanish-PD-Books")
    assert "share" not in source or source["share"] == 1.0


def test_document_id_prefers_stable_source_identifier():
    item = {"identifier": "urn:example:book:42", "title": "Libro"}
    assert get_document_id(item, "spanish_pd_books", 3) == "urn:example:book:42"


def test_metadata_text_keeps_attribution_metadata_but_excludes_body():
    item = {
        "text": "Texto del libro que no debe duplicarse",
        "title": "Título",
        "creator": "Autor",
        "license": "public_domain",
    }
    flattened = metadata_text(item)
    assert "Título" in flattened
    assert "Autor" in flattened
    assert "public_domain" in flattened
    assert "Texto del libro" not in flattened
