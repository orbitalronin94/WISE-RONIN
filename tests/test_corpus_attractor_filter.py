import unittest

from scripts.filter_corpus_attractors import compile_terms, normalize


class AttractorFilterTests(unittest.TestCase):
    def setUp(self):
        self.patterns = compile_terms([
            "contexto documental", "democracia", "música", "deformación", "deformaciones"
        ])

    def hits(self, text):
        value = normalize(text)
        return [term for term, pattern in self.patterns if pattern.search(value)]

    def test_phrase_is_blocked(self):
        self.assertIn("contexto documental", self.hits("Contexto documental: texto"))

    def test_accent_and_case_are_ignored(self):
        self.assertIn("música", self.hits("MUSICA y MUSÍCA"))

    def test_whole_word_boundary(self):
        self.assertEqual([], self.hits("democraciometro"))

    def test_unrelated_document_passes(self):
        self.assertEqual([], self.hits("La fotosíntesis transforma energía luminosa."))

    def test_deformation_variant(self):
        self.assertIn("deformaciones", self.hits("deformaciones del material"))


if __name__ == "__main__":
    unittest.main()
