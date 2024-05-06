import unittest

from dolma.core.data_types import Document
from dolma.taggers.part_of_speech import PartOfSpeechTagger

POS_TEST = """
Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, 
thank you very much. 
They were the last people you’d expect to be involved in anything strange or mysterious, 
because they just didn’t hold with such nonsense.
"""

class TestPOSTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = Document(source=__file__, id="0", text=POS_TEST)

        return super().setUp()

    def test_pos(self):
        result = PartOfSpeechTagger().predict(self.doc)
        self.assertEqual(len(result.spans), len(POS_TEST))

