import nltk

from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger
from ..core.utils import split_words
from length import WhitespaceLengthV1

nltk.download('averaged_perceptron_tagger') # required for the tagging 

@add_tagger("part_of_speech")
class PartOfSpeechTagger(WhitespaceLengthV1):
    def predict(self, doc: Document) -> DocResult:
        spans = [
            Span(start=p.start, end=p.end, type="part_of_speech", score=nltk.pos_tag(p.text)[1])
            for p in split_words(doc.text)
        ]

        # return the span wrapped in a DocResult object
        return DocResult(doc=doc, spans=[span])