import nltk

from dolma.core.data_types import DocResult, Document, Span
from ..core.taggers import BaseTagger
from ..core.registry import TaggerRegistry
from ..core.utils import split_words

nltk.download('averaged_perceptron_tagger') # required for the tagging 

@TaggerRegistry.add("part_of_speech")
class PartOfSpeechTagger():
    def predict(self, doc: Document) -> DocResult:
        spans = [
            Span(start=p.start, end=p.end, type=nltk.pos_tag([p.text][0][1])) # return only POS tag
            for p in split_words(doc.text)
        ]
        # return the span wrapped in a DocResult object
        return DocResult(doc=doc, spans=spans)




