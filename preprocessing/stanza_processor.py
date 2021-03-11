import stanza

from preprocessing.document import Document

class StanzaAnnotator:
    def __init__(self):
        stanza.download('sv')
        self.nlp = stanza.Pipeline(lang='sv')

    def annotateDocument(self, doc: Document):
        doc.stanzaAnnotation = self.nlp(doc.text)