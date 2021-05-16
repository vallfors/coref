
from typing import List

class Features:
    headWord: int # Stanza id of the headword of the mention.
    upos: str # Part-of-speach in universal dependencies format
    definite: str # IND, DEF, UNKNOWN
    gender: str # Grammatical gender: UTR, NEU, NEU/UTR, UNKNOWN
    number: str # SIN, PLU, UNKNOWN
    headWordDeprel: str
    headWordLemma: str
    nerTag: str
    animacy: str # ANIMATE, INANIMATE, UNKNOWN
    naturalGender: str # MALE, FEMALE, UNKNOWN
    isSubject: bool 
    isObject: bool # Direct or indirect object
    headWordHead: int # Id of the head of the headword
    person: str # first, second, third or UNKNOWN

class ClusterFeatures:
    nonStopWords: List[str]
    headWords: List[str]