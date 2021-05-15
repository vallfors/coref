from preprocessing.document import Document, Mention
from preprocessing.config import Config

from typing import List
import operator

# Returns an ordered list of candidate antecedents for a mention,
# in the order they should be considered in.
def getCandidateAntecedents(config: Config, doc: Document, mention: Mention, maxSentenceDistance: int) -> List[Mention]:
    x = list(doc.predictedMentions.values())
    x.sort(key=operator.attrgetter('startPos'))
    antecedents = []
    for idx, a in enumerate(x):
        if a.stanzaSentence >= mention.stanzaSentence:
            break
        if mention.stanzaSentence - a.stanzaSentence >= maxSentenceDistance:
            continue
        antecedents.append(a)
    

    # For mentions in the same sentence, we want to prioritize subjects
    subjects = []
    nonSubjects = []
    sameSentence = []
    while idx < len(x) and x[idx].stanzaSentence == mention.stanzaSentence:
        if x[idx].startPos >= mention.startPos:
            break
        if x[idx].features.isSubject:
            subjects.append(x[idx])
        else:
            nonSubjects.append(x[idx])
        sameSentence.append(x[idx])
        idx += 1
    # Later antecedents should come first.
    antecedents.reverse()

    if config.orderAntecedentsBySubject:
        antecedents = subjects + nonSubjects + antecedents
    else:
        antecedents = sameSentence + antecedents
    
    return antecedents