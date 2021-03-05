# This file contains functionality to read coreference data in the CoNLL-2011 format used in SUC-CORE.

from typing import List

class CoNLLRow:
    documentId: str
    partNumber: int
    wordNumber: int
    form: str # Word
    lemma: str 
    cpostag: str
    postag: str
    feats: str
    head: int
    deprel: str
    entity: str
    coreference: str

class CoNLLFile:
    fileName: str
    rows: List[CoNLLRow]

    def __init__(self, fileName, rows):
        self.fileName = fileName
        self.rows = rows


# Takes file in CoNLL-2011 format (not CoNLL-u) and reads and stores as an CoNLLFile object
def loadFromFile(filename: str) -> CoNLLFile:
    rows = []
    file = open(filename)
    for line in file.readlines():
        columns = line.split()
        # Sometimes there is a space instead of _ in the word/form column, causing the split to go wrong.
        while len(columns) > 12:
            columns[3] += " " + columns[4]
            columns.pop(4)
        # Some lines are empty, or the beginning or end row
        if len(columns) < 12:
            continue
        row = CoNLLRow()
        row.documentId = columns[0]
        row.partNumber = columns[1]
        row.wordNumber = columns[2]
        row.form = columns[3]
        row.lemma = columns[4]
        row.cpostag = columns[5]
        row.postag = columns[6]
        row.feats = columns[7]
        row.head = columns[8]
        row.deprel = columns[9]
        row.entity = columns[10]
        row.coreference = columns[11]
        rows.append(row)
    return CoNLLFile(filename, rows)

def main():
    obj = loadFromFile('./data/suc-core-conll/aa05.conll')

if __name__ == "__main__":
    main()