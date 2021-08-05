import sys
filename = sys.argv[1]

corefLines = []
with open(filename) as f:
    for line in f.readlines():
        if line.startswith('Coreference:'):
            corefLines.append(line)
conll = 0
cnt = 0
for idx, line in enumerate(corefLines):
    if idx == 2 or idx == 4: # skip ceafm and BLANC
        continue
    for i, c in enumerate(line):
        if c == '%':
            cnt+=1
            if cnt%3 == 0:
                cnt = 0
                conll+= float(line[i-5:i])
            print(' & ', end = '')
            print(line[i-5:i], end='')
conll = round(conll/3, 2)
print(f' & {conll}')