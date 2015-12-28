def insertCost(a):
    return 1


def deleteCost(a):
    return 1


def substCost(a, b):
    return 1


def min_edit_dist(source, target):
    n = len(target)
    m = len(source)

    distance = [[0 for i in range(m + 1)] for j in range(n + 1)]

    for i in range(1, n + 1):
        distance[i][0] = distance[i - 1][0] + insertCost(target[i - 1])

    for j in range(1, m + 1):
        distance[0][j] = distance[0][j - 1] + deleteCost(source[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i][j] = min(distance[i - 1][j] + 1,
                                 distance[i][j - 1] + 1,
                                 distance[i - 1][j - 1] + substCost(source[j - 1], target[i - 1]))
    return distance[n][m]


def dameraulevenshtein(seq1, seq2):
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if x > 0 and y > 0 and seq1[x] == seq2[y - 1] and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]:
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


def dldist_with_op(seq1, seq2):
    oneago = None
    thisrow = [(x, '') for x in range(1, len(seq2) + 1)] + [(0, '')]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [(0, '')] * len(seq2) + [(x + 1, '')]
        for y in xrange(len(seq2)):
            delcost = oneago[y][0] + 1
            addcost = thisrow[y - 1][0] + 1
            subcost = oneago[y - 1][0] + (seq1[x] != seq2[y])
            mincost = min(delcost, addcost, subcost)
            op = ''
            if addcost == mincost:
                op = thisrow[y - 1][1] + 'a'
            elif subcost == mincost:
                if seq1[x] != seq2[y]:
                    op = oneago[y - 1][1] + 's'
                else:
                    op = oneago[y - 1][1]
            elif delcost == mincost:
                op = oneago[y][1] + 'd'
            if x > 0 and y > 0 and seq1[x] == seq2[y - 1] and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]:
                if twoago[y - 2][0] + 1 < mincost:
                    mincost = twoago[y - 2][0] + 1
                    op = twoago[y - 2][1] + 't'
            thisrow[y] = (mincost, op)
    return thisrow[len(seq2) - 1]


def count_ops():
    with open('raw/last_as_label', 'r') as fin:
        with open('raw/edit.log', 'w') as fout:
            for line in fin:
                line = line.strip().split()
                edit = dldist_with_op(line[1], line[2])
                fout.write(str(edit[0]) + '\t' + edit[1] + '\n')
