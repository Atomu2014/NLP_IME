edit_cost = {}


def init_edit_cost(editor):
    print 'init edit cost', editor
    edit_cost['a'] = editor['a']
    edit_cost['s'] = editor['s']
    edit_cost['d'] = editor['d']
    edit_cost['t'] = editor['t']


def min_edit_dist(source, target):
    n = len(target)
    m = len(source)

    distance = [[0 for i in range(m + 1)] for j in range(n + 1)]

    for i in range(1, n + 1):
        distance[i][0] = distance[i - 1][0] + edit_cost['a']

    for j in range(1, m + 1):
        distance[0][j] = distance[0][j - 1] + edit_cost['d']

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i][j] = min(distance[i - 1][j] + 1,
                                 distance[i][j - 1] + 1,
                                 distance[i - 1][j - 1] + edit_cost['s'])
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
            delcost = oneago[y] + edit_cost['d']
            addcost = thisrow[y - 1] + edit_cost['a']
            subcost = oneago[y - 1]
            if seq1[x] != seq2[y]:
                subcost += edit_cost['s']
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if x > 0 and y > 0 and seq1[x] == seq2[y - 1] and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]:
                thisrow[y] = min(thisrow[y], twoago[y - 2] + edit_cost['t'])
    return thisrow[len(seq2) - 1]


def dldist_with_op(seq1, seq2):
    oneago = None
    thisrow = [(x, '') for x in range(1, len(seq2) + 1)] + [(0, '')]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [(0, '')] * len(seq2) + [(x + 1, '')]
        for y in xrange(len(seq2)):
            delcost = oneago[y][0] + edit_cost['d']
            addcost = thisrow[y - 1][0] + edit_cost['a']
            subcost = oneago[y - 1][0]
            if seq1[x] != seq2[y]:
                subcost += edit_cost['s']
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
                if twoago[y - 2][0] + edit_cost['t'] < mincost:
                    mincost = twoago[y - 2][0] + edit_cost['t']
                    op = twoago[y - 2][1] + 't'
            thisrow[y] = (mincost, op)
    return thisrow[len(seq2) - 1]


def dldist_with_op_trace(seq1, seq2):
    oneago = None
    thisrow = [(x, '') for x in range(1, len(seq2) + 1)] + [(0, '')]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [(0, '')] * len(seq2) + [(x + 1, '')]
        for y in xrange(len(seq2)):
            delcost = oneago[y][0] + edit_cost['d']
            addcost = thisrow[y - 1][0] + edit_cost['a']
            subcost = oneago[y - 1][0]
            if seq1[x] != seq2[y]:
                subcost += edit_cost['s']
            mincost = min(delcost, addcost, subcost)
            op = ''
            if addcost == mincost:
                op = thisrow[y - 1][1] + 'a<tr>' + seq2[y] + '<op>'
            elif subcost == mincost:
                if seq1[x] != seq2[y]:
                    op = oneago[y - 1][1] + 's<tr>' + seq1[x] + '<sep>' + seq2[y] + '<op>'
                else:
                    op = oneago[y - 1][1]
            elif delcost == mincost:
                op = oneago[y][1] + 'd<tr>' + seq1[x] + '<op>'
            if x > 0 and y > 0 and seq1[x] == seq2[y - 1] and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]:
                if twoago[y - 2][0] + edit_cost['t'] < mincost:
                    mincost = twoago[y - 2][0] + edit_cost['t']
                    op = twoago[y - 2][1] + 't<tr>' + seq1[x] + '<sep>' + seq2[y] + '<op>'
            thisrow[y] = (mincost, op)
    return thisrow[len(seq2) - 1]
