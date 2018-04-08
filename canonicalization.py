
from file_utils import file_contents

from pytokens        import PYKEYWORDS
from type_simulation import TYPES

from nlparser import QueryParser
from type_simulation import guess_types, get_referencables
from canon_utils import fetch_queries_codes, parsable_code, reproducable_code, give_me_5

import ast
# import codegen
import re
import math

## Globals #####################################################################

ANNOT = "./gendata/all.anno"
CODE  = "./gendata/all.code"

qparser = QueryParser()

## Utilities ###################################################################

# replace stuff in query that can be directly mapped
def canonicalize_query_part1(q, ref_type):

    # for every string in ref types, check if we have a match in query
    keys_ordered = [] if ref_type.keys() is None else sorted(ref_type.keys(), key = (lambda x: len(str(x))), reverse = True)

    for k in keys_ordered:

        if ref_type[k].startswith(TYPES['str']):

            # double quoted
            q = q.replace(" \"" + k + "\" ", " " + ref_type[k] + " ")
            q = q.replace("\"" + k + "\" ",  " " + ref_type[k] + " ")
            q = q.replace(" \"" + k + "\"",  " " + ref_type[k] + " ")

            # single quoted
            q = q.replace(" \'" + k + "\' ", " " + ref_type[k] + " ")
            q = q.replace("\'" + k + "\' ",  " " + ref_type[k] + " ")
            q = q.replace(" \'" + k + "\'",  " " + ref_type[k] + " ")

    # represent "class . attribute" as class.attribute
    # modify direct mapping
    tokens = []
    for t in q.split(" "):

        if t not in PYKEYWORDS and ref_type.get(t.strip()) is not None:
            tokens.append(ref_type[t.strip()])
        else:
            tokens.append(t)

    q = " ".join(tokens)

    # represent "class.attribute" as "class . attribute"
    q = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1 . \2', q)

    # modify direct mapping
    tokens = []
    for t in q.split(" "):

        if t not in PYKEYWORDS and ref_type.get(t.strip()) is not None:
            tokens.append(ref_type[t.strip()])
        else:
            tokens.append(t)

    # new query
    q = " ".join(tokens)

    return q

# replace stuff in query based on pos tag of each token
def canonicalize_query_part2(q_pos, ref_type):

    canon_query_tokens = []
    for t, pos in q_pos:

        # if the token belongs to a closed class; dont bother working it
        if pos == "IN" or                                                  \
           pos == "DT" or pos == "PDT" or pos == "WDT" or                  \
           pos == "CC" or                                                  \
           pos == "PRP" or pos == "PRP$" or pos == "WP" or pos == "WP$":
            # keep token as-is
            canon_query_tokens.append(t)

        else:

            # do we have a type for this noun/symbol ?
            if ref_type.get(t.strip()) is not None:
                canon_query_tokens.append(ref_type[t.strip()])

            elif t.isdigit() and ref_type.get(int(t)) is not None:
                canon_query_tokens.append(ref_type[int(t)])

            else:
                canon_query_tokens.append(t)

    canon_query = " ".join(canon_query_tokens)

    return canon_query

def canonicalize_code(code, ref_type):

    try:
        ast.parse(codegen.to_source(ast.fix_missing_locations(ast.parse(code.strip()))))
    except:
        return ""

    # replace all identifiers in the code by TYPES as guessed
    # parse code
    root = ast.parse(code)

    # walk the ast
    for node in ast.walk(root):

        # fix all identifiers
        try:
            # modify identifier with type
            if ref_type.get(node.id) is not None:
                node.id = ref_type[node.id]
        except:
            pass

        ## fix all attributes
        try:
            if ref_type.get(node.attr) is not None:
                node.attr = ref_type[node.attr]
        except:
            pass

        #3 fix all strings
        try:
            if isinstance(node, ast.Str):
                if ref_type.get(node.s) is not None:
                    node.s = ref_type[node.s]
        except:
            pass

        # fix all numbers
        try:
            if isinstance(node, ast.Num):
                if ref_type.get(node.n) is not None:
                    node.n = ref_type[node.n]
        except:
            pass

        # fix all alias
        try:
            if isinstance(node, ast.alias):
                if ref_type.get(node.name) is not None:
                    node.name = ref_type[node.name]
        except:
            pass

        # fix all function definitions
        try:
            if isinstance(node, ast.FunctionDef):
                if ref_type.get(node.name) is not None:
                    node.name = ref_type[node.name]
        except:
            pass


        # fix all class definitions
        try:
            if isinstance(node, ast.ClassDef):
                if ref_type.get(node.name) is not None:
                    node.name = ref_type[node.name]
        except:
            pass

        # fix all kword definitions
        try:
            if isinstance(node, ast.keyword):
                if ref_type.get(node.arg) is not None:
                    node.arg = ref_type[node.arg]
        except:
            pass

    # looks like codegen is bugggy !! hence the patchwork
    try:
        # can we parse and unparse the code ??
        ast.parse(codegen.to_source(ast.fix_missing_locations(root)))

        code = codegen.to_source(ast.fix_missing_locations(root))

        # code gen does a pretty bad job at generating code :@
        # it generated - raiseFUNC('STR' % ANY)
        # while it should be, raise FUNC('STR' % ANY)

        # make a space in code when such things happen
        for t in TYPES.values():
            code = re.sub(r'([a-zA-Z]+)' + t, r'\1' + " " + t, code)

        # check if we can parse it
        ast.parse(codegen.to_source(ast.fix_missing_locations(root)))

        return code

    except:
        return ""

def batch_deep_phrases(queries):

    n = math.ceil(float(len(queries)) / 18.0)
    n = int(n)

    parts = [ queries[x * n : (x + 1) * n] for x in range(18) ]

    # remove empty parts
    parts = filter(lambda part: part != [], parts)

    # sum of part lengths
    pl_sum = 0
    for part in parts:
        pl_sum = pl_sum + len(part)
    assert (len(queries) == pl_sum)

    # Do parsing for every batch
    qparser = QueryParser()

    result = []

    for part in parts:
        print "Parsing queries ", parts.index(part) * n, " Onwards - ", len(queries)
        result = result + qparser.deep_phrases(part)

    assert (len(result) == len(queries))

    return result

def tag_pos(queries):

    phrase_pos = batch_deep_phrases(queries)

    assert (len(phrase_pos) == len(queries))

    # get pos alone
    tok_pos = map(lambda ppos: ppos[1], phrase_pos)

    return tok_pos

# unnormalize numbers
def unnormalize_numbers(reftypes_queries):

    ref_types = []

    for ref_type, query in reftypes_queries:

        for k, v in ref_type[1].iteritems():
            if v not in query and v.startswith(TYPES["num"]):
                ref_type[1][k] = k

        ref_types.append(ref_type)

    assert (len(ref_types) == len(reftypes_queries))

    return ref_types

def canonicalize_bunch(queries, codes):

    assert (len(queries) == len(codes))

    # assign ids to queries, codes for tracking
    id_queries_codes = [ (i, queries[i], codes[i]) for i in range(len(queries)) ]

    # get all ref types
    ref_types = map(lambda d: (d[0], guess_types(d[2], d[1])), id_queries_codes)

    # Canonicalizeing all queries; direct mapping part 1
    id_queries_codes = map(lambda ref_type, d: (d[0],                                       \
                                                canonicalize_query_part1(d[1], ref_type[1]),\
                                                d[2]), ref_types, id_queries_codes)

    # batch process all queries for pos after canonicalization part 1
    canon_p1_queries = map(lambda d: d[1], id_queries_codes)
    id_qpos = tag_pos(canon_p1_queries)
    # update id_queries_codes
    id_queries_codes = map(lambda qpos, d: (d[0], qpos, d[2]), id_qpos, id_queries_codes)

    # Canonicalize all queries; pos_info part 2
    id_queries_codes = map(lambda ref_type, d: (d[0],                                          \
                                                 canonicalize_query_part2(d[1], ref_type[1]),   \
                                                 d[2]), ref_types, id_queries_codes)

    # change in reftypes; ugly;
    canon_p2_queries = map(lambda d: d[1], id_queries_codes)
    assert (len(canon_p2_queries) == len(ref_types))
    # if something is a number and does not appear in the query; leave it alone
    ref_types = unnormalize_numbers(zip(ref_types, canon_p2_queries))

    # finally canonicalize code
    id_queries_codes = map(lambda ref_type, d: (d[0], d[1], canonicalize_code(d[2], ref_type[1])), \
                                                ref_types, id_queries_codes)

    # get canonicalized queries
    canon_queries = map(lambda d: d[1], id_queries_codes)
    canon_codes   = map(lambda d: d[2], id_queries_codes)
    ref_types     = map(lambda rt: rt[1], ref_types)

    return ref_types, canon_queries, canon_codes


## Main ########################################################################

if __name__ == '__main__':

    queries, codes = fetch_queries_codes(ANNOT, CODE)
    queries_codes  = zip(queries, codes)

    queries_codes = map(lambda q_c: (q_c[0], parsable_code(q_c[1])), queries_codes)
    queries_codes = map(lambda q_c: (q_c[0], reproducable_code(q_c[1])), queries_codes)
    queries_codes = map(lambda q_c: (q_c[0], parsable_code(q_c[1])), queries_codes)
    queries_codes = filter(lambda q_c: q_c[1] != "", queries_codes)

    queries, codes = zip(*queries_codes)

    #give_me_5(queries, "queries")

    #give_me_5(codes, "codes")

    # have only 5
    queries = queries[4:5]
    codes   = codes[4:5]

    assert (len(queries) == len(codes))

    ref_types, canon_queries, canon_codes = canonicalize_bunch(queries, codes)

    assert (len(queries) == len(codes))

    n = len(queries)

    for idx in range(0, n):

        q = queries[idx]

        c = codes[idx]

        cq = canon_queries[idx]

        cc = canon_codes[idx]

        print "-" * 60

        print "\n"

        print "index - ", idx

        print "\n"

        print "QUERY - "

        print "\n"

        print q

        print "\n"

        print cq

        print "\n"

        print "CODE - "

        print "\n"

        print c

        print "\n"

        print cc

        print "\n"

        x = raw_input()
