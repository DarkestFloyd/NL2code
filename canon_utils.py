
from file_utils import file_contents

import ast
import codegen

## Common utilities ############################################################

def fetch_queries_codes(annot_file, code_file):

    # get all queries
    contents = file_contents(annot_file)
    examples = contents.split("\n\n")
    examples = examples[:-1]
    annots = []

    for ex in examples:

        parts = ex.split("\n")
        parts = filter(lambda x: x.strip() != "", parts)
        assert (len(parts) == 2)
        assert (parts[0].startswith("example"))
        eid = int(parts[0].split(" ")[1])
        query = parts[1]
        annots.append(query)


    # get all code
    contents = file_contents(code_file)
    examples = contents.split("\n\n")
    examples = examples[:-1]
    codes = []

    for ex in examples:

        parts = ex.split("\n")
        parts = filter(lambda x: x.strip() != "", parts)
        assert (len(parts) >= 2)
        assert (parts[0].startswith("example"))
        eid = int(parts[0].split(" ")[1])
        code = "\n".join(parts[1:])
        codes.append(code)

    return annots, codes

def parsable_code(code):

    try:
        # try to parse
        ast.parse(code)
        return code
    except:
        pass

    try:
        # try to parse with pass statement
        code = code + "\n" + "\tpass"
        ast.parse(code)
        return code
    except:
        pass

    # not parsable
    return ""

def reproducable_code(code):

    try:
        ast.parse(codegen.to_source(ast.fix_missing_locations(ast.parse(code.strip()))))
        return code
    except:
        pass

    try:
        # try to reproduce with pass statement
        code = code + "\n" + "\tpass"
        ast.parse(codegen.to_source(ast.fix_missing_locations(ast.parse(code.strip()))))
        return code
    except:
        pass

    # not reproducable
    return ""


def give_me_5(seq, desc = ""):

    print "\n\n"

    print desc + " - \n\n"

    for s in seq[:5]:
        print s
