from pytokens import PYKEYWORDS

import ast
import codegen
import re

## TYPES #######################################################################
# types
TYPES = {}
# list and tuple map to the same type
TYPES["seq"]   = "SEQ"
TYPES["dict"]  = "DICT"
TYPES["class"] = "CLASS"
TYPES["func"]  = "FUNC"
TYPES["gen"]   = "GEN" # can be either string or num
TYPES["any"]   = "ANY" # can't determine
TYPES["num"]   = "NUM"
TYPES["str"]   = "STR"
TYPES["funcdef"] = "FUNCDEF"
TYPES["classdef"] = "CLASSDEF"

## AST NodeVisitors ############################################################

class CallVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_Call(self, node):

        try:

            if isinstance(node.func, ast.Name):

                identifier = node.func.id
                if self.d.get(identifier) is None and identifier not in PYKEYWORDS:
                    self.d[identifier] = TYPES["func"]

            if isinstance(node.func, ast.Attribute):

                # the deepest attribute is a function
                if isinstance(node.func, ast.Name):
                    identifier1 = node.func.id
                    if self.d.get(identifier1) is None and identifier1 not in PYKEYWORDS:
                        self.d[identifier1] = TYPES["class"]

                identifier2 = node.func.attr
                if self.d.get(identifier2) is None and identifier2 not in PYKEYWORDS:
                    self.d[identifier2] = TYPES["func"]


        except:
            pass

class AttributeVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_Attribute(self, node):

        try:

            if isinstance(node.value, ast.Attribute):
                # nice ; call recursively
                self.visit_Attribute(node.value)

            if isinstance(node.value, ast.Name):
                identifier1 = node.value.id
                if self.d.get(identifier1) is None and identifier1 not in PYKEYWORDS:
                    self.d[identifier1] = TYPES["class"]

            identifier2 = node.attr
            if self.d.get(identifier2) is None and identifier2 not in PYKEYWORDS:
                # if this guy's parent is an attribute; then this is a class
                if isinstance(node.parent, ast.Attribute):
                    self.d[identifier2] = TYPES["class"]
                else:
                    # likely a variable
                    self.d[identifier2] = TYPES["any"] # can easily be a num or str !
        except:
            pass

class SubscriptVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_Subscript(self, node):

        ids = []

        has_int_key = False
        has_str_key = False

        try:
            if (node.value.id is not None and node.value.id not in PYKEYWORDS):
                ids.append(node.value.id)

            if (isinstance(node.slice.value, ast.Num)):
                has_int_key = True

            if (isinstance(node.slice.value, ast.Str)):
                has_str_key = True

        except:
            pass

        if len(ids) == 1:

            identifier = ids[0]

            # if identifier already filled; move on
            if self.d.get(identifier) is None:

                # can make confident predictions
                if has_int_key:
                    # it is likely to be a list
                    self.d[identifier] = TYPES["seq"]

                if has_str_key:
                    # it is likely to be a dict
                    self.d[identifier] = TYPES["dict"]

class FunctionDefVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_FunctionDef(self, node):

        try:
            name = node.name
            self.d[name] = TYPES["funcdef"]
        except:
            pass

class ClassDefVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_ClassDef(self, node):

        try:
            name = node.name
            self.d[name] = TYPES["classdef"]
        except:
            pass

class KWVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_keyword(self, node):

        try:
            if node.arg not in PYKEYWORDS:
                argname = node.arg
                self.d[argname] = TYPES["any"] # really could be any thing
        except:
            pass

class StrVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_Str(self, node):

        try:
            if node.s not in PYKEYWORDS:
                s = node.s
                self.d[s] = TYPES["str"]
        except:
            pass

class NumVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_Num(self, node):

        try:
            n = node.n
            self.d[n] = TYPES["num"]
        except:
            pass

class AliasVisitor(ast.NodeVisitor):

    # constructor
    def __init__(self, d_ = {}):
        self.d = d_

    def visit_alias(self, node):

        try:

            if node.name not in PYKEYWORDS:
                name = node.name
                self.d[name] = TYPES["str"]
        except:
            pass



## Helpers and exports #########################################################


def get_unique_ref_types(ref_types):

    # create a reference-type map; for example if there are 2 referenceable,
    # "a" and "b" whose types are "ANY", then map "a" and "b" to "ANY_1" and
    # "ANY_2"
    ref_type_unique = {}

    # count number of times each type appears
    type_f = {}
    for ref, t in ref_types.iteritems():

        if type_f.get(t) is None:
            # first occurance
            type_f[t] = 0
        else:
            type_f[t] = type_f[t] + 1

        c = type_f[t]

        # including this we have seen this type c #times
        ref_type_unique[ref] = t + "_" + str(c)

    return ref_type_unique

def get_referencables(code):

    # parse code
    root = ast.parse(code)

    refable = []

    # walk the ast
    for node in ast.walk(root):

        # can defenitely reference a identifier from query
        try:
            ident = node.id
            if ident not in PYKEYWORDS:
                refable.append(ident)
        except:
            pass

        # can reference strings
        try:
            s = node.s
            if s not in PYKEYWORDS:
                refable.append(s)
        except:
            pass

        # can reference numbers
        try:
            n = node.n
            refable.append(n)
        except:
            pass

        # work with aliases
        try:
            if isinstance(node, ast.alias):
                s = node.name
                if s not in PYKEYWORDS:
                    refable.append(s)

                s = node.asname
                if isinstance(s, str) and s not in PYKEYWORDS:
                    refable.append(s)

        except:
            pass

    return set(refable)

def unique_ref_types(ref_types):

    # type counter
    type_counter = {}

    for k, v in ref_types.iteritems():

        if type_counter.get(v) is None:
            type_counter[v] = 0
        else:
            type_counter[v] = type_counter[v] + 1

        ref_types[k] = v + "_" + str(type_counter[v])

    return ref_types

def guess_types(code, query):

    # get all referencables
    referencables = get_referencables(code)

    root = None

    # the code must be parsable
    try:
        root = ast.parse(code)
    except:
        assert (False and "code non parsable")

    ref_types = {}

    # add parent information to all nodes in ast tree
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            # add weak reference
            child.parent = node

    # figure out tuples/lists/dict to the best we can
    subscript_v = SubscriptVisitor(ref_types)
    subscript_v.visit(root)
    ref_types = subscript_v.d

    # figure out functions
    call_v = CallVisitor(ref_types)
    call_v.visit(root)
    ref_types = call_v.d

    # figure out class objects, class members
    attribute_v = AttributeVisitor(ref_types)
    attribute_v.visit(root)
    ref_types = attribute_v.d

    # figure out strings
    string_v = StrVisitor(ref_types)
    string_v.visit(root)
    ref_types = string_v.d

    # figure out numbers
    num_v = NumVisitor(ref_types)
    num_v.visit(root)
    ref_types = num_v.d

    # figure out all aliases
    alias_v = AliasVisitor(ref_types)
    alias_v.visit(root)
    ref_types = alias_v.d

    # figure out all function definitions
    fdef_v = FunctionDefVisitor(ref_types)
    fdef_v.visit(root)
    ref_types = fdef_v.d

    # figure out all class definitions
    cdef_v = ClassDefVisitor(ref_types)
    cdef_v.visit(root)
    ref_types = cdef_v.d

    # figure out all keywords
    kw_v = KWVisitor(ref_types)
    kw_v.visit(root)
    ref_types = kw_v.d

    # figure out if there is any string in query !! regex :P
    matches = []
    # double quotes
    matches = matches + re.findall(r'\"(.+?)\"', query)
    # single quotes
    matches = matches + re.findall(r'\'(.+?)\'', query)
    # back ticks :P `
    matches = matches + re.findall(r'\`(.+?)\`', query)
    # these matches are strings
    for m in matches:
        if m not in ref_types.keys():
            ref_types[m] = TYPES["str"]

    # all referencable that we were not able to figure out is any type
    for refable in referencables:
        if refable not in ref_types.keys():
            ref_types[refable] = TYPES["any"]

    # make types unique !!; dont have DICT for 2 variables; instead have DICT_0
    # and DICT_1
    ref_types = unique_ref_types(ref_types)

    return ref_types

