# Utilities and main function for figuring the phrase maps and pos tags of a query

# This file is a wrapper around the stanford parser
# code written with template from https://stackoverflow.com/questions/13883277/stanford-parser-and-nltk

import os
from nltk.parse import stanford
from nltk.translate.ribes_score import position_of_ngram
import nltk

os.environ['STANFORD_PARSER'] = "./"
os.environ['STANFORD_MODELS'] = "./"

class QueryParser:

    # parser
    parser_ = ""

    # constructor
    def __init__(self):

        # global variable; initialize parser once
        self.parser = stanford.StanfordParser(model_path = "./englishPCFG.ser.gz")

    @staticmethod
    def recurse(s, d, sentence = None):

        if not isinstance(s, nltk.Tree):
            return d

        if (s.label() != "NP" and s.label() != "VP" and s.label().endswith("P")):

            if sentence is not None:

                word_lst = s.leaves()
                sent_part = " ".join(word_lst)
                # figure out the position
                p = position_of_ngram(tuple(sent_part.split(" ")), sentence.split(" "))

                d[(p, p + len(s.leaves()) - 1, sent_part.encode("ascii", "ignore"))] = s.label().encode("ascii", "ignore")

            ## this is neither a NP nor a VP; but is a Phrase none the less
            #words = " ".join(s.leaves()).encode("ascii", "ignore")
            #d[words] = s.label().encode("ascii", "ignore")
            ## return immediately
            return d

        # not an interesting phrase; lets go on
        for t in s:
            d = QueryParser().recurse(t, d, sentence)

        return d

    @staticmethod
    def map_phrase(canon_query, detected_phrases):

        # tokens on canon query
        qtokens = canon_query.split(" ")

        n = len(qtokens)
        assert (n > 0)

        # mapper
        mapper = [""] * n

        for tup, ptype in detected_phrases.iteritems():

            spos = tup[0]
            epos = tup[1]
            # mapped words
            map_words = tup[2].split(" ")

            # sanity check
            assert (spos <= epos)
            assert (spos >= 0)
            assert (spos < n)
            assert (epos >= 0)
            assert (epos < n)

            # make sure that within the range specified by spos and epos we have
            # map words in qtokens
            for i in range(spos, epos + 1):

                mapper[i] = ptype

                assert (qtokens[i] == map_words[i - spos])

        return mapper

    # give a list of sentences to parse
    @staticmethod
    def deep_phrases(parser, sents):

        sentences = parser.raw_parse_sents(sents)

        phrase_pos = []

        for sentence in sentences:

            for s in sentence:

                sent_leaves = s.leaves()
                #sent_leaves = map(lambda l: l.label(), sent_leaves)
                sent_str = " ".join(sent_leaves)

                d = {}
                postags = []

                d = QueryParser().recurse(s, d, sent_str)
                postags = s.pos()

                # make every string in pos tag ascii
                postags = map(lambda t: (t[0].encode("ascii", "ignore"), \
                                         t[1].encode("ascii", "ignore")), postags)

                phrase_pos.append((sent_str, QueryParser.map_phrase(sent_str, d), postags))

        return phrase_pos

if __name__ == "__main__":

    qparser = QueryParser()

    s = " key  wathari_l into the dict. hello world. bye world"

    #print qparser.deep_phrases(["# divide a and b and store into c"])
    print qparser.deep_phrases([" key  wathari_l into the dict. hello world. bye world", "hello world"])
