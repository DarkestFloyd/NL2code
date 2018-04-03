## This file contains various file read/write utilities

import os

## file read utils #############################################################

# given a path to a file;
# returns the contents of that file
def file_contents(f):

    assert(os.path.exists(f))

    contents = ""
    with open(f, "r") as fp:
        contents = fp.read()

    return contents
