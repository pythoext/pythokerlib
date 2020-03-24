from os.path import basename
import pytest
from pythokerlib.functions import opener_ufsa, lister_ufsa


@pytest.mark.parametrize("fname", ["test1.txt", "test2.log"])
def test_opener_ufsa(fname):
    content = "Contenuto di " + fname
    with opener_ufsa(fname, "w") as tu:
        assert basename(tu.name) == fname
        tu.write(content)
    with opener_ufsa(fname) as tu:
        assert tu.read() == content


def test_lister_ufsa():
    for fname in lister_ufsa():
        with opener_ufsa(fname) as tu:
            print("{}: {}".format(fname, tu.read(10)))
