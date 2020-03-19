from pythokerlib.functions import opener_ufsa


def test_opener_ufsa():
    with opener_ufsa("testufsa.tmp", "w") as tu:
        tu.write("Plutone")
    with opener_ufsa("testufsa.tmp") as tu:
        assert tu.read() == "Plutone"
