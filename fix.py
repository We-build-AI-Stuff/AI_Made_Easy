with open('tests/test_v34.py') as f:
    code = f.read()

target = """    expected = sorted((x + 400, y + 250) for x, y in before.values())
        assert len(reopened) == len(expected)
        for r, e in zip(reopened, expected):
            assert r[0] == pytest.approx(e[0])
            assert r[1] == pytest.approx(e[1])"""

replacement = """    expected = sorted((x + 400, y + 250) for x, y in before.values())
    assert len(reopened) == len(expected)
    for r, e in zip(reopened, expected):
        assert r[0] == pytest.approx(e[0])
        assert r[1] == pytest.approx(e[1])"""

code = code.replace(target, replacement)

with open('tests/test_v34.py', 'w') as f:
    f.write(code)
