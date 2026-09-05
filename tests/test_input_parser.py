from pathlib import Path

from taichimps.input import LAMMPSInputParser


def test_variable_evaluation():
    p = Path("/tmp/test_var.in")
    p.write_text("""
variable Poi equal 0.23
variable Gshear equal 29.1e9
variable kn equal 4.0*${Gshear}/(3*(1-${Poi}))
variable kt equal ${kn}*2*(1-${Poi})/(2-${Poi})
""")
    parser = LAMMPSInputParser(p)
    parser.load_script()
    for cmd in parser.commands:
        tokens = parser.parse_line(cmd)
        if tokens and tokens[0] == "variable":
            name = tokens[1]
            expr = "".join(tokens[3:])
            parser.variables[name] = parser.evaluate_expression(expr)

    assert abs(parser.variables["Poi"] - 0.23) < 1e-6
    assert abs(parser.variables["Gshear"] - 29.1e9) < 1e-6
    expected_kn = 4.0 * 29.1e9 / (3.0 * (1.0 - 0.23))
    assert abs(parser.variables["kn"] - expected_kn) < 1e-3
    if p.exists():
        p.unlink()
