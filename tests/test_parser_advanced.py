import taichi as ti

from taichimps.input import LAMMPSInputParser


def test_jump_and_variable_loops(tmp_path):
    script_file = tmp_path / "in.loop"
    content = """
    variable a loop 3
    variable b equal ${a}*10
    print "Current a: ${a}, b: ${b}"
    next a
    jump SELF
    """
    script_file.write_text(content)

    parser = LAMMPSInputParser(script_file, default_fp=ti.f64, arch=ti.cpu)
    parser.execute()
    assert float(parser.variables["a"]) == 3.0
    assert float(parser.variables["b"]) == 30.0
