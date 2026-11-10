import subprocess
import sys


def test_importing_the_cli_does_not_load_heavy_libraries():
    """The default install must stay light: torch, transformers and llama_cpp load on demand."""
    code = (
        "import sys, quantgap.cli, quantgap.backends, quantgap.score.run, quantgap.pack;"
        "bad = [m for m in ('torch', 'transformers', 'llama_cpp', 'mlx') if m in sys.modules];"
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
