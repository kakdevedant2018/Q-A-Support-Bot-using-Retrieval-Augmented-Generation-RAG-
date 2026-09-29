"""Test package.

This file is not incidental. It makes `tests` a package so pytest prepends the
repository root to sys.path rather than the tests directory, which is what lets
both `import app...` and `from tests.conftest import ...` resolve when pytest is
run from the project root.
"""
