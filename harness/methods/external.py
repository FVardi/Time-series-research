"""Importing official method code from external/ (D19).

Each official repository imports its own files as top-level modules
('import utils', 'from models import ...'), so its folder must be on the import
path while it is imported. Two repositories can use the same names (TS2Vec and
T-Loss both have a 'utils.py'), so each one is imported with those names
temporarily cleared, and the import path is restored afterwards. The imported
code keeps its own references, so both methods can be used in one process.
"""

import importlib
import os
import sys
import types

EXTERNAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external"))


def import_official(repo, module, local_names, stub_packages=(), preload=()):
    """Import `module` from external/<repo> and return it.

    local_names:   top-level module names the repository uses for its own files.
    stub_packages: packages whose __init__.py must be skipped (see harness/methods/tloss.py);
                   an empty package object pointing at the folder is used instead.
    preload:       submodules to import explicitly (needed when __init__.py is skipped).
    """
    repo_dir = os.path.join(EXTERNAL, repo)

    def ours(name):
        return name.split(".")[0] in local_names

    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if ours(n)}
    sys.path.insert(0, repo_dir)
    try:
        for pkg in stub_packages:
            stub = types.ModuleType(pkg)
            stub.__path__ = [os.path.join(repo_dir, pkg)]
            sys.modules[pkg] = stub
        for name in preload:
            importlib.import_module(name)
        return importlib.import_module(module)
    finally:
        sys.path.remove(repo_dir)
        for n in [n for n in sys.modules if ours(n)]:
            del sys.modules[n]
        sys.modules.update(saved)
