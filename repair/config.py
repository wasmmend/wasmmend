"""
Shared configuration for libclang path auto-detection.
Searches common LLVM install locations so the repo works across machines
without hardcoded paths.
"""
import os
import glob
from clang import cindex

def _find_libclang():
    """Auto-detect the libclang shared library path."""
    # 1. Honour explicit env var override (highest priority)
    env_path = os.environ.get("LIBCLANG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Search system LLVM installs (prefer higher versions)
    search_patterns = [
        "/usr/lib/llvm-*/lib/libclang-*.so.1",
        "/usr/lib/llvm-*/lib/libclang.so",
        "/usr/lib/x86_64-linux-gnu/libclang-*.so.1",
        "/usr/lib/libclang.so",
    ]
    candidates = []
    for pattern in search_patterns:
        candidates.extend(glob.glob(pattern))

    # Sort descending so highest LLVM version wins
    candidates.sort(reverse=True)
    if candidates:
        return candidates[0]

    # 3. Check user-local installs
    home = os.path.expanduser("~")
    local_patterns = [
        os.path.join(home, "llvm-*/lib/libclang*.so*"),
        os.path.join(home, ".local/lib/libclang*.so*"),
    ]
    for pattern in local_patterns:
        hits = sorted(glob.glob(pattern), reverse=True)
        if hits:
            return hits[0]

    return None


def setup_libclang():
    """Configure clang.cindex with the detected libclang path.
    Call this once at import time; subsequent calls are no-ops.
    """
    if getattr(setup_libclang, "_done", False):
        return
    path = _find_libclang()
    if path:
        try:
            cindex.Config.set_library_file(path)
        except Exception:
            pass  # Already configured by another module
    else:
        raise RuntimeError(
            "Could not find libclang shared library. "
            "Install LLVM or set the LIBCLANG_PATH environment variable.\n"
            "  Example: export LIBCLANG_PATH=/usr/lib/llvm-15/lib/libclang-15.so.1"
        )
    setup_libclang._done = True


# Auto-configure on import
setup_libclang()
