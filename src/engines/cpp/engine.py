"""engines.cpp.engine — C++ engine (engine_core.cpp via ctypes)."""
from ..native.loader import NativeEngine


class CppEngine(NativeEngine):
    name = "C++ (engine_core.cpp)"
    backend = "cpp"
    SO_NAME = "engine_core.so"
    SRC_NAME = "engine_core.cpp"
    COMPILERS = ["g++", "clang++"]

    @classmethod
    def build_cmd(cls, compiler):
        return [compiler, "-O3", "-march=native", "-funroll-loops",
                "-shared", "-fPIC", cls.src_path(), "-o", cls.so_path()]
