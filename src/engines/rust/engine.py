"""engines.rust.engine — Rust engine (engine_core.rs via ctypes).

Built with one `rustc` call (no cargo). Identical RNG + move order to the C++
engine, so for a given seed the two produce bit-identical graphs. Rust buys
safety/ergonomics here, not speed: it lowers to the same native code and hits
the same DRAM-latency wall as C++.
"""
from ..native.loader import NativeEngine


class RustEngine(NativeEngine):
    name = "Rust (engine_core.rs)"
    backend = "rust"
    SO_NAME = "engine_core_rs.so"
    SRC_NAME = "engine_core.rs"
    COMPILERS = ["rustc"]

    @classmethod
    def build_cmd(cls, compiler):
        return [compiler, "-C", "opt-level=3", "-C", "target-cpu=native",
                "--crate-type=cdylib", cls.src_path(), "-o", cls.so_path()]
