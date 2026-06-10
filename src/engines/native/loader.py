"""
engines.native.loader — shared ctypes base for the C++ and Rust engines
========================================================================
Both native engines expose the identical C ABI (engine_create/warm/measure/
stats/export/free), so they share everything here; the subclasses only differ
in their library name, source file and build command. Because they also share
the same splitmix64 RNG and move order, the C++ and Rust engines produce
bit-identical graphs for a given seed.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

from ..base import Engine, Param
from ..hamiltonian import HAMILTONIAN_PARAMS

NATIVE_DIR = os.path.dirname(os.path.abspath(__file__))

_I32 = ctypes.POINTER(ctypes.c_int32)
_I64 = ctypes.POINTER(ctypes.c_int64)


def load_lib(path: str):
    """Load a native engine .so and declare all entry-point signatures."""
    lib = ctypes.CDLL(path)
    lib.engine_create.restype = ctypes.c_void_p
    lib.engine_create.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64]
    sweep = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_double,
             ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.engine_warm.restype = ctypes.c_int
    lib.engine_warm.argtypes = sweep
    lib.engine_measure.restype = ctypes.c_double
    lib.engine_measure.argtypes = sweep
    lib.engine_stats.argtypes = [ctypes.c_void_p, _I64, _I32]
    lib.engine_export.argtypes = [ctypes.c_void_p, _I32, _I32]
    lib.engine_free.argtypes = [ctypes.c_void_p]
    return lib


class NativeEngine(Engine):
    SO_NAME = ""
    SRC_NAME = ""
    COMPILERS: list[str] = []
    _lib_cache: dict[str, object] = {}

    # ---- discovery / build ----
    @classmethod
    def so_path(cls) -> str:
        return os.path.join(NATIVE_DIR, cls.SO_NAME)

    @classmethod
    def src_path(cls) -> str:
        return os.path.join(NATIVE_DIR, cls.SRC_NAME)

    @classmethod
    def parameters(cls) -> list[Param]:
        return list(HAMILTONIAN_PARAMS)

    @classmethod
    def _compiler(cls):
        for c in cls.COMPILERS:
            if shutil.which(c):
                return c
        return None

    @classmethod
    def build_cmd(cls, compiler: str) -> list[str]:
        raise NotImplementedError

    @classmethod
    def available(cls) -> bool:
        so, src = cls.so_path(), cls.src_path()
        return os.path.exists(so) and (not os.path.exists(src)
                                       or os.path.getmtime(so) >= os.path.getmtime(src))

    @classmethod
    def ensure_available(cls, log=print) -> bool:
        if cls.available():
            return True
        comp = cls._compiler()
        if not comp:
            log(f"[engines] {cls.backend}: no compiler ({'/'.join(cls.COMPILERS)}) — unavailable")
            return False
        # Build to a unique temp path then atomically rename onto the final .so.
        # Many sweep workers may reach this at once; compiling straight to the
        # shared so_path() would let two g++ runs write the same file and
        # corrupt it. Temp-then-os.replace means every reader sees either the
        # old or the new *complete* library, never a half-written one.
        cmd = cls.build_cmd(comp)
        final = cls.so_path()
        tmp = f"{final}.tmp.{os.getpid()}.so"
        if len(cmd) >= 2 and cmd[-2] == "-o":
            cmd = cmd[:-1] + [tmp]
        else:                                   # unexpected shape — fall back to direct
            tmp = final
        log("[engines] build:", " ".join(cmd))
        ok = subprocess.call(cmd) == 0 and os.path.exists(tmp)
        if ok and tmp != final:
            os.replace(tmp, final)
        if not ok:
            if tmp != final and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            log(f"[engines] {cls.backend}: build failed — unavailable")
        return ok and os.path.exists(final)

    @classmethod
    def _lib(cls):
        path = cls.so_path()
        if path not in cls._lib_cache:
            cls._lib_cache[path] = load_lib(path)
        return cls._lib_cache[path]

    # ---- lifecycle ----
    def __init__(self, n, *, max_degree: int = 32, seed: int = 0, mode: int = 1, **params):
        super().__init__(n, max_degree=max_degree, seed=seed, **params)
        self.mode = int(mode)
        self.lib = self._lib()
        self.h = self.lib.engine_create(self.n, self.max_degree,
                                        ctypes.c_uint64(self.seed))
        if not self.h:
            raise MemoryError(f"{self.backend} engine_create failed at N={self.n}")

    def step(self, n_steps: int) -> None:
        p = self.params
        rc = self.lib.engine_warm(self.h, int(n_steps), p["degree_penalty"],
                                  p["temperature"], p["edge_cost"],
                                  p["locality_bias"], self.mode)
        if rc != 0:
            raise RuntimeError(f"max_degree={self.max_degree} hit in {self.backend} engine")

    def neighbors_degrees(self):
        nb = np.empty((self.n, self.max_degree), dtype=np.int32)
        dg = np.empty(self.n, dtype=np.int32)
        self.lib.engine_export(self.h, nb.ctypes.data_as(_I32), dg.ctypes.data_as(_I32))
        return nb, dg

    def degree_array(self) -> np.ndarray:
        dg = np.empty(self.n, dtype=np.int32)
        self.lib.engine_export(self.h, None, dg.ctypes.data_as(_I32))
        return dg

    @property
    def peak_degree(self) -> int:
        e = ctypes.c_int64(0); p = ctypes.c_int32(0)
        self.lib.engine_stats(self.h, ctypes.byref(e), ctypes.byref(p))
        return int(p.value)

    def stats(self) -> dict:
        e = ctypes.c_int64(0); p = ctypes.c_int32(0)
        self.lib.engine_stats(self.h, ctypes.byref(e), ctypes.byref(p))
        return dict(edges=int(e.value), k_avg=2 * int(e.value) / max(self.n, 1),
                    peak_degree=int(p.value))

    def close(self):
        if getattr(self, "h", None):
            self.lib.engine_free(self.h)
            self.h = None
