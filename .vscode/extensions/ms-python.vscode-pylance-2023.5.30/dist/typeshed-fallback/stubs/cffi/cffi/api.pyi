import distutils.core
import sys
import types
from _typeshed import Incomplete, ReadableBuffer, WriteableBuffer
from collections.abc import Callable, Hashable
from typing import Any, TypeVar, overload
from typing_extensions import Literal, TypeAlias

import _cffi_backend

_T = TypeVar("_T")

basestring: TypeAlias = str  # noqa: Y042

class FFI:
    CData: TypeAlias = _cffi_backend._CDataBase
    CType: TypeAlias = _cffi_backend.CType
    buffer: TypeAlias = _cffi_backend.buffer  # noqa: Y042

    BVoidP: CType
    BCharA: CType
    NULL: CType
    errno: int

    def __init__(self, backend: types.ModuleType | None = None) -> None: ...
    def cdef(self, csource: str, override: bool = False, packed: bool = False, pack: int | None = None) -> None: ...
    def embedding_api(self, csource: str, packed: bool = False, pack: bool | int | None = None) -> None: ...

    if sys.platform == "win32":
        def dlopen(self, name: str, flags: int = ...) -> _cffi_backend.Lib: ...
    else:
        def dlopen(self, name: str | None, flags: int = ...) -> _cffi_backend.Lib: ...

    def dlclose(self, lib: _cffi_backend.Lib) -> None: ...
    def typeof(self, cdecl: str | CData | types.BuiltinFunctionType | types.FunctionType) -> CType: ...
    def sizeof(self, cdecl: str | CData) -> int: ...
    def alignof(self, cdecl: str | CData) -> int: ...
    def offsetof(self, cdecl: str | CData, *fields_or_indexes: str | int) -> int: ...
    def new(self, cdecl: str | CType, init: Incomplete | None = None) -> CData: ...
    def new_allocator(
        self,
        alloc: Callable[[int], CData] | None = None,
        free: Callable[[CData], Any] | None = None,
        should_clear_after_alloc: bool = True,
    ) -> _cffi_backend._Allocator: ...
    def cast(self, cdecl: str | CType, source: CData | int) -> CData: ...
    def string(self, cdata: CData, maxlen: int = -1) -> bytes | str: ...
    def unpack(self, cdata: CData, length: int) -> bytes | str | list[Any]: ...
    @overload
    def from_buffer(self, cdecl: ReadableBuffer, require_writable: Literal[False] = False) -> CData: ...
    @overload
    def from_buffer(self, cdecl: WriteableBuffer, require_writable: Literal[True]) -> CData: ...
    @overload
    def from_buffer(
        self, cdecl: str | CType, python_buffer: ReadableBuffer, require_writable: Literal[False] = False
    ) -> CData: ...
    @overload
    def from_buffer(self, cdecl: str | CType, python_buffer: WriteableBuffer, require_writable: Literal[True]) -> CData: ...
    def memmove(self, dest: CData | WriteableBuffer, src: CData | ReadableBuffer, n: int) -> None: ...
    @overload
    def callback(
        self,
        cdecl: str | CType,
        python_callable: None = None,
        error: Any = None,
        onerror: Callable[[Exception, Any, Any], None] | None = None,
    ) -> Callable[[Callable[..., _T]], Callable[..., _T]]: ...
    @overload
    def callback(
        self,
        cdecl: str | CType,
        python_callable: Callable[..., _T],
        error: Any = None,
        onerror: Callable[[Exception, Any, Any], None] | None = None,
    ) -> Callable[..., _T]: ...
    def getctype(self, cdecl: str | CType, replace_with: str = "") -> str: ...
    @overload
    def gc(self, cdata: CData, destructor: Callable[[CData], Any], size: int = 0) -> CData: ...
    @overload
    def gc(self, cdata: CData, destructor: None, size: int = 0) -> None: ...
    def verify(self, source: str = "", tmpdir: str | None = None, **kwargs: Any) -> _cffi_backend.Lib: ...
    # Technically exists on all OSs, but crashes on all but Windows. So we hide it in stubs
    if sys.platform == "win32":
        def getwinerror(self, code: int = -1) -> tuple[int, str] | None: ...

    def addressof(self, cdata: CData, *fields_or_indexes: str | int) -> CData: ...
    def include(self, ffi_to_include: FFI) -> None: ...
    def new_handle(self, x: Any) -> CData: ...
    def from_handle(self, x: CData) -> Any: ...
    def release(self, x: CData) -> None: ...
    def set_unicode(self, enabled_flag: bool) -> None: ...
    def set_source(self, module_name: str, source: str, source_extension: str = ".c", **kwds: Any) -> None: ...
    def set_source_pkgconfig(
        self, module_name: str, pkgconfig_libs: list[str], source: str, source_extension: str = ".c", **kwds: Any
    ) -> None: ...
    def distutils_extension(self, tmpdir: str = "build", verbose: bool = True) -> distutils.core.Extension: ...
    def emit_c_code(self, filename: str) -> None: ...
    def emit_python_code(self, filename: str) -> None: ...
    def compile(self, tmpdir: str = ".", verbose: int = 0, target: str | None = None, debug: bool | None = None) -> str: ...
    def init_once(self, func: Callable[[], Any], tag: Hashable) -> Any: ...
    def embedding_init_code(self, pysource: str) -> None: ...
    def def_extern(self, *args: Any, **kwds: Any) -> None: ...
    def list_types(self) -> tuple[list[str], list[str], list[str]]: ...
