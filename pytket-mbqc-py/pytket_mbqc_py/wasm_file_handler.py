from pathlib import Path
from pytket import wasm
import importlib_resources

def get_wasm_file_handler():
    return wasm.WasmFileHandler(
        importlib_resources.files("pytket_mbqc_py").joinpath('pytket_mbqc_rus.wasm')
    )
