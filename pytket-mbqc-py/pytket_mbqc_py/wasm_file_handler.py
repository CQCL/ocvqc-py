from pathlib import Path
from pytket import wasm

def get_wasm_file_handler():
    rus_dir = Path().cwd().joinpath('../pytket-mbqc-rus')
    wasm_file = rus_dir.joinpath('target/wasm32-unknown-unknown/release/pytket_mbqc_rus.wasm')
    return wasm.WasmFileHandler(wasm_file)
