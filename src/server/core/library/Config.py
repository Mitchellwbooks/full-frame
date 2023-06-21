from typing import List


class Config:
    folder_path: str = '/pictures'
    file_extensions: List[str] = [
        '.awr'
    ]

    filesystem_state_file: str = '/data/file_system_state.json'
