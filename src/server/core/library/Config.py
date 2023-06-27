from typing import List


class Config:
    folder_path: str = '/pictures'
    thumbnail_path: str = '/thumbnail'
    label_index_path: str = '/labels'
    raw_file_extensions: List[str] = [
        '.3fr',
        '.ari',
        '.arw',
        '.bay',
        '.braw',
        '.crw',
        '.cr2',
        '.cr3',
        '.cap',
        '.data',
        '.dcs',
        '.dcr',
        '.dng',
        '.drf',
        '.eip',
        '.erf',
        '.fff',
        '.gpr',
        '.iiq',
        '.k25',
        '.kdc',
        '.mdc',
        '.mef',
        '.mos',
        '.mrw',
        '.nef',
        '.nrw',
        '.obm',
        '.orf',
        '.pef',
        '.ptx',
        '.pxn',
        '.r3d',
        '.raf',
        '.raw',
        '.rwl',
        '.rw2',
        '.rwz',
        '.sr2',
        '.srf',
        '.srw',
        '.tif',
        '.x3f'
    ]

    filesystem_state_file: str = '/data/file_system_state.json'
