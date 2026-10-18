"""Read arrays directly from ZIP_STORED .npz, sharing OS pages across workers."""
import struct,zipfile
from pathlib import Path
import numpy as np


def map_array(path,key):
    with zipfile.ZipFile(path) as z:
        info=z.getinfo(key+'.npy')
        if info.compress_type!=zipfile.ZIP_STORED:raise ValueError('Cache must be uncompressed np.savez')
        with open(path,'rb') as f:
            f.seek(info.header_offset)
            header=f.read(30)
            namelen,extralen=struct.unpack_from('<HH',header,26)
            f.seek(namelen+extralen,1)
            version=np.lib.format.read_magic(f)
            shape,fortran,dtype=np.lib.format._read_array_header(f,version)
            if dtype.hasobject:raise ValueError('Object cache forbidden')
            offset=f.tell()
    return np.memmap(path,mode='r',offset=offset,shape=shape,dtype=dtype,order='F' if fortran else 'C')


class MappedCache:
    def __init__(self,path):
        self.path=str(Path(path).resolve());self._open()
    def _open(self):
        self.xy=map_array(self.path,'xy');self.conf=map_array(self.path,'conf')
        self.offs=map_array(self.path,'offsets');self.fps=map_array(self.path,'fps')
        self.wh=map_array(self.path,'wh');self.ids=map_array(self.path,'ids')
    def __getstate__(self):return {'path':self.path}
    def __setstate__(self,state):self.path=state['path'];self._open()
    def __len__(self):return len(self.ids)
    def clip(self,i):
        a,b=self.offs[i:i+2]
        return np.array(self.xy[a:b]),self.conf[a:b].astype(np.float32),float(self.fps[i])
