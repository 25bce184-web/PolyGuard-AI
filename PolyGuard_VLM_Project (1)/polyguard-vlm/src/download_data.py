"""Download/import original archives without flattening filenames or overwriting raw data."""
from pathlib import Path
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile

DATASET_PAGE = 'https://data.mendeley.com/datasets/k62rfd23kz/2'
# Mendeley Download All endpoint. If its API changes, use --archive after Download All.
DATASET_ZIP = 'https://data.mendeley.com/public-api/zip/k62rfd23kz/download/2'


def safe_extract(archive: Path, destination: Path):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination is nonempty; choose a new raw folder to preserve originals')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staging = Path(tmp)/'content'
        staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 3*1024**3:
                raise ValueError('Archive exceeds 3 GiB uncompressed limit')
            for info in z.infolist():
                name = info.filename.replace('\\','/')
                target = (staging/name).resolve()
                if not target.is_relative_to(staging.resolve()) or ':' in name or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Unsafe archive member')
                if info.is_dir():
                    target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src,dst)
        destination.mkdir(exist_ok=True)
        for item in staging.iterdir():
            shutil.move(str(item),str(destination/item.name))


def download(destination: Path, archive: Path | None = None, kaggle: str | None = None):
    if archive:
        safe_extract(archive,destination)
        return {'status':'imported','source':str(archive)}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'dataset.zip'
        if kaggle:
            if not (os.getenv('KAGGLE_API_TOKEN') or
                    (os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY')) or
                    (Path.home()/'.kaggle/kaggle.json').is_file()):
                raise ValueError('Kaggle credentials missing; see README setup. No secrets logged.')
            if not shutil.which('kaggle'):
                raise ValueError('Install optional dependency: python -m pip install kaggle')
            proc = subprocess.run(['kaggle','datasets','download','-d',kaggle,'-p',tmp],
                                  capture_output=True,timeout=300)
            if proc.returncode:
                raise ValueError('Kaggle download failed; check account, dataset access, and slug')
            archives = list(Path(tmp).glob('*.zip'))
            if len(archives)!=1:
                raise ValueError('Expected one Kaggle ZIP')
            path=archives[0]
        else:
            with urllib.request.urlopen(DATASET_ZIP,timeout=30) as response, path.open('wb') as f:
                size=0
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>1024**3:
                        raise ValueError('Download exceeds 1 GiB limit')
                    f.write(chunk)
        safe_extract(path,destination)
    return {'status':'downloaded','source':kaggle or DATASET_PAGE}
