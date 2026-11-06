from pathlib import Path
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from . import __version__
from .image_utils import EXTENSIONS, preprocess, sha256
from .extraction_prompt import get_prompt
from .schema import Prescription
from .io_utils import atomic_json
from .evaluator import evaluate
from .dataset_helper import read_truth

log=logging.getLogger(__name__)


def select_images(root, image=None, limit=10, all_images=False, truth=None, category=None):
    root=root.resolve()
    if image:
        path=image.resolve()
        if not path.exists(): raise ValueError('Image does not exist')
        if not path.is_relative_to(root): root=path.parent
        candidates=[path]
    else:
        if not root.is_dir(): raise ValueError('Dataset directory does not exist')
        candidates=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    truth=truth or {}
    if category:
        candidates=[p for p in candidates if truth.get(p.relative_to(root).as_posix(),{}).get('handwriting_type')==category]
    # Prioritize human-tagged hard handwriting; unknown before printed. Never infer difficulty.
    rank={'very_messy':0,'moderate':1,'handwritten_unspecified':2,'unknown':3,'printed':4}
    candidates.sort(key=lambda p:(rank.get(truth.get(p.relative_to(root).as_posix(),{}).get('handwriting_type','unknown'),3),p.as_posix()))
    unique,seen,duplicates=[],set(),0
    for p in candidates:
        try: digest=sha256(p)
        except OSError: digest=str(p)
        if digest in seen:
            duplicates+=1
            continue
        seen.add(digest)
        unique.append(p)
    if not all_images and not image: unique=unique[:limit]
    return root,unique,duplicates


def run_batch(root: Path, output: Path, truth_path: Path, settings, image=None,
              limit=10, all_images=False, variant='original', rotation=0,
              category=None, replay=False, extractor=None):
    if limit<=0: raise ValueError('limit must be positive')
    truth=read_truth(truth_path)
    root,images,duplicates=select_images(root,image,limit,all_images,truth,category)
    if not images: raise ValueError('No matching images. Download/import the dataset first; see README.')
    if not replay and extractor is None: settings.require_api()
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'.run.lock'
    try: lock.touch(exist_ok=False)
    except FileExistsError: raise ValueError('Output is locked; see README recovery instructions') from None
    own_client=False
    start=time.monotonic()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    run_dir=output/'runs'/run_id
    records=[]
    prompt=get_prompt()
    config={'model':settings.model,'variant':variant,'rotation':rotation,'max_side':settings.max_side,
            'version':__version__,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'schema_sha256':hashlib.sha256(json.dumps(Prescription.model_json_schema(),sort_keys=True).encode()).hexdigest(),
            'temperature':0,'max_output_tokens':8192}
    atomic_json(run_dir/'manifest.json',{'run_id':run_id,'created_utc':datetime.now(timezone.utc).isoformat(),
        'configuration':config,'images':[p.relative_to(root).as_posix() for p in images],
        'exact_duplicate_files_skipped':duplicates,'replay_only':replay,
        'ground_truth_sha256':sha256(truth_path) if truth_path.exists() else None})
    handler=logging.FileHandler(run_dir/'run.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        if extractor is None and not replay:
            from .gemini_client import GeminiExtractor
            extractor=GeminiExtractor(settings)
            own_client=True
        for index,path in enumerate(images):
            item_start=time.monotonic()
            key=path.relative_to(root).as_posix()
            rec={'image_id':key,'image_path':str(path),'status':'failed','prediction':None,
                 'json_valid':None,'schema_valid':False,'attempts':0,'retries':0,
                 'api_errors':0,'cached':False,'error':'','latency_seconds':0}
            try:
                _,image_bytes=preprocess(path,settings,variant,rotation)
                cache_key=hashlib.sha256(image_bytes+json.dumps(config,sort_keys=True).encode()).hexdigest()
                cache=output/'cache'/(cache_key+'.json')
                cached=None
                if cache.exists():
                    try:
                        candidate=json.loads(cache.read_text(encoding='utf-8'))
                        Prescription.model_validate(candidate['prediction'])
                        if candidate.get('status')=='success': cached=candidate
                    except (ValueError,KeyError,TypeError):
                        log.warning('Invalid cache entry ignored for selected image %d',index+1)
                if cached:
                    rec.update(cached,cached=True)
                elif replay:
                    rec['error']='cache_miss'
                else:
                    processed=output/'processed'/(cache_key+'.png')
                    processed.parent.mkdir(exist_ok=True)
                    if not processed.exists(): processed.write_bytes(image_bytes)
                    result=extractor.extract(image_bytes,prompt)
                    rec.update(result)
                    if rec['status']=='success':
                        Prescription.model_validate(rec['prediction'])
                        atomic_json(cache,result)
                rec['cache_key']=cache_key
            except Exception as exc:
                rec.update(status='failed',prediction=None,error=type(exc).__name__)
            rec.update(image_id=key,image_path=str(path),configuration=config,
                       processing_seconds=round(time.monotonic()-item_start,4))
            # Atomic per-image output is authoritative even after interruption.
            atomic_json(run_dir/'records'/f'{index:06d}.json',rec)
            records.append(rec)
            log.info('Image %d/%d: %s%s',index+1,len(images),rec['status'],' (cached)' if rec['cached'] else '')
        metrics=evaluate(records,truth,run_dir,time.monotonic()-start)
        evaluate(records,truth,output,time.monotonic()-start)
        atomic_json(output/'latest_run.json',{'run_id':run_id,'path':str(run_dir)})
        return metrics
    finally:
        try:
            if own_client: extractor.close()
        finally:
            handler.close()
            log.removeHandler(handler)
            lock.unlink(missing_ok=True)
