"""Run from the project folder: python -m src.main --help."""
import argparse
import json
import logging
from pathlib import Path
from .config import ROOT, Settings
from .io_utils import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='PolyGuard: prescription VLM research CLI')
    subs=parser.add_subparsers(dest='command',required=True)
    dl=subs.add_parser('download',help='Download selected Mendeley dataset, or import a ZIP')
    dl.add_argument('--destination',type=Path,default=ROOT/'data/raw/selected')
    source=dl.add_mutually_exclusive_group()
    source.add_argument('--archive',type=Path)
    source.add_argument('--kaggle',help='Optional OWNER/DATASET-SLUG; verify its licence first')
    inspect=subs.add_parser('inspect')
    inspect.add_argument('--data',type=Path,default=ROOT/'data/raw/selected')
    inspect.add_argument('--truth',type=Path,default=ROOT/'data/ground_truth.csv')
    inspect.add_argument('--report-dir',type=Path,default=ROOT/'reports')
    run=subs.add_parser('run')
    run.add_argument('--data',type=Path,default=ROOT/'data/raw/selected')
    run.add_argument('--truth',type=Path,default=ROOT/'data/ground_truth.csv')
    run.add_argument('--output',type=Path,default=ROOT/'outputs')
    selection=run.add_mutually_exclusive_group()
    selection.add_argument('--image',type=Path)
    selection.add_argument('--limit',type=int,default=10)
    selection.add_argument('--all',action='store_true')
    run.add_argument('--variant',choices=['original','contrast','grayscale'],default='original')
    run.add_argument('--rotation',type=int,choices=[0,90,180,270],default=0)
    run.add_argument('--category',choices=['very_messy','moderate','handwritten_unspecified','printed','unknown'])
    run.add_argument('--replay',action='store_true',help='Use cached real results only; never call API')
    ev=subs.add_parser('evaluate')
    ev.add_argument('--predictions',type=Path,default=ROOT/'outputs/predictions.jsonl')
    ev.add_argument('--truth',type=Path,default=ROOT/'data/ground_truth.csv')
    ev.add_argument('--output',type=Path,default=ROOT/'outputs/reevaluated')
    subs.add_parser('models',help='List available Gemini models using your key')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    # SDK transport logs can contain sensitive data; keep third-party logs quiet.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    settings=Settings.from_env()
    try:
        if args.command=='download':
            from .download_data import download,DATASET_PAGE
            try:
                report=download(args.destination,args.archive,args.kaggle)
            except Exception as exc:
                report={'status':'blocked','error_type':type(exc).__name__,
                        'http_status':getattr(exc,'code',None),'source':DATASET_PAGE,
                        'manual_fallback':'Open the dataset page, Download All; python -m src.main download --archive PATH.zip'}
                atomic_json(ROOT/'reports/download_status.json',report)
                logging.error('%s',json.dumps(report))
                return 2
            atomic_json(ROOT/'reports/download_status.json',report)
        elif args.command=='inspect':
            from .dataset_helper import inspect_dataset
            report=inspect_dataset(args.data,args.truth.parent,args.report_dir,settings,args.truth)
        elif args.command=='run':
            from .batch_runner import run_batch
            report=run_batch(args.data,args.output,args.truth,settings,args.image,args.limit,
                             args.all,args.variant,args.rotation,args.category,args.replay)
        elif args.command=='evaluate':
            from .dataset_helper import read_truth
            from .evaluator import evaluate
            records=[json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
            ids=[r['image_id'] for r in records]
            if len(ids)!=len(set(ids)): raise ValueError('Duplicate image IDs: evaluate one run at a time')
            report=evaluate(records,read_truth(args.truth),args.output)
        else:
            from google import genai
            if not settings.api_key or settings.api_key=='your_key_here': raise ValueError('Set GEMINI_API_KEY first')
            with genai.Client(api_key=settings.api_key) as client:
                report={'models':[m.name for m in client.models.list()]}
        logging.info('%s',json.dumps(report,indent=2))
        return 0
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        logging.error('%s',exc)
        return 2
    except Exception as exc:
        logging.error('Operation failed: %s. No credentials or response content logged.',type(exc).__name__)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
