"""Paired preprocessing comparison. No winner selection without complete ground truth."""
import argparse
from pathlib import Path
from .config import ROOT,Settings
from .batch_runner import run_batch
from .io_utils import write_csv

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/raw/selected')
    parser.add_argument('--truth',type=Path,default=ROOT/'data/ground_truth.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/comparison')
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    settings=Settings.from_env()
    rows=[]
    for variant in ['original','contrast','grayscale']:
        m=run_batch(args.data,args.output/variant,args.truth,settings,
                    limit=args.limit,variant=variant,replay=args.replay)
        score=m['name_and_detail_metrics'] or {}
        names=score.get('case_insensitive',{})
        rows.append({'variant':variant,'selected_images':m['selected_images'],
                     'quantitative_images':m['quantitative_images'],
                     'name_f1':names.get('f1'),'missing':names.get('missing'),
                     'extra':names.get('extra'),'failed':m['failed_image_count'],
                     'latency_seconds':m['average_source_latency_seconds'],
                     'unclear_fields':m['unclear_fields']})
    write_csv(args.output/'comparison.csv',rows,list(rows[0]))

if __name__=='__main__':main()
