"""Non-destructive dataset inspection and an explicit, lossless GT contract."""
from pathlib import Path
from collections import Counter
import csv
import json
from PIL import Image, ImageDraw
from .image_utils import EXTENSIONS, load_image, sha256
from .io_utils import atomic_json, write_csv

GT_FIELDS = ['image_path', 'raw_ground_truth', 'medicine_names', 'dosages',
             'frequencies', 'durations', 'routes', 'instructions', 'handwriting_type',
             'annotation_status', 'notes']
CATEGORIES = {'very_messy', 'moderate', 'printed', 'handwritten_unspecified', 'unknown'}


def read_truth(path: Path) -> dict:
    """JSON arrays retain medicine alignment; blanks mean unavailable, [] means none."""
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'image_path', 'medicine_names', 'annotation_status'} <= set(reader.fieldnames or []):
            raise ValueError('ground_truth CSV is missing required columns')
        for row in reader:
            key = row['image_path'].replace('\\', '/')
            if not key or key in result or key.startswith('/') or '..' in Path(key).parts:
                raise ValueError('Empty, duplicate, or unsafe ground-truth image_path')
            status = row.get('annotation_status', '').strip() or 'unavailable'
            if status not in {'complete', 'partial', 'unavailable'}:
                raise ValueError('annotation_status must be complete, partial, or unavailable')
            names = row.get('medicine_names', '').strip()
            names = json.loads(names) if names else None
            if names is not None and (not isinstance(names, list) or
                                      not all(isinstance(n, str) and n.strip() for n in names)):
                raise ValueError('medicine_names must be a JSON string array')
            if status == 'complete' and (names is None or any(n.upper() == 'UNCLEAR' for n in names)):
                raise ValueError('Complete ground truth cannot contain unknown names')
            category = row.get('handwriting_type') or 'unknown'
            if category not in CATEGORIES:
                raise ValueError('Unknown handwriting_type')
            details = {}
            for field in ['dosages', 'frequencies', 'durations', 'routes', 'instructions']:
                values = row.get(field, '').strip()
                values = json.loads(values) if values else None
                if values is not None and (names is None or not isinstance(values, list)
                        or len(values) != len(names)
                        or not all(v is None or isinstance(v, str) for v in values)):
                    raise ValueError(f'{field} must align with medicine_names; use null for unknown')
                details[field] = values
            result[key] = {**row, **details, 'medicine_names': names,
                           'annotation_status': status, 'handwriting_type': category}
    return result


def inspect_dataset(root: Path, data_dir: Path, report_dir: Path, settings, truth_path: Path):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError('Dataset directory does not exist')
    gt = read_truth(truth_path)
    rows, seen, thumbnails = [], {}, []
    files = sorted(p for p in root.rglob('*') if p.is_file())
    labels = [p.relative_to(root).as_posix() for p in files if p.suffix.lower() in {'.csv','.txt','.json','.xlsx'}]
    for path in files:
        if path.suffix.lower() not in EXTENSIONS:
            continue
        key = path.relative_to(root).as_posix()
        row = {'image_path': key, 'extension': path.suffix.lower(),
               'width': '', 'height': '', 'status': 'invalid', 'error': '',
               'duplicate_of': '', 'sha256': '', 'pixel_sha256': '',
               'handwriting_type': gt.get(key, {}).get('handwriting_type', 'unknown'),
               'ground_truth_status': gt.get(key, {}).get('annotation_status', 'unavailable')}
        try:
            import hashlib
            im = load_image(path, settings)
            row.update(width=im.width, height=im.height, sha256=sha256(path))
            pixel_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            row['pixel_sha256'] = pixel_hash
            row['status'] = 'usable'
            if pixel_hash in seen:
                row.update(status='duplicate', duplicate_of=seen[pixel_hash])
            else:
                seen[pixel_hash] = key
                if len(thumbnails) < 24:
                    im.thumbnail((220, 170))
                    thumbnails.append((key, im.copy()))
        except Exception as exc:
            row['error'] = type(exc).__name__
        rows.append(row)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = ['image_path','extension','width','height','status','error','duplicate_of',
              'sha256','pixel_sha256','handwriting_type','ground_truth_status']
    write_csv(data_dir / 'metadata.csv', rows, fields)
    # Append missing records without altering existing user annotations.
    existing_keys = set(gt)
    missing = [{'image_path': r['image_path'], 'handwriting_type':'unknown',
                'annotation_status':'unavailable'} for r in rows if r['image_path'] not in existing_keys]
    if not truth_path.exists():
        write_csv(truth_path, missing, GT_FIELDS)
    elif missing:
        write_csv(data_dir / 'ground_truth_new_rows.csv', missing, GT_FIELDS)
    if thumbnails:
        sheet = Image.new('RGB', (960, ((len(thumbnails)+3)//4)*210), 'white')
        draw = ImageDraw.Draw(sheet)
        for i,(name,im) in enumerate(thumbnails):
            x,y=(i%4)*240,(i//4)*210
            sheet.paste(im,(x,y))
            draw.text((x+3,y+175), name.encode('ascii','replace').decode()[:35], fill='black')
        sheet.save(report_dir/'contact_sheet.jpg')
    summary = {'dataset_root':str(root), 'total_images':len(rows),
               'usable_unique_images':sum(r['status']=='usable' for r in rows),
               'duplicate_images':sum(r['status']=='duplicate' for r in rows),
               'invalid_images':sum(r['status']=='invalid' for r in rows),
               'handwriting_categories':dict(Counter(r['handwriting_type'] for r in rows)),
               'complete_ground_truth_images':sum(r['ground_truth_status']=='complete' for r in rows),
               'missing_or_partial_ground_truth_images':sum(r['ground_truth_status']!='complete' for r in rows),
               'label_candidates':labels,
               'note':'Handwriting type is human annotation, never inferred from filenames. YOLO boxes are not transcriptions.'}
    atomic_json(report_dir/'inspection.json',summary)
    return summary
