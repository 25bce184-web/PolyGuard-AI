"""One-to-one matching; no brand substitution or unverified hallucination claims."""
from difflib import SequenceMatcher
from collections import Counter
from .io_utils import write_csv, atomic_json

FIELDS = {'dosage':'dosages','frequency':'frequencies','duration':'durations',
          'route':'routes','instructions':'instructions'}

def normalized(text):
    return ' '.join(text.split()).casefold()

def match_names(expected, predicted, mode='case_insensitive', threshold=0.85):
    # Maximum-cardinality bipartite matching avoids reusing a prediction.
    def similarity(a,b):
        if mode=='exact': return float(a==b)
        if mode=='case_insensitive': return float(normalized(a)==normalized(b))
        return SequenceMatcher(None,normalized(a),normalized(b)).ratio()
    adjacency=[]
    for e in expected:
        scored=[(j,similarity(e,p)) for j,p in enumerate(predicted)]
        adjacency.append([j for j,s in sorted(scored,key=lambda x:(-x[1],x[0]))
                          if s >= (threshold if mode=='fuzzy' else 1)])
    owner={}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):
                owner[j]=i
                return True
        return False
    for i in range(len(expected)): augment(i,set())
    return sorted((i,j) for j,i in owner.items())

def ratio(n,d):
    return n/d if d else None

def summarize_rows(rows):
    out={'images':len(rows)}
    for mode in ['exact','case_insensitive','fuzzy']:
        tp=sum(r[mode+'_correct'] for r in rows)
        fp=sum(r[mode+'_extra'] for r in rows)
        fn=sum(r[mode+'_missing'] for r in rows)
        out[mode]={'correct':tp,'extra':fp,'missing':fn,
                   'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
                   'f1':ratio(2*tp,2*tp+fp+fn)}
    for field in FIELDS:
        correct=sum(r[field+'_correct'] for r in rows)
        total=sum(r[field+'_total'] for r in rows)
        out[field]={'correct':correct,'labelled_fields':total,'accuracy':ratio(correct,total)}
    out['confident_unmatched_names']=sum(r['confident_unmatched_names'] for r in rows)
    out['incomplete_prescriptions']=sum(r['case_insensitive_missing']>0 for r in rows)
    out['correct_medicine_count_images']=sum(r['expected_count']==r['predicted_count'] for r in rows)
    out['complete_name_set_match_rate']=ratio(sum(r['case_insensitive_missing']==0 and r['case_insensitive_extra']==0
                                                for r in rows),len(rows))
    return out

def evaluate(records: list[dict], truth: dict, output, wall_seconds=0.0):
    rows,qualitative,pred_csv,errors=[],[],[],[]
    uncertain=unclear=0
    for record in records:
        key=record['image_id']
        prediction=record.get('prediction') or {}
        medicines=prediction.get('medicines',[])
        if record['status']!='success':
            errors.append({k:record.get(k) for k in ['image_id','status','error','attempts','retries','api_errors']})
        uncertain+=sum(m['confidence']<0.8 or m['unclear_text']!='' or any(v=='UNCLEAR' for v in m.values()) for m in medicines)
        unclear+=sum(v=='UNCLEAR' for m in medicines for v in m.values())
        for m in medicines:
            pred_csv.append({'image_id':key,**m})
        gt=truth.get(key,{})
        qualitative.append({'image_id':key,'image_preview_path':record.get('image_path',''),
            'status':record['status'],'ground_truth_status':gt.get('annotation_status','unavailable'),
            'model_output':prediction,'human_review_notes':'', 'confirmed_hallucinations':'',
            'review_required':True})
        # Partial/absent labels cannot establish false positives or full recall.
        if gt.get('annotation_status')!='complete':
            continue
        expected=gt['medicine_names']
        visible=[(i,m) for i,m in enumerate(medicines) if m['drug_name'].strip().upper()!='UNCLEAR']
        names=[m['drug_name'] for _,m in visible]
        row={'image_id':key,'handwriting_type':gt['handwriting_type'],
             'status':record['status'],'expected_count':len(expected),'predicted_count':len(medicines)}
        for mode in ['exact','case_insensitive','fuzzy']:
            pairs=match_names(expected,names,mode)
            row.update({mode+'_correct':len(pairs),mode+'_missing':len(expected)-len(pairs),
                        mode+'_extra':len(names)-len(pairs)})
        pairs=match_names(expected,names)
        mapped={i:j for i,j in pairs}
        matched_predictions={j for _,j in pairs}
        row['confident_unmatched_names']=sum(m['confidence']>=0.8 for j,(_,m) in enumerate(visible) if j not in matched_predictions)
        for field,gt_field in FIELDS.items():
            values=gt.get(gt_field)
            correct=total=0
            if values is not None:
                for i,value in enumerate(values):
                    # null means not annotated/absent; UNCLEAR is not a scorable label.
                    if value is None or not value.strip() or value.upper()=='UNCLEAR': continue
                    total+=1
                    predicted=visible[mapped[i]][1].get(field) if i in mapped else None
                    correct+=bool(isinstance(predicted,str) and normalized(predicted)==normalized(value))
            row[field+'_correct']=correct
            row[field+'_total']=total
        rows.append(row)
    success=sum(r['status']=='success' for r in records)
    response_records=[r for r in records if r.get('json_valid') is not None]
    fresh=[r for r in records if not r.get('cached')]
    metrics={'selected_images':len(records),'successful_images':success,'failed_image_count':len(records)-success,
             'successful_processing_rate':ratio(success,len(records)),
             'json_validity_rate':ratio(sum(r.get('json_valid') is True for r in response_records),len(response_records)),
             'json_validity_denominator':len(response_records),
             'schema_validity_rate':ratio(sum(r.get('schema_valid') is True for r in response_records),len(response_records)),
             'average_source_latency_seconds':ratio(sum(r.get('latency_seconds',0) for r in records),len(records)),
             'current_run_wall_seconds':wall_seconds,
             'cached_images':sum(bool(r.get('cached')) for r in records),
             'current_run_api_calls':sum(r.get('attempts',0) for r in fresh),
             'current_run_retries':sum(r.get('retries',0) for r in fresh),
             'current_run_api_errors':sum(r.get('api_errors',0) for r in fresh),
             'uncertain_medicine_predictions':uncertain,'unclear_fields':unclear,
             'confirmed_hallucinations':None,
             'quantitative_images':len(rows),'qualitative_only_images':len(records)-len(rows),
             'name_and_detail_metrics':summarize_rows(rows) if rows else None,
             'by_handwriting_type':{c:summarize_rows([r for r in rows if r['handwriting_type']==c])
                                   for c in sorted({r['handwriting_type'] for r in rows})},
             'notes':['Fuzzy is spelling similarity, not clinical equivalence.',
                      'Unmatched names require review; not automatically proven hallucinations.',
                      'Confident unmatched uses subjective confidence >=0.8; it is not calibrated.',
                      'Failures on complete-labelled selected images contribute missing medicines.',
                      'Missing denominators are null; partial labels excluded from quantitative scoring.']}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'metrics.json',metrics)
    write_csv(output/'evaluation_report.csv',rows,list(rows[0]) if rows else ['image_id','handwriting_type','status'])
    write_csv(output/'qualitative_review.csv',qualitative,['image_id','image_preview_path','status','ground_truth_status',
              'model_output','review_required','human_review_notes','confirmed_hallucinations'])
    write_csv(output/'predictions.csv',pred_csv,['image_id','drug_name',*FIELDS,'confidence','unclear_text'])
    write_csv(output/'errors.csv',errors,['image_id','status','error','attempts','retries','api_errors'])
    tmp=output/'predictions.jsonl.tmp'
    import json,os
    with tmp.open('w',encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,output/'predictions.jsonl')
    return metrics
