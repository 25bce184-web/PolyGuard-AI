import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch,MagicMock
from PIL import Image
from src.config import Settings
from src.schema import Prescription
from src.json_parser import parse_response,OutputError
from src.image_utils import preprocess,load_image
from src.dataset_helper import inspect_dataset,read_truth,GT_FIELDS
from src.download_data import safe_extract
from src.evaluator import match_names,evaluate
from src.batch_runner import run_batch
from src.io_utils import write_csv


def prediction(name='DrugA'):
    return {'raw_text':name,'medicines':[{'drug_name':name,'dosage':'5 mg',
        'frequency':'od','duration':None,'route':None,'instructions':None,
        'confidence':0.9,'unclear_text':''}],
        'overall_confidence':0.9,'contains_unclear_content':False,'notes':''}

class FakeExtractor:
    """A deterministic transport stub, NOT VLM inference."""
    def __init__(self): self.calls=0
    def extract(self,*args):
        self.calls+=1
        return {'status':'success','prediction':prediction(),'raw_response':json.dumps(prediction()),
                'json_valid':True,'schema_valid':True,'attempts':1,'retries':0,
                'api_errors':0,'latency_seconds':0.01,'error':''}

class TempCase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.settings=Settings(model='offline-test-model',rpm=10000)
    def tearDown(self): self.temp.cleanup()
    def image(self,name='image.png',size=(100,50),color='white'):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',size,color).save(p)
        return p

class ParserTests(unittest.TestCase):
    def test_valid(self): self.assertEqual(parse_response(json.dumps(prediction()))[0].medicines[0].drug_name,'DrugA')
    def test_fence(self): self.assertTrue(parse_response('```json\n'+json.dumps(prediction())+'\n```')[1])
    def test_empty(self):
        with self.assertRaises(OutputError): parse_response('')
    def test_truncated(self):
        with self.assertRaises(OutputError): parse_response('{"medicines":[')
    def test_prose_not_silently_repaired(self):
        with self.assertRaises(OutputError): parse_response('Result: '+json.dumps(prediction()))
    def test_missing_field(self):
        p=prediction();del p['medicines'][0]['drug_name']
        with self.assertRaises(OutputError) as ctx: parse_response(json.dumps(p))
        self.assertTrue(ctx.exception.json_valid)
    def test_confidence_range(self):
        p=prediction();p['overall_confidence']=1.2
        with self.assertRaises(OutputError): parse_response(json.dumps(p))
    def test_high_confidence_unclear(self):
        with self.assertRaises(OutputError): parse_response(json.dumps(prediction('UNCLEAR')))
    def test_unclear_flag(self):
        p=prediction();p['medicines'][0]['dosage']='UNCLEAR'
        with self.assertRaises(OutputError): parse_response(json.dumps(p))
    def test_nan(self):
        with self.assertRaises(OutputError): parse_response(json.dumps(prediction()).replace('0.9','NaN'))
    def test_no_medicines_is_valid(self):
        p=prediction();p['medicines']=[]
        self.assertEqual(parse_response(json.dumps(p))[0].medicines,[])

class ImageTests(TempCase):
    def test_all_formats(self):
        for ext in ['jpg','jpeg','png','webp']:
            self.assertEqual(load_image(self.image('file.'+ext),self.settings).mode,'RGB')
    def test_resize_preserves_ratio(self):
        p=self.image(size=(500,250));im,_=preprocess(p,Settings(max_side=100))
        self.assertEqual(im.size,(100,50))
        with Image.open(p) as original:
            self.assertEqual(original.size,(500,250))
    def test_rotation(self):
        im,_=preprocess(self.image(),self.settings,rotation=90)
        self.assertEqual(im.size,(50,100))
    def test_variants(self):
        p=self.image(color='blue')
        a,_=preprocess(p,self.settings,'original');b,_=preprocess(p,self.settings,'grayscale')
        self.assertNotEqual(a.tobytes(),b.tobytes())
    def test_corrupt(self):
        p=self.root/'bad.png';p.write_bytes(b'invalid')
        with self.assertRaises(Exception): load_image(p,self.settings)
    def test_empty(self):
        p=self.root/'empty.jpg';p.touch()
        with self.assertRaises(ValueError): load_image(p,self.settings)
    def test_pixel_limit(self):
        with self.assertRaises(ValueError): load_image(self.image(),Settings(max_pixels=10))
    def test_missing(self):
        with self.assertRaises(FileNotFoundError): load_image(self.root/'missing.png',self.settings)
    def test_unsupported(self):
        with self.assertRaises(ValueError): load_image(self.root/'file.gif',self.settings)
    def test_exif_orientation(self):
        p=self.root/'exif.jpg';im=Image.new('RGB',(100,50));exif=im.getexif();exif[274]=6;im.save(p,exif=exif)
        self.assertEqual(load_image(p,self.settings).size,(50,100))

class DatasetTests(TempCase):
    def test_inspect_duplicate_and_corrupt(self):
        self.image('raw/a.png');self.image('raw/b.png')
        (self.root/'raw/bad.jpg').write_bytes(b'broken')
        s=inspect_dataset(self.root/'raw',self.root/'data',self.root/'reports',self.settings,self.root/'data/gt.csv')
        self.assertEqual((s['total_images'],s['usable_unique_images'],s['duplicate_images'],s['invalid_images']),(3,1,1,1))
        self.assertEqual(len(read_truth(self.root/'data/gt.csv')),3)
    def test_partial_not_complete(self):
        path=self.root/'gt.csv'
        write_csv(path,[{'image_path':'a.png','medicine_names':['UNCLEAR'],'annotation_status':'complete'}],GT_FIELDS)
        with self.assertRaises(ValueError): read_truth(path)
    def test_alignment(self):
        path=self.root/'gt.csv'
        write_csv(path,[{'image_path':'a.png','medicine_names':['A','B'],'dosages':['5mg'],'annotation_status':'complete'}],GT_FIELDS)
        with self.assertRaises(ValueError): read_truth(path)
    def test_zip_slip(self):
        z=self.root/'bad.zip'
        with zipfile.ZipFile(z,'w') as f:f.writestr('../escape','bad')
        with self.assertRaises(ValueError):safe_extract(z,self.root/'raw')
        self.assertFalse((self.root/'escape').exists())
    def test_zip_import(self):
        z=self.root/'good.zip'
        with zipfile.ZipFile(z,'w') as f:f.writestr('images/test.txt','original')
        safe_extract(z,self.root/'raw')
        self.assertEqual((self.root/'raw/images/test.txt').read_text(),'original')
        with self.assertRaises(ValueError):safe_extract(z,self.root/'raw')

class EvaluationTests(TempCase):
    def test_one_to_one(self):
        self.assertEqual(len(match_names(['A','A'],['A'])),1)
    def test_case_sensitivity(self):
        self.assertEqual(len(match_names(['DrugA'],['druga'],'exact')),0)
        self.assertEqual(len(match_names(['DrugA'],['druga'])),1)
    def test_fuzzy_separate(self):
        self.assertEqual(len(match_names(['paracetamol'],['paracetmol'])),0)
        self.assertEqual(len(match_names(['paracetamol'],['paracetmol'],'fuzzy')),1)
    def test_failed_images_are_misses_and_partial_excluded(self):
        truth={k:{'medicine_names':['DrugA'],'annotation_status':s,'handwriting_type':'very_messy',
                   'dosages':['5 mg']} for k,s in [('a','complete'),('b','complete'),('c','partial')]}
        records=[{'image_id':'a','status':'success','prediction':prediction()},
                 {'image_id':'b','status':'failed','prediction':None},
                 {'image_id':'c','status':'success','prediction':prediction('Extra')}]
        m=evaluate(records,truth,self.root/'out')
        self.assertEqual(m['quantitative_images'],2)
        self.assertEqual(m['name_and_detail_metrics']['case_insensitive']['recall'],0.5)
        self.assertEqual(m['name_and_detail_metrics']['dosage']['accuracy'],0.5)
        self.assertIsNone(m['confirmed_hallucinations'])
    def test_empty_metrics_not_fake_zero(self):
        m=evaluate([],{},self.root/'out')
        self.assertIsNone(m['name_and_detail_metrics'])
        self.assertIsNone(m['successful_processing_rate'])
    def test_extra_confident_prediction(self):
        truth={'a':{'medicine_names':['Other'],'annotation_status':'complete','handwriting_type':'moderate'}}
        m=evaluate([{'image_id':'a','status':'success','prediction':prediction()}],truth,self.root/'out')
        self.assertEqual(m['name_and_detail_metrics']['confident_unmatched_names'],1)

class BatchTests(TempCase):
    def test_cache_resume_and_config_invalidation(self):
        self.image('raw/a.png');fake=FakeExtractor()
        kw=dict(root=self.root/'raw',output=self.root/'out',truth_path=self.root/'gt.csv',settings=self.settings,extractor=fake)
        run_batch(**kw);m=run_batch(**kw)
        self.assertEqual(fake.calls,1);self.assertEqual(m['cached_images'],1)
        self.assertEqual(m['current_run_api_calls'],0)
        run_batch(**kw,variant='grayscale')
        self.assertEqual(fake.calls,2)
    def test_batch_continues_after_bad_image(self):
        self.image('raw/good.png');(self.root/'raw/bad.png').write_bytes(b'x')
        m=run_batch(self.root/'raw',self.root/'out',self.root/'gt.csv',self.settings,all_images=True,extractor=FakeExtractor())
        self.assertEqual((m['successful_images'],m['failed_image_count']),(1,1))
    def test_replay_never_calls_api(self):
        self.image('raw/a.png')
        m=run_batch(self.root/'raw',self.root/'out',self.root/'gt.csv',self.settings,replay=True)
        self.assertEqual(m['current_run_api_calls'],0);self.assertEqual(m['failed_image_count'],1)
    def test_missing_key_before_outputs(self):
        self.image('raw/a.png')
        with self.assertRaises(ValueError): run_batch(self.root/'raw',self.root/'out',self.root/'gt.csv',self.settings)
        self.assertFalse((self.root/'out').exists())
    def test_same_filename_different_subfolders(self):
        self.image('raw/x/a.png',color='red');self.image('raw/y/a.png',color='blue')
        m=run_batch(self.root/'raw',self.root/'out',self.root/'gt.csv',self.settings,extractor=FakeExtractor())
        self.assertEqual(m['selected_images'],2)

class ClientTests(unittest.TestCase):
    def test_transient_retry(self):
        from src.gemini_client import GeminiExtractor
        import httpx
        response=MagicMock();response.text=json.dumps(prediction());response.usage_metadata=None
        response.candidates=[MagicMock(finish_reason='STOP')]
        with patch('src.gemini_client.genai.Client') as cls,patch('src.gemini_client.time.sleep'):
            cls.return_value.models.generate_content.side_effect=[httpx.ReadTimeout('timeout'),response]
            client=GeminiExtractor(Settings(api_key='test-only-not-real',model='offline',rpm=10000))
            r=client.extract(b'fake','prompt')
            self.assertEqual((r['status'],r['attempts'],r['api_errors']),('success',2,1))
    def test_nontransient_no_retry(self):
        from src.gemini_client import GeminiExtractor
        class AuthError(Exception):code=401
        with patch('src.gemini_client.genai.Client') as cls:
            cls.return_value.models.generate_content.side_effect=AuthError()
            r=GeminiExtractor(Settings(api_key='test-only-not-real',model='offline')).extract(b'fake','prompt')
            self.assertEqual(r['attempts'],1)

if __name__=='__main__':unittest.main()
