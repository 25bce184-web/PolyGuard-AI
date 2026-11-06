import time
import random
from google import genai
from google.genai import types
import httpx
from .schema import Prescription
from .json_parser import parse_response, OutputError

class GeminiExtractor:
    def __init__(self, settings):
        settings.require_api()
        self.settings = settings
        self.client = genai.Client(api_key=settings.api_key,
            http_options=types.HttpOptions(timeout=int(settings.timeout*1000),
                retry_options=types.HttpRetryOptions(attempts=1)))
        self.last_request = None

    def close(self):
        self.client.close()

    def _pace(self):
        if self.last_request is not None:
            time.sleep(max(0, 60/self.settings.rpm - (time.monotonic()-self.last_request)))
        self.last_request=time.monotonic()

    def extract(self, image_bytes: bytes, prompt: str) -> dict:
        start = time.monotonic()
        result = {'status':'failed','prediction':None,'raw_response':'',
                  'json_valid':None,'schema_valid':False,'fence_removed':False,
                  'attempts':0,'retries':0,'api_errors':0,'usage':{},'error':''}
        for attempt in range(self.settings.retries+1):
            self._pace()
            result['attempts']+=1
            result['retries']=attempt
            try:
                response = self.client.models.generate_content(
                    model=self.settings.model,
                    contents=[prompt,types.Part.from_bytes(data=image_bytes,mime_type='image/png')],
                    config=types.GenerateContentConfig(response_mime_type='application/json',
                        response_schema=Prescription,temperature=0,max_output_tokens=8192))
                result['raw_response']=response.text or ''
                result['usage']=response.usage_metadata.model_dump(mode='json') if response.usage_metadata else {}
                candidates=response.candidates or []
                finish=str(candidates[0].finish_reason) if candidates else ''
                result['finish_reason']=finish
                if not finish.endswith('STOP'):
                    result.update(error='non_stop_response',json_valid=False)
                    break
                try:
                    parsed,fenced=parse_response(result['raw_response'])
                    result.update(status='success',prediction=parsed.model_dump(),
                                  json_valid=True,schema_valid=True,fence_removed=fenced)
                except OutputError as exc:
                    result.update(error=str(exc),json_valid=exc.json_valid)
                break  # Never resample malformed/uncertain output to cherry-pick a result.
            except Exception as exc:
                code=getattr(exc,'code',None)
                result['api_errors']+=1
                result['error']=f'api_{code}' if code else type(exc).__name__
                transient=code in {408,429,500,502,503,504} or isinstance(exc,(httpx.TimeoutException,httpx.TransportError))
                if not transient or attempt==self.settings.retries:
                    break
                delay=min(60,2**attempt+random.uniform(0,1))
                response=getattr(exc,'response',None)
                if response is not None:
                    try:
                        delay=max(delay,float(response.headers.get('Retry-After','0')))
                    except (ValueError,TypeError):
                        pass
                time.sleep(delay)
        result['latency_seconds']=round(time.monotonic()-start,4)
        return result
