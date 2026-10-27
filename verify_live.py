"""Exercise a running server and save genuine HTTP responses."""
import csv
import io
import json
import sys
import time
from pathlib import Path
import httpx

root = Path(__file__).resolve().parent
sample = json.loads((root/'evidence/sample.json').read_text())
base = sys.argv[1] if len(sys.argv)>1 else 'http://localhost:8004'
evidence = {}
with httpx.Client(base_url=base, timeout=30) as client:
    def call(name, method, path, **kwargs):
        response = client.request(method,path,**kwargs)
        evidence[name] = {'method':method,'path':path,'status':response.status_code,
                          'request_id':response.headers.get('x-request-id'),
                          'duration_ms':response.headers.get('x-process-time-ms'),
                          'body':response.json() if response.content else None}
        return response
    assert call('health','GET','/').status_code==200
    single=call('single','POST','/predict',json=sample).json()
    assert call('batch','POST','/predict/batch',json={'samples':[sample,sample]}).status_code==200
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=list(sample)); writer.writeheader();writer.writerow(sample);writer.writerow(sample)
    (root/'evidence/sample.csv').write_text(stream.getvalue())
    assert call('csv','POST','/predict/csv',files={'file':('wine.csv',stream.getvalue(),'text/csv')}).status_code==200
    assert call('annotate','PATCH',f"/predictions/{single['prediction_id']}",json={'note':'Verified through live API'}).status_code==200
    assert call('history','GET','/predictions?cultivar=class_0&limit=2&offset=0').status_code==200
    assert call('invalid','POST','/predict',json={**sample,'alcohol':-1}).status_code==422
    assert call('missing','GET','/predictions/999999').status_code==404
    job=call('create_job','POST','/jobs',json={'samples':[sample,sample]}).json()
    for _ in range(30):
        status=call('job_status','GET',job['status_url']).json()
        if status['status'] in ['completed','failed']:break
        time.sleep(.1)
    assert status['status']=='completed'
    assert call('delete','DELETE',f"/predictions/{single['prediction_id']}").status_code==204
    assert call('deleted','GET',f"/predictions/{single['prediction_id']}").status_code==404
    assert call('metrics','GET','/metrics').status_code==200
(root/'evidence/live-api.json').write_text(json.dumps(evidence,indent=2))
print('Live checks: single, batch, CSV, history, annotations, deletion, 422, 404, background job and monitoring passed.')
