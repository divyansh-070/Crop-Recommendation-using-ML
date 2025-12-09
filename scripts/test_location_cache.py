import requests, time

url='http://127.0.0.1:5000/crop/location-suggest'
headers={'Content-Type':'application/json'}
payload={'lat':17.3850,'lon':78.4867}

for i in range(1,3):
    t0=time.perf_counter()
    try:
        r=requests.post(url,json=payload,headers=headers,timeout=10)
        elapsed=time.perf_counter()-t0
        try:
            data=r.json()
        except Exception:
            data=r.text
        print(f'request {i} status={r.status_code} elapsed={elapsed:.3f}s data={data}')
    except Exception as e:
        print('request',i,'failed',e)
