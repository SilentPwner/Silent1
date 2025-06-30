# coinex_sdk.py
import hashlib
import json
import time
import requests

class CoinEx:
    def __init__(self, access_id, secret_key):
        self.access_id = access_id
        self.secret_key = secret_key
        self.base_url = "https://api.coinex.com/v2"
        self.headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
        self.account = Account(self)
        self.market = Market(self)

    def request(self, method, path, params=None, need_sign=False):
        params = params or {}
        url = self.base_url + path
        headers = self.headers.copy()

        if need_sign:
            t = int(time.time() * 1000)
            headers['X-COINEX-APIKEY'] = self.access_id
            headers['X-COINEX-TIMESTAMP'] = str(t)
            
            if method.upper() in ['GET', 'DELETE']:
                query_string = '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
                to_sign = f"{query_string}&timestamp={t}{self.secret_key}" if query_string else f"timestamp={t}{self.secret_key}"
            else: # POST/PUT
                body_str = json.dumps(params, separators=(',', ':'))
                to_sign = f"{body_str}timestamp={t}{self.secret_key}"

            headers['X-COINEX-SIGNATURE'] = hashlib.sha256(to_sign.encode('utf-8')).hexdigest()

        try:
            if method.upper() in ['GET', 'DELETE']:
                response = requests.request(method, url, params=params, headers=headers)
            else:
                response = requests.request(method, url, data=json.dumps(params), headers=headers)
            
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Return error in a consistent format
            return {"code": 9999, "message": str(e), "data": {}}

        return response.json()

class Account:
    def __init__(self, client):
        self.client = client

    def get_account_info(self):
        # Corrected endpoint for V2
        return self.client.request('GET', '/asset/spot/balance', need_sign=True)

class Market:
    def __init__(self, client):
        self.client = client

    def ping(self):
        return self.client.request('GET', '/ping')
