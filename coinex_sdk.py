# coinex_sdk.py
# Description: A corrected and improved SDK for interacting with the CoinEx v2 API.
# Author: AI Assistant
# Version: 1.1.0

import hashlib
import json
import time
import requests
import asyncio
import websockets
import logging

class CoinEx:
    def __init__(self, access_id, secret_key):
        self.access_id = access_id
        self.secret_key = secret_key
        self.base_url = "https://api.coinex.com/v2"  # Corrected: Removed trailing space
        self.headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
        self.account = Account(self)
        self.market = Market(self)
        self.websocket = WebSocketClient(self)

    def request(self, method, path, params=None, need_sign=False):
        params = params or {}
        url = f"{self.base_url}{path}"
        headers = self.headers.copy()

        if need_sign:
            t = int(time.time() * 1000)
            headers['X-COINEX-APIKEY'] = self.access_id
            headers['X-COINEX-TIMESTAMP'] = str(t)
            
            if method.upper() in ['GET', 'DELETE']:
                query_string = '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
                to_sign_str = f"{query_string}&timestamp={t}{self.secret_key}" if query_string else f"timestamp={t}{self.secret_key}"
            else: # POST/PUT
                body_str = json.dumps(params, separators=(',', ':'))
                to_sign_str = f"{body_str}timestamp={t}{self.secret_key}"

            headers['X-COINEX-SIGNATURE'] = hashlib.sha256(to_sign_str.encode('utf-8')).hexdigest()

        try:
            if method.upper() in ['GET', 'DELETE']:
                response = requests.request(method, url, params=params, headers=headers)
            else:
                response = requests.request(method, url, data=json.dumps(params), headers=headers)
            
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"API Request failed for {method} {path}: {e}")
            return {"code": 9999, "message": str(e), "data": {}}

        return response.json()

class Account:
    def __init__(self, client):
        self.client = client

    def get_account_info(self):
        return self.client.request('GET', '/asset/spot/balance', params={}, need_sign=True)

class Market:
    def __init__(self, client):
        self.client = client

    def ping(self):
        return self.client.request('GET', '/ping')

class WebSocketClient:
    def __init__(self, client):
        self.client = client
        self.ws_url = "wss://socket.coinex.com/v2/websocket"
        self.ws = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.listen_task = None
        self.on_message_callback = None

    async def connect(self, on_message_callback):
        self.on_message_callback = on_message_callback
        while not self.connected and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.ws = await websockets.connect(self.ws_url)
                logging.info("WebSocket connected successfully.")
                self.connected = True
                self.reconnect_attempts = 0
                self.listen_task = asyncio.create_task(self._listen())
                return True
            except Exception as e:
                logging.error(f"WebSocket connection failed: {e}. Attempt {self.reconnect_attempts + 1}")
                self.connected = False
                self.reconnect_attempts += 1
                await asyncio.sleep(5 * self.reconnect_attempts)
        logging.critical("WebSocket could not connect after multiple attempts.")
        return False

    async def subscribe_to_tickers(self, symbols: list):
        if not self.connected or not self.ws:
            logging.warning("Cannot subscribe, WebSocket is not connected.")
            return

        formatted_symbols = [s.replace('/', '') for s in symbols]
        msg = {
            "method": "state.subscribe",
            "params": formatted_symbols,
            "id": int(time.time())
        }
        await self.ws.send(json.dumps(msg))
        logging.info(f"Sent subscription request for symbols: {formatted_symbols}")

    async def _listen(self):
        try:
            async for message in self.ws:
                if self.on_message_callback:
                    await self.on_message_callback(message)
        except websockets.exceptions.ConnectionClosed as e:
            logging.warning(f"WebSocket connection closed: {e}. Reconnecting...")
            self.connected = False
            await asyncio.sleep(5)
            await self.connect(self.on_message_callback) # Attempt to reconnect

    async def close(self):
        if self.listen_task:
            self.listen_task.cancel()
        if self.connected and self.ws:
            await self.ws.close()
            self.connected = False
            logging.info("WebSocket connection closed by client.")
