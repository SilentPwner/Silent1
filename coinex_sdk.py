# coinex_sdk.py
# Description: A robust SDK for CoinEx v2 API with corrected endpoints.
# Author: AI Assistant
# Version: 1.2.1 (Fixed account balance endpoint)

import hashlib
import json
import time
import requests
import asyncio
import websockets
import logging

# Configure logging for the SDK
sdk_logger = logging.getLogger(__name__)

class CoinEx:
    def __init__(self, access_id, secret_key):
        self.access_id = access_id
        self.secret_key = secret_key
        self.base_url = "https://api.coinex.com/v2"
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
                to_sign_str = f"{query_string}×tamp={t}{self.secret_key}" if query_string else f"timestamp={t}{self.secret_key}"
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
            sdk_logger.error(f"API Request failed for {method} {path}: {e}")
            return {"code": 9999, "message": str(e), "data": {}}

        return response.json()

class Account:
    def __init__(self, client):
        self.client = client

    def get_account_info(self):
        # CORRECTED: The endpoint for v2 asset balance is /asset/balance
        return self.client.request('GET', '/asset/balance', params={}, need_sign=True)

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
        self.max_reconnect_attempts = 15
        self.listen_task = None
        self.on_message_callback = None
        self.subscribed_channels = []

    async def connect(self, on_message_callback):
        self.on_message_callback = on_message_callback
        if self.listen_task:
            self.listen_task.cancel()
        self.listen_task = asyncio.create_task(self._connection_manager())
        sdk_logger.info("WebSocket connection manager started.")

    async def _connection_manager(self):
        while True:
            try:
                sdk_logger.info(f"Attempting to connect to WebSocket... (Attempt {self.reconnect_attempts + 1})")
                async with websockets.connect(self.ws_url) as ws:
                    self.ws = ws
                    self.connected = True
                    self.reconnect_attempts = 0
                    sdk_logger.info("WebSocket connected successfully.")
                    
                    if self.subscribed_channels:
                        await self.subscribe_to_tickers(self.subscribed_channels, resubscribing=True)

                    async for message in self.ws:
                        if self.on_message_callback:
                            await self.on_message_callback(message)
            
            except websockets.exceptions.ConnectionClosed as e:
                sdk_logger.warning(f"WebSocket connection closed: {e}. Reconnecting...")
            except Exception as e:
                sdk_logger.error(f"An unexpected error occurred in WebSocket connection: {e}. Reconnecting...")
            
            self.connected = False
            self.ws = None
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                wait_time = min(60, 5 * self.reconnect_attempts)
                sdk_logger.info(f"Waiting {wait_time} seconds before next reconnect attempt.")
                await asyncio.sleep(wait_time)
            else:
                sdk_logger.critical("Max reconnect attempts reached. Stopping connection manager.")
                break

    async def subscribe_to_tickers(self, symbols: list, resubscribing=False):
        if not self.connected or not self.ws:
            sdk_logger.warning("Cannot subscribe, WebSocket is not connected.")
            return

        if not resubscribing:
            self.subscribed_channels = symbols

        formatted_symbols = [s.replace('/', '') for s in symbols]
        msg = {
            "method": "state.subscribe",
            "params": formatted_symbols,
            "id": int(time.time())
        }
        await self.ws.send(json.dumps(msg))
        sdk_logger.info(f"Sent subscription request for symbols: {formatted_symbols}")

    async def close(self):
        if self.listen_task:
            self.listen_task.cancel()
        if self.connected and self.ws:
            try:
                await self.ws.close()
            except websockets.exceptions.ConnectionClosed:
                pass
        self.connected = False
        sdk_logger.info("WebSocket connection closed by client.")
