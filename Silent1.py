# في ملف Silent1.py

# === Connection Diagnostics ===
async def perform_connection_tests():
    system_status["connection_errors"] = []
    logging.info("--- STARTING DIAGNOSTIC TESTS ---")

    # Test 1: Basic API Ping
    logging.info("Diagnostic 1/3: Testing API Ping to /ping...")
    ping_res = await asyncio.to_thread(coinex_client.market.ping)
    if ping_res.get('code') != 0:
        system_status["connection_errors"].append(f"Ping Failed: {ping_res.get('message', 'Unknown Error')}")
        logging.error("CRITICAL: Ping test failed. Cannot reach CoinEx API.")
        return {"success": False, "errors": system_status["connection_errors"]}
    logging.info("Diagnostic 1/3: Ping Successful.")

    # Test 2: Authenticated, reliable endpoint (/spot/pending-order)
    logging.info("Diagnostic 2/3: Testing a reliable authenticated endpoint (/spot/pending-order)...")
    pending_orders_res = await asyncio.to_thread(coinex_client.account.get_pending_orders)
    if pending_orders_res.get('code') != 0:
        error_msg = pending_orders_res.get('message', str(pending_orders_res))
        system_status["connection_errors"].append(f"Reliable Endpoint Test Failed: {error_msg}")
        logging.error(f"CRITICAL: Authentication test with /spot/pending-order failed. Response: {pending_orders_res}")
        return {"success": False, "errors": system_status["connection_errors"]}
    logging.info("Diagnostic 2/3: Reliable authenticated endpoint test SUCCESSFUL. Signing mechanism is working.")

    # Test 3: The problematic endpoint (/spot/balance)
    logging.info("Diagnostic 3/3: Testing the problematic balance endpoint (/spot/balance)...")
    auth_res = await asyncio.to_thread(coinex_client.account.get_account_info)
    if auth_res.get('code') != 0:
        error_msg = auth_res.get('message', str(auth_res))
        system_status["connection_errors"].append(f"Balance Endpoint Failed: {error_msg}")
        logging.warning(f"WARNING: The /spot/balance endpoint failed as before. Response: {auth_res}")
        # We can decide to continue if the reliable test passed, but for now we fail.
        # This confirms the problem is ONLY with this specific endpoint.
        return {"success": False, "errors": system_status["connection_errors"]}
    
    logging.info("Diagnostic 3/3: Balance endpoint test was successful!")
    system_status["api_authenticated"] = True
    logging.info("--- ALL DIAGNOSTIC TESTS PASSED ---")
    return {"success": True, "errors": []}
