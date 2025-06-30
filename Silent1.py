try:
    # تهيئة عميل CoinEx من الـ SDK
    coinex_client = CoinEx(API_KEY, API_SECRET)
    
    # 1. اختبار الاتصال العام (Ping)
    logging.info("Testing public API endpoint (Ping)...")
    ping_response = coinex_client.market.ping()
    if ping_response.get('code') == 0:
        logging.info(f"Ping successful! Response: {ping_response}")
    else:
        logging.error(f"Ping failed! Response: {ping_response}")
        exit() # الخروج إذا فشل الاتصال الأساسي

    # 2. اختبار المصادقة (Authentication) عن طريق طلب الرصيد
    logging.info("Testing private API endpoint (Account Balance)...")
    account_info = coinex_client.account.get_account_info()
    
    # التحقق من نجاح الطلب
    if account_info.get('code') == 0:
        logging.info("Authentication successful! Account info received.")
        # طباعة جزء من البيانات للتأكيد
        if 'data' in account_info and 'spot_asset' in account_info['data']:
            usdt_asset = account_info['data']['spot_asset'].get('USDT')
            if usdt_asset:
                logging.info(f"USDT Balance: {usdt_asset}")
            else:
                logging.info("No USDT asset found in account.")
        else:
            logging.info(f"Full account data: {account_info.get('data')}")
    else:
        logging.error(f"Authentication failed! API returned an error.")
        logging.error(f"Response Code: {account_info.get('code')}")
        logging.error(f"Response Message: {account_info.get('message')}")

except Exception as e:
    logging.critical(f"A critical error occurred: {e}", exc_info=True)
