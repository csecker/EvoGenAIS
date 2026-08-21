#!python3

import requests
import time

def wait_for_http_service(url, retry_interval=5, max_retries=50):
    retries = 0
    print("Test5", flush=True)

    while retries < max_retries:
        try:
            response = requests.get(url)
            print(response, flush=True)
            if response.status_code == 200:
                print(f"HTTP service is available at {url}", flush=True)
                return True
        except requests.ConnectionError:
            print(f"HTTP service not available. Retrying in {retry_interval} seconds...", flush=True)
            time.sleep(retry_interval)
        retries += 1
        print("Test6", flush=True)

gpusim_server = 'http://' + 'localhost' + ':' + '5000'
wait_for_http_service(gpusim_server)
