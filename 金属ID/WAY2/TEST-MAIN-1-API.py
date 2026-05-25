import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://10.10.185.18:30067/v1/extract"
IMAGE_PATH = r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID\02033711E02.jpg"
CONCURRENT = 10      # 并发数
TOTAL_REQUESTS = 100 # 总请求数

def send_request(i):
    start = time.time()
    try:
        with open(IMAGE_PATH, "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            resp = requests.post(URL, files=files, timeout=30)
        elapsed = time.time() - start
        return (i, resp.status_code, elapsed, resp.json())
    except Exception as e:
        elapsed = time.time() - start
        return (i, -1, elapsed, str(e))

def main():
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENT) as pool:
        futures = {pool.submit(send_request, i): i for i in range(1, TOTAL_REQUESTS+1)}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"[{res[0]}] status={res[1]} time={res[2]:.2f}s")

    success = sum(1 for r in results if r[1] == 200)
    fail = TOTAL_REQUESTS - success
    total_time = time.time() - t0
    print(f"\n--- 结果 ---")
    print(f"总数:{TOTAL_REQUESTS} 成功:{success} 失败:{fail}")
    print(f"总耗时:{total_time:.1f}s QPS:{TOTAL_REQUESTS/total_time:.1f}")
    # 可选：计算平均延迟
    times = [r[2] for r in results if r[1]==200]
    if times:
        print(f"平均延迟:{sum(times)/len(times):.2f}s 最长:{max(times):.2f}s 最短:{min(times):.2f}s")

if __name__ == "__main__":
    main()