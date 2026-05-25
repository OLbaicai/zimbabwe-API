import time
import sys
import math
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
URL = "http://10.45.135.56:4560/extract"
IMAGE_PATH = r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID\02033711E02.jpg"
CONCURRENT = 20
TOTAL = 150

# ── 辅助函数：百分位数计算（线性插值） ────────────────
def percentile(sorted_data, p):
    """计算排序列表的第 p 百分位数 (0 <= p <= 100)"""
    if not sorted_data:
        return 0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    # 使用线性插值：索引 = (p/100) * (n-1)
    rank = (p / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_data[lower]
    weight = rank - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

def format_duration(seconds):
    """将秒数格式化为可读字符串"""
    if seconds >= 1:
        return f"{seconds:.2f}s"
    elif seconds >= 0.001:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds*1e6:.1f}μs"

# ── 请求函数 ──────────────────────────────────────────
def send_request(i):
    start = time.time()
    try:
        img_bytes = Path(IMAGE_PATH).read_bytes()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        resp = requests.post(URL, files=files, timeout=30)
        elapsed = time.time() - start
        if resp.status_code == 200:
            return (i, resp.status_code, elapsed, resp.json(), None)
        else:
            # 提取服务端错误详情
            try:
                detail = resp.json().get("detail", resp.text[:200])
            except:
                detail = resp.text[:200]
            return (i, resp.status_code, elapsed, None, f"[{resp.status_code}] {detail}")
    except Exception as e:
        elapsed = time.time() - start
        return (i, -1, elapsed, None, str(e))

# ── 主函数 ────────────────────────────────────────────
def main():
    # 1. 连通性预检
    print(">>> 预检: 检查服务连通性...")
    try:
        r = requests.post(URL,
                          files={"file": ("test.jpg", open(IMAGE_PATH, "rb"), "image/jpeg")},
                          timeout=10)
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
            sample = r.json()
            print(f"    返回样例: {sample}")
        else:
            print(f"    返回内容: {r.text[:300]}")
    except Exception as e:
        print(f"    连接失败: {e}")
        print("请确认服务已启动，并且 URL 正确。如果是本机请尝试 http://127.0.0.1:4560/extract")
        sys.exit(1)

    # 2. 开始压测
    print(f"\n开始压测: 并发 {CONCURRENT}, 总数 {TOTAL}")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENT) as pool:
        futures = {pool.submit(send_request, i): i for i in range(1, TOTAL+1)}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status_str = "OK" if res[1] == 200 else f"FAIL({res[1]})"
            err_str = f" err={res[4]}" if res[4] else ""
            print(f"[{res[0]:03d}] {status_str} time={res[2]:.2f}s{err_str}")

    total_time = time.time() - t0

    # 3. 分类统计
    success_results = [r for r in results if r[1] == 200]
    fail_results = [r for r in results if r[1] != 200]

    success_count = len(success_results)
    fail_count = len(fail_results)

    # 延迟列表（成功请求）
    times = [r[2] for r in success_results]

    # ── 4. 详细报告 ─────────────────────────────────────
    print("\n" + "="*60)
    print("                 压 测 详 细 报 告")
    print("="*60)
    print(f"  目标 URL        : {URL}")
    print(f"  并发数          : {CONCURRENT}")
    print(f"  总请求数        : {TOTAL}")
    print(f"  成功数          : {success_count}")
    print(f"  失败数          : {fail_count}")
    print(f"  成功率          : {success_count/TOTAL*100:.1f}%")
    print(f"  总耗时          : {total_time:.2f}s")
    print(f"  平均 QPS        : {TOTAL/total_time:.2f} req/s")
    if times:
        avg_t = sum(times) / len(times)
        # 服务端实际处理吞吐量（忽略并发等待）粗略估计：
        # 如果请求是瞬时并发的，实际处理能力 = 并发数 / 平均延迟
        theoretical_qps = CONCURRENT / avg_t if avg_t > 0 else 0
        print(f"  平均响应时间    : {format_duration(avg_t)}")
        print(f"  理论最大 QPS    : {theoretical_qps:.2f} (并发/平均延迟)")

    # ── 延迟分布 ────────────────────────────────────────
    if times:
        sorted_times = sorted(times)
        min_t = sorted_times[0]
        max_t = sorted_times[-1]
        avg_t = sum(times) / len(times)
        median = percentile(sorted_times, 50)
        p90 = percentile(sorted_times, 90)
        p95 = percentile(sorted_times, 95)
        p99 = percentile(sorted_times, 99)
        p999 = percentile(sorted_times, 99.9)

        print("\n  ── 响应时间分布 ──")
        print(f"  最小            : {format_duration(min_t)}")
        print(f"  平均            : {format_duration(avg_t)}")
        print(f"  中位数 (P50)    : {format_duration(median)}")
        print(f"  P90             : {format_duration(p90)}")
        print(f"  P95             : {format_duration(p95)}")
        print(f"  P99             : {format_duration(p99)}")
        print(f"  P999            : {format_duration(p999)}")
        print(f"  最大            : {format_duration(max_t)}")

        # 简单分布桶
        print("\n  ── 延迟区间分布 ──")
        buckets = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, float('inf')]
        bucket_labels = ["<100ms", "100-200ms", "200-500ms", "0.5-1s", "1-2s", "2-5s",
                         "5-10s", "10-20s", "20-30s", ">30s"]
        counts = [0]*len(buckets)
        for t in sorted_times:
            for i, bound in enumerate(buckets):
                if t <= bound:
                    counts[i] += 1
                    break
        for label, cnt in zip(bucket_labels, counts):
            if cnt > 0:
                print(f"  {label:12s} : {cnt:4d}  ({cnt/len(times)*100:5.1f}%)")

    # ── 错误分类 ────────────────────────────────────────
    if fail_results:
        print("\n  ── 失败请求分析 ──")
        error_types = Counter()
        for r in fail_results:
            err_msg = r[4] if r[4] else "未知错误"
            # 提取错误类型简写
            if "Connection" in err_msg:
                error_types["连接错误"] += 1
            elif "Timeout" in err_msg:
                error_types["超时"] += 1
            elif "500" in err_msg:
                error_types["服务端500错误"] += 1
            elif "502" in err_msg:
                error_types["网关错误"] += 1
            elif "503" in err_msg:
                error_types["服务不可用"] += 1
            elif "400" in err_msg:
                error_types["客户端请求错误"] += 1
            else:
                error_types["其他"] += 1
            # 打印前3个具体错误信息
            if fail_results.index(r) < 3:
                print(f"  示例错误: {err_msg[:150]}")
        print("\n  错误类型统计:")
        for err_type, cnt in error_types.most_common():
            print(f"    {err_type:16s} : {cnt}")
    else:
        print("\n  ✅ 所有请求均成功。")

    print("="*60)

if __name__ == "__main__":
    main()