#!/usr/bin/env python
"""CesiumGPU 离线依赖服务一键管理。

覆盖示例页依赖的三个本地服务（全部当前需手工启动的问题由此收口）：

  8890  XYZ 影像瓦片（E:/maptiles/Mapnik z0-16，.omc/tmp/maptiles_server.py）
  8888  地形瓦片 stand-in（E:/gis_data/terrain/terrain_hubei，
        .omc/tmp/terrain_server.py，quantized-mesh）
  8892  空间分析后台（D:/work/离线地图服务/backend，FastAPI + PostGIS；
        数据库为 Docker 容器 geoserver_postgres，宿主端口 5433）

用法：
  python scripts/start-offline-services.py           # 启动全部（幂等，已在跑则跳过）
  python scripts/start-offline-services.py status     # 只查状态
  python scripts/start-offline-services.py stop       # 停三个本地服务（不动 Postgres 容器）

8892 依赖 Postgres；Docker Desktop 未运行时给出提示并继续启动其余服务
（FastAPI 启动期不强制连库，但空间分析端点会失败）。
"""
import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request

# bash/UTF-8 终端下默认 GBK 管道编码会把中文打成乱码
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = r"D:\work\离线地图服务\backend"
LOGDIR = os.path.join(REPO, ".omc", "tmp", "services")
OFFLINE_MAP = r"D:\work\离线地图服务"

SERVICES = [
    {
        "name": "tiles-8890",
        "port": 8890,
        "url": "http://127.0.0.1:8890/",
        # 纯静态服务器：任意路径 200/404 都算进程活着
        "cmd": [sys.executable, os.path.join(REPO, ".omc", "tmp", "maptiles_server.py")],
        "cwd": REPO,
    },
    {
        "name": "terrain-8888",
        "port": 8888,
        "url": "http://127.0.0.1:8888/",
        "cmd": [sys.executable, os.path.join(REPO, ".omc", "tmp", "terrain_server.py")],
        "cwd": REPO,
    },
    {
        "name": "spatial-8892",
        "port": 8892,
        "url": "http://127.0.0.1:8892/health",
        # 不用 `-m uvicorn`：venv 依赖的 uv cpython-3.12 标准库损坏
        # （runpy → machinery → NamespaceLoader ImportError，-c import 不触发），
        # 编程入口 uvicorn.run() 与 run_server.py 同模式，绕开 runpy。
        "cmd": [
            os.path.join(BACKEND, ".venv", "Scripts", "python.exe"),
            "-c",
            "import uvicorn; uvicorn.run('app.main:app', host='127.0.0.1', port=8892)",
        ],
        "cwd": BACKEND,
    },
]


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def http_alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError:
        return True  # 有 HTTP 应答 = 进程活着（404 等路由问题不算死）
    except Exception:
        return False


def pids_on_port(port: int):
    """netstat 反查端口 PID 列表（Windows）。"""
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING":
            if parts[1].endswith(f":{port}"):
                pids.append(parts[4])
    return pids


def is_python_pid(pid: str) -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return False
    return "python" in out.lower() or "uvicorn" in out.lower()


def docker_daemon_up() -> bool:
    try:
        r = subprocess.run(["docker", "ps"], capture_output=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


def pick_backend_python() -> str | None:
    """选一个能 import uvicorn 的解释器。

    venv 依赖 uv 管理的基础解释器，偶发 SRE/导入竞态（杀毒扫描、缓存
    竞争），因此每个候选重试并使用清理过 PYTHON* 的子环境。
    """
    candidates = [
        # 系统 python 优先：backend 两个 venv 的基础解释器都是 uv 管理的
        # cpython-3.12，其标准库安装损坏（SRE/NamespaceLoader 间歇 ImportError），
        # 而系统 Python 的依赖（uvicorn/fastapi/shapely 等）已验证齐全。
        sys.executable,
        os.path.join(BACKEND, "venv", "Scripts", "python.exe"),
        os.path.join(BACKEND, ".venv", "Scripts", "python.exe"),
        r"D:\Program\python\python.exe",
    ]
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    for py in candidates:
        if not os.path.isfile(py):
            print(f"  候选跳过（不存在）: {py}")
            continue
        last_err = ""
        for attempt in range(3):
            try:
                r = subprocess.run(
                    # import re 显式覆盖 SRE 损坏路径（.venv 曾漏检）
                    [py, "-c", "import re, uvicorn, fastapi"],
                    capture_output=True, timeout=60, env=env,
                )
                if r.returncode == 0:
                    if attempt:
                        print(f"  {py} 第 {attempt + 1} 次尝试通过（前次瞬态失败）")
                    return py
                last_err = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1] or "(无 stderr)"
            except subprocess.TimeoutExpired:
                last_err = "超时(60s)"
            except Exception as e:
                last_err = str(e)
        print(f"  候选失败: {py} — {last_err}")
    return None


def ensure_postgres() -> bool:
    """PostGIS 容器（geoserver_postgres，宿主 5433）。"""
    if port_open(5433):
        return True
    if not docker_daemon_up():
        print("[postgres] Docker daemon 未运行——请启动 Docker Desktop 后重跑")
        return False
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "postgres"],
            cwd=OFFLINE_MAP, capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        print("[postgres] docker 命令不存在，请安装 Docker Desktop")
        return False
    except subprocess.TimeoutExpired:
        print("[postgres] docker compose 超时")
        return False
    for _ in range(30):
        if port_open(5433):
            return True
        time.sleep(1)
    print("[postgres] 5433 未就绪：docker ps / docker logs 查看容器")
    return False


def spawn_detached(cmd, cwd, logpath):
    """启动脱离父进程生命周期的后台服务。

    DETACHED_PROCESS 不足以逃出调用者的 Job Object（代理/沙箱终端会在
    命令结束时 kill-on-close 收割整棵进程树——曾导致"启动成功"的服务
    在脚本退出后全部消失）。CREATE_BREAKAWAY_FROM_JOB 需要作业对象许可，
    不被允许时退回普通 detached（用户在普通终端跑脚本时无此问题）。
    """
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    stamp = f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} {cmd[0]} =====\n"
    try:
        f = open(logpath, "ab")
        f.write(stamp.encode("utf-8"))
        return subprocess.Popen(
            cmd, cwd=cwd, stdout=f, stderr=f,
            creationflags=flags | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000),
            close_fds=True,
        )
    except OSError:
        return subprocess.Popen(
            cmd, cwd=cwd, stdout=open(logpath, "ab"), stderr=open(logpath, "ab"),
            creationflags=flags,
            close_fds=True,
        )


def start_one(svc) -> bool:
    if port_open(svc["port"]):
        print(f"[{svc['name']}] 已在运行（端口 {svc['port']}），跳过")
        return True
    os.makedirs(LOGDIR, exist_ok=True)
    spawn_detached(svc["cmd"], svc["cwd"], os.path.join(LOGDIR, f"{svc['name']}.log"))
    deadline = time.time() + (60 if svc["port"] == 8892 else 15)
    while time.time() < deadline:
        time.sleep(1)
        if http_alive(svc["url"]):
            print(f"[{svc['name']}] 启动成功 → {svc['url']}  "
                  f"(日志 .omc/tmp/services/{svc['name']}.log)")
            return True
    print(f"[{svc['name']}] 启动失败，看日志 .omc/tmp/services/{svc['name']}.log")
    return False


def do_start() -> int:
    ok = True
    print("[postgres] 检查 PostGIS（Docker geoserver_postgres :5433）...")
    if not ensure_postgres():
        ok = False
        print("[postgres] 跳过数据库；spatial-8892 的空间分析端点将不可用")

    backend_py = pick_backend_python()
    if backend_py is None:
        print("[spatial-8892] 找不到可用的 backend 解释器（uvicorn/fastapi 均导入失败）")
        ok = False
    else:
        SERVICES[2]["cmd"][0] = backend_py

    for svc in SERVICES:
        ok &= start_one(svc)
    return 0 if ok else 1


def do_status() -> int:
    ok = True
    rows = [("postgres-5433", port_open(5433))]
    rows += [(s["name"], http_alive(s["url"])) for s in SERVICES]
    for name, alive in rows:
        print(f"{'OK ' if alive else 'DOWN'} {name}")
        ok &= alive
    return 0 if ok else 1


def do_stop() -> int:
    for svc in SERVICES:
        pids = pids_on_port(svc["port"])
        killed = False
        for pid in pids:
            if is_python_pid(pid):
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                print(f"[{svc['name']}] 已终止 PID {pid}")
                killed = True
        if not killed:
            print(f"[{svc['name']}] 无运行实例（或端口被非 python 进程占用，未动）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", nargs="?", default="start",
                    choices=["start", "status", "stop"])
    args = ap.parse_args()
    if args.command == "start":
        return do_start()
    if args.command == "status":
        return do_status()
    return do_stop()


if __name__ == "__main__":
    sys.exit(main())
