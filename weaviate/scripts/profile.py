import os
import multiprocessing as mp
import time
import json
import numpy as np
import argparse
import csv
import psutil
import sys
import re

def extract_last_digit(name: str) -> int:
    """Return the last digit (or number) before .event in filename."""
    base = name.rsplit(".event", 1)[0]   # strip the suffix
    match = re.search(r"(\d+)$", base)   # look for trailing digits
    return int(match.group(1)) if match else -1   # -1 if none found



def monitorSystem(batch=-1, interval=1.0, stop_event=None):
    rank = sys.argv[1]
    platform = sys.argv[2]
    

    cpuNum = len(psutil.cpu_percent(interval=1.0,percpu=True))
    headers = ['timestamp','max_rss','total_rss','disk_read','disk_write','net_recv','net_sent',
                "systemCPU"
                ] + [f"cpu{i}" for i in range(cpuNum)]
    
    if platform == "POLARIS":
        from pynvml import (
            nvmlInit,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetName,
            nvmlShutdown
        )
        nvmlInit()

        headers += [
            "gpu0_util", "gpu0_mem","gpu0_totalMem",
            "gpu1_util", "gpu1_mem","gpu1_totalMem",
            "gpu2_util", "gpu2_mem","gpu2_totalMem",
            "gpu3_util", "gpu3_mem","gpu3_totalMem",
        ]

    event_row = ["event" for i in range(len(headers) - 1)]

    stats = [
        headers
        
    ]
    
    seen = []
    last_flush = time.time()
    while True:

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cpu_percent = psutil.cpu_percent(interval=1.0,percpu=True)
            cpu_percent = [np.mean(cpu_percent)] + cpu_percent

            mem = psutil.virtual_memory()
            total_rss = sum(proc.memory_info().rss for proc in psutil.process_iter(attrs=['memory_info']))
            system_memory_total = mem.total
            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            row = [timestamp,total_rss,system_memory_total,disk.read_bytes, disk.write_bytes,net.bytes_recv,net.bytes_sent]
 
            row.extend(cpu_percent)

            if platform == "POLARIS":
                for i in range(4):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    util = nvmlDeviceGetUtilizationRates(handle)
                    meminfo = nvmlDeviceGetMemoryInfo(handle)

                    row.extend([util.gpu, meminfo.used, meminfo.total])
            
            stats.append(row)
        except:
            pass

        
        event_files = [entry.name for entry in os.scandir("./runtime_state/") if entry.is_file() and entry.name.endswith(".event")]

        if event_files:
            sorted_events = sorted(event_files, key=extract_last_digit)
            for entry in sorted_events:
                prefix = entry.rsplit(".event", 1)[0]
                if prefix not in seen:
                    # print("New Event detected", flush=True)
                    stats.append([prefix] + event_row)
                    seen.append(prefix)
     
        
        if time.time() - last_flush > 60:
            with open(f"{rank}_system_stats.csv", mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(stats)
            
            last_flush = time.time()
        
        if os.path.exists("./runtime_state/flag.txt"):
            # print("Exit loop", flush=True)
            break

    with open(f"{rank}_system_stats_final.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(stats)
    try:
        os.remove(f"{rank}_system_stats.csv")
    except FileNotFoundError:
        pass

monitorSystem()
