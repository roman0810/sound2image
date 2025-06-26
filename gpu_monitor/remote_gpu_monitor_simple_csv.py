# remote_gpu_monitor_simple_csv.py
import time
import subprocess
import csv
import re
import sys

def clean_output(output):
    ansi_escape = re.compile(r'(?:\x1B[@-Z\\-_]|[\x08\x0E\x0F\x1B])/')
    return ansi_escape.sub('', output).strip()

def get_gpu_csv_line():
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,fan.speed',
        '--format=csv,noheader,nounits'
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return ""
        return clean_output(result.stdout)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        return ""

if __name__ == "__main__":
    print("index,name,temperature,load,mem_used,mem_total", flush=True)
    while True:
        csv_data = get_gpu_csv_line()
        if csv_data:
            for line in csv_data.strip().split('\n'):
                print(line.strip(), flush=True)
        time.sleep(1)