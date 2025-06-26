# local_client.py
import paramiko
import time
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import BarColumn

from io import StringIO
import csv

class Connection:
    def __init__(self, host, user, password=None, key_filename=None):
        self.host = host
        self.user = user
        self.password = password
        self.key_filename = key_filename
        self.ssh = None
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.console = Console()

    def connect(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            hostname=self.host,
            username=self.user,
            password=self.password,
            key_filename=self.key_filename
        )
        # Запуск удалённого скрипта
        self.stdin, self.stdout, self.stderr = self.ssh.exec_command("python3 remote_gpu_monitor_simple_csv.py")

class Monitor:
    def __init__(self, logging=True, condition = None):
        self.connections = []
        self.console = Console()
        self.load_history = {}

        self.logging = logging
        self.last_log_time = 0
        self.log_file = None
        self.csv_writer = None

        if condition:
            self.conditional_monitoring = True
            self.condition = condition
        else:
            self.conditional_monitoring = False

    def add_connection(self, connection):
        self.connections.append(connection)

    def get_gpus(self):
        gpus = []
        for con in self.connections:
            line = con.stdout.readline()

            f = StringIO(line.strip())
            reader = csv.DictReader(
                f,
                fieldnames=["index", "name", "temperature", "load", "mem_used", "mem_total", "power", "fan"]
            )
            gpu = list(reader)
            gpus.append(gpu)

        return gpus

    def build_table(self, gpus):
        table = Table(title="[bold]Remote GPU Usage[/bold]", box=None)
        table.add_column("Node")
        table.add_column("GPU")
        table.add_column("Name")
        table.add_column("Temp (°C)")
        table.add_column("Load (%)")
        table.add_column("Fan (%)")
        table.add_column("Power (W)")
        table.add_column("Memory Usage")
        table.add_column("Load Graph", justify="left")

        BAR_LENGTH = 20
        HISTORY_LENGTH = 20
        graph_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

        for i, con in enumerate(self.connections):
            node = con.user
            for gpu in gpus[i]:
                index = gpu['index']
                name = gpu['name']
                temp = f"{gpu['temperature']}°C"
                load = f"{gpu['load']}%"
                fan_speed = f"{int(gpu['fan'])}%"
                power = f"{float(gpu['power']):.1f} W"

                if int(gpu['temperature']) > 75:
                    temp = f"[red]{temp}[/red]"

                # mem usage bar
                mem_used = int(float(gpu['mem_used']))
                mem_total = int(float(gpu['mem_total']))
                percent = mem_used / mem_total if mem_total else 0
                filled_length = int(BAR_LENGTH * percent)
                bar = '█' * filled_length + '░' * (BAR_LENGTH - filled_length)

                # load graph
                load_percent = int(gpu['load'])
                self.load_history.setdefault(i, {})
                self.load_history[i].setdefault(index, [])
                self.load_history[i][index].append(load_percent)
                self.load_history[i][index] = self.load_history[i][index][-HISTORY_LENGTH:]

                normalized = [int((p / 100) * len(graph_chars)) for p in self.load_history[i][index]]
                load_graph = ''.join([graph_chars[min(i, len(graph_chars)-1)] for i in normalized]).ljust(HISTORY_LENGTH)


                mem_str = f"{(mem_used/1024.0):.1f}/{(mem_total/1024.0):.1f} GB"
                table.add_row(node, index, name, temp, load, fan_speed, power, f"{mem_str} [{bar}]", load_graph)

        return table

    def start_log(self):
        # Инициализация лог-файла
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.dirname(os.path.abspath(__file__))+"/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{log_dir}/gpu_log_{timestamp}.csv"

        self.log_file = open(log_filename, mode='w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.log_file, fieldnames=[
            'timestamp', 'node', 'gpu_index', 'temperature', 'load',
            'fan_speed', 'power', 'mem_used'
        ])
        self.csv_writer.writeheader()

    def write_log(self, gpus):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, con in enumerate(self.connections):
            node = con.user
            for gpu in gpus[i]:
                self.csv_writer.writerow({
                    'timestamp': current_time,
                    'node': node,
                    'gpu_index': gpu['index'],
                    'temperature': int(gpu['temperature']),
                    'load': int(gpu['load']),
                    'fan_speed': int(gpu['fan']),
                    'power': float(gpu['power']),
                    'mem_used': int(float(gpu['mem_used']))
                })

        self.log_file.flush()  # гарантирует запись на диск

    def run(self):
        if self.logging:
            LOG_INTERVAL = 10
            self.start_log()

        try:
            with Live(refresh_per_second=1) as live:
                headers = self.get_gpus()
                while True:
                    gpus = self.get_gpus()
                    live.update(self.build_table(gpus))

                    if self.logging:
                        now = time.time()
                        if now - self.last_log_time >= LOG_INTERVAL:
                            self.write_log(gpus)
                            self.last_log_time = time.time()

                    if self.conditional_monitoring:
                        self.condition(gpus)


        except KeyboardInterrupt:
            self.console.print("[yellow]Stopping monitor...[/yellow]")
            for con in self.connections:
                con.ssh.close()

if __name__ == "__main__":
    import pickle

    hosts = ["10.162.1.50", "10.162.1.82", "10.162.1.71", "10.162.1.51",
     "10.162.1.91", "10.162.1.92", "10.162.1.93", "10.162.1.94"]

    users = ["usr", "usr2", "usr3", "usr4", "usr5", "usr6", "usr7", "usr8"]

    with open(os.path.dirname(os.path.abspath(__file__))+"/passwords", "rb") as fp:
        passwords = pickle.load(fp)


    monitor = Monitor(logging = False)

    for host, user, password in zip(hosts, users, passwords):
        con = Connection(
            host=host,
            user=user,
            password=password
        )
        con.connect()

        monitor.add_connection(con)

    monitor.run()
