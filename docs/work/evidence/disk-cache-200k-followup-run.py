"""Run one follow-up scenario: start the server, sample host/VRAM, run the client, stop everything.

    python build-cuda/run_followup.py baseline|disk|b1|b2

Every run loads the model from scratch and writes its server output, engine log, sampler CSV and client report
under build-cuda/.  It refuses to start when a strata process already exists or the port is taken.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r'C:\tools\strata\.venv\Scripts\python.exe'
SERVER_OUT = ROOT / 'build-cuda'


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def strata_pids():
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "Get-Process | Where-Object { $_.ProcessName -like '*strata*' } | "
                          "Select-Object -ExpandProperty Id"],
                         capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def main():
    name = sys.argv[1]
    runs = {
        'baseline': ('followup-baseline.json', [
            PY, '-u', str(ROOT / 'build-cuda' / 'followup_harness.py'), 'http://127.0.0.1:8099',
            'qwen3.8-flash-next-unsloth-ud-iq4_xs', '--key', 'REDACTED', '--scenario', 'baseline',
            '--main-words', '50000', '--step-words', '20', '--sub-words', '300',
            '--out', str(ROOT / 'build-cuda' / 'followup-baseline-client.json'),
            '--phase', str(ROOT / 'build-cuda' / 'followup-phase.txt')]),
        'disk': ('followup-disk-nohash.json', [
            PY, '-u', str(ROOT / 'build-cuda' / 'followup_harness.py'), 'http://127.0.0.1:8099',
            'qwen3.8-flash-next-unsloth-ud-iq4_xs', '--key', 'REDACTED', '--scenario', 'disk2',
            '--main-words', '50000', '--step-words', '20', '--sub-words', '300',
            '--out', str(ROOT / 'build-cuda' / 'followup-disk-client.json'),
            '--phase', str(ROOT / 'build-cuda' / 'followup-phase.txt')]),
        'b1': ('followup-b1-ram-refusal.json', [
            PY, '-u', str(ROOT / 'tools' / 'conversation_disk_cache_test.py'), 'http://127.0.0.1:8099',
            'qwen3.8-flash-next-unsloth-ud-iq4_xs', '--key', 'REDACTED', '--turns', '3',
            '--main-words', '50000', '--step-words', '20', '--sub-words', '300',
            '--output', str(ROOT / 'build-cuda' / 'followup-b1-report.json'), '--run']),
        'staging512': ('followup-staging512.json', [
            PY, '-u', str(ROOT / 'build-cuda' / 'followup_harness.py'), 'http://127.0.0.1:8099',
            'qwen3.8-flash-next-unsloth-ud-iq4_xs', '--key', 'REDACTED', '--scenario', 'baseline',
            '--main-words', '50000', '--step-words', '20', '--sub-words', '300',
            '--out', str(ROOT / 'build-cuda' / 'followup-staging512-client.json'),
            '--phase', str(ROOT / 'build-cuda' / 'followup-phase.txt')]),
        'b2': ('followup-b2-budget-refusal.json', [
            PY, '-u', str(ROOT / 'tools' / 'conversation_disk_cache_test.py'), 'http://127.0.0.1:8099',
            'qwen3.8-flash-next-unsloth-ud-iq4_xs', '--key', 'REDACTED', '--turns', '3',
            '--main-words', '50000', '--step-words', '20', '--sub-words', '300',
            '--output', str(ROOT / 'build-cuda' / 'followup-b2-report.json'), '--run']),
    }
    config, command = runs[name]
    phase = ROOT / 'build-cuda' / 'followup-phase.txt'
    csv_path = SERVER_OUT / f'followup-{name}-sampler.csv'
    server_out = SERVER_OUT / f'followup-{name}-server.out'
    summary = {'run': name, 'config': config, 'started_utc': utc(), 'command': command}

    if strata_pids():
        raise SystemExit(f'refusing: strata processes already running: {strata_pids()}')
    if not port_free(8099):
        raise SystemExit('refusing: port 8099 is in use')
    phase.write_text('baseline\n', encoding='utf-8')
    if csv_path.exists():
        csv_path.unlink()

    env = dict(os.environ)
    env.pop('STRATA_STATE_HASH', None)
    with open(server_out, 'w', encoding='utf-8', errors='replace') as so:
        server = subprocess.Popen([PY, '-u', '-m', 'serve.server', '--engine', 'strata', '--config',
                                   str(SERVER_OUT / config), '--host', '127.0.0.1', '--port', '8099'],
                                  cwd=str(ROOT), stdout=so, stderr=subprocess.STDOUT, env=env,
                                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        sampler = None
        try:
            deadline = time.time() + 900
            while time.time() < deadline:
                if server.poll() is not None:
                    raise SystemExit(f'the server exited during load ({server.returncode}); see {server_out}')
                if 'ready: http://' in server_out.read_text(encoding='utf-8', errors='replace'):
                    break
                time.sleep(2)
            else:
                raise SystemExit('the server did not become ready in 900 s')
            summary['ready_utc'] = utc()
            time.sleep(5)
            sampler = subprocess.Popen([PY, '-u', str(SERVER_OUT / 'sample_memory.py'), str(csv_path), str(phase)],
                                       cwd=str(ROOT))
            time.sleep(5)
            if name in ('b1', 'b2'):
                phase.write_text('run\n', encoding='utf-8')
            summary['client_start_utc'] = utc()
            started = time.time()
            proc = subprocess.run(command, cwd=str(ROOT), env=env)
            summary['client_seconds'] = round(time.time() - started, 3)
            summary['client_exit'] = proc.returncode
            summary['client_end_utc'] = utc()
            time.sleep(2)
            phase.write_text('stop\n', encoding='utf-8')
            if sampler is not None:
                try:
                    sampler.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    sampler.kill()
            summary['sampler_exit'] = sampler.returncode if sampler is not None else None
        finally:
            phase.write_text('stop\n', encoding='utf-8')
            if sampler is not None and sampler.poll() is None:
                sampler.kill()
            if server.poll() is None:
                try:
                    server.send_signal(signal.CTRL_BREAK_EVENT)
                except OSError:
                    pass
                try:
                    server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/PID', str(server.pid), '/T', '/F'], capture_output=True)
                    server.wait(timeout=30)
    summary['server_exit'] = server.returncode
    for _ in range(30):
        pids = strata_pids()
        if not pids:
            break
        time.sleep(1)
    summary['strata_pids_after'] = strata_pids()
    summary['ended_utc'] = utc()
    out = SERVER_OUT / f'followup-{name}-run.json'
    out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if summary['strata_pids_after']:
        raise SystemExit(f'strata processes remain: {summary["strata_pids_after"]}')


if __name__ == '__main__':
    main()
