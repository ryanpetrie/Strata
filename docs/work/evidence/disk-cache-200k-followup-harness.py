"""Send the 194K MAIN / small SUB pattern over HTTP with per-request sampler phase markers.

Reuses tools/conversation_disk_cache_test.py's Client and filler so the prompts and timing are the tool's.
Phases:  main1, after_main1, sub1, after_sub1, main2, after_main2, sub2, after_sub2
Writes a JSON with each request's prompt tokens, reuse, HTTP seconds and UTC window.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.conversation_disk_cache_test import Client, filler  # noqa: E402


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def set_phase(path, label):
    Path(path).write_text(label + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('model')
    ap.add_argument('--key', default=None)
    ap.add_argument('--scenario', choices=['disk2', 'baseline'], required=True)
    ap.add_argument('--main-words', type=int, default=50000)
    ap.add_argument('--sub-words', type=int, default=300)
    ap.add_argument('--step-words', type=int, default=20)
    ap.add_argument('--settle', type=float, default=2.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--phase', required=True)
    args = ap.parse_args()

    client = Client(args.url, args.key)
    mains = [{"role": "system", "content": "You are the main agent. " + filler("m", args.main_words)}]
    subs = [{"role": "system", "content": "You are a sub-agent that summarizes. " + filler("s", args.sub_words)}]
    rows = []

    def ask(name, turn, conversation, step):
        label = name.lower() + str(turn)
        set_phase(args.phase, label)
        conversation.append({"role": "user", "content": f"step {turn}: {filler(name[0].lower() + str(turn), step)} "
                                                        "Reply with one word."})
        before = client.metrics()["conversation_cache"]
        start_utc, t0 = utc(), time.time()
        reply = client.chat(args.model, conversation)
        seconds = time.time() - t0
        end_utc = utc()
        after = client.metrics()["conversation_cache"]
        if not reply.get('choices'):
            raise RuntimeError(f'{name} {turn}: missing completion')
        conversation.append({"role": "assistant", "content": reply['choices'][0]['message'].get('content') or 'ok'})
        row = {
            'name': name, 'turn': turn, 'prompt_tokens': (reply.get('usage') or {}).get('prompt_tokens', 0),
            'reused': int(after.get('reused_tokens', 0)) - int(before.get('reused_tokens', 0)),
            'request_reused': after.get('last_reused'), 'request_prompt': after.get('last_prompt'),
            'seconds': round(seconds, 3), 'start_utc': start_utc, 'end_utc': end_utc,
            'disk_saves': after.get('disk_saves'), 'disk_restores': after.get('disk_restores'),
            'disk_errors': after.get('disk_errors'),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        set_phase(args.phase, 'after_' + label)
        time.sleep(args.settle)
        return row

    turn1 = {}
    try:
        turn1['main'] = ask('MAIN', 1, mains, args.step_words)
        turn1['sub'] = ask('SUB', 1, subs, max(8, args.sub_words // 4))
        if args.scenario == 'disk2':
            turn1['main2'] = ask('MAIN', 2, mains, args.step_words)
            turn1['sub2'] = ask('SUB', 2, subs, max(8, args.sub_words // 4))
    finally:
        set_phase(args.phase, 'stop')
    Path(args.out).write_text(json.dumps({'scenario': args.scenario, 'rows': rows,
                                          'cache_after': client.metrics()['conversation_cache']}, indent=2) + '\n',
                              encoding='utf-8')
    print('PASS harness', args.out, flush=True)


if __name__ == '__main__':
    main()
