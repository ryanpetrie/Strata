"""End-to-end check of the disk conversation cache (L2) against one running Strata server.

It alternates a long MAIN conversation with a short SUB conversation, so MAIN has to leave the RAM cache and come back,
and reads /metrics before and after every request: the check needs at least one disk spill, at least one disk restore
and non-zero prompt reuse, or it fails and says what the server reported. A JSON report is written either way.

THIS TOOL HAS NOT BEEN RUN AGAINST A MODEL. It follows the HTTP patterns of tools/conversation_cache_http_smoke.py and
tools/harness_reuse_probe.py; the run that produces numbers must be yours. Start an idle test server with the disk tier
on (in the run config's "args": "--conversation-cache-disk-mib", "32768" and "--conversation-cache-spill-dir",
"<folder>"; add "--conversation-cache-disk-only" to send every save to the disk tier), then:

    python tools/conversation_disk_cache_test.py http://127.0.0.1:8080 <model> --output report.json --run

Dry-run by default. It refuses to claim a measurement when /metrics says the tier is off: no prompt is sent then.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def filler(tag, words):
    """Deterministic text that differs per tag, so MAIN and SUB share no prefix."""
    return " ".join(f"{tag}{i % 977}" for i in range(words))


class Client:
    def __init__(self, url, key=None):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": "Bearer " + key} if key else {}

    def _get(self, path):
        with urllib.request.urlopen(urllib.request.Request(self.url + path, headers=self.headers), timeout=30) as response:
            return json.load(response)

    def metrics(self):
        return self._get("/metrics")

    def chat(self, model, messages, count=8):
        body = {"model": model, "messages": messages, "max_tokens": count, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", **self.headers})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)


DISK_KEYS = ("disk_saves", "disk_restores", "disk_evictions", "disk_refusals", "disk_errors", "disk_bytes",
             "disk_entries", "disk_last_save_ms", "disk_last_restore_ms", "disk_last_restore_mib_s", "last_tier",
             "disk_paged", "disk_pool_mib", "disk_last_write_mib", "disk_last_reuse_mib", "disk_last_write_share",
             "disk_written_mib", "disk_reused_mib", "disk_compactions", "disk_compacted_mib", "disk_demotions")
INFO_KEYS = ("model", "conversation_cache_mib", "conversation_cache_slots", "conversation_disk_mib",
             "conversation_disk_dir", "conversation_disk_slots", "conversation_disk_paged",
             "conversation_disk_pool_bytes")


def run(client, model, turns, main_words, sub_words, step_words=None):
    first = client.metrics()
    require(first["live"]["state"] == "idle" and first["live"]["queued"] == 0, "the server is busy; use an idle one")
    cache = first["conversation_cache"]
    require(cache.get("disk_enabled"), "the disk tier is off (/metrics conversation_cache.disk_enabled is false): "
                                       "no measurement is claimed; start the server with --conversation-cache-disk-mib "
                                       "above 0 and a --conversation-cache-spill-dir")
    info = first["engine"]
    before = dict(cache)
    mains = [{"role": "system", "content": "You are the main agent. " + filler("m", main_words)}]
    subs = [{"role": "system", "content": "You are a sub-agent that summarizes. " + filler("s", sub_words)}]
    rows = []

    def ask(name, turn, conversation, step):
        conversation.append({"role": "user", "content": f"step {turn}: {filler(name[0].lower() + str(turn), step)} "
                                                        "Reply with one word."})
        cache_before = client.metrics()["conversation_cache"]
        started = time.time()
        reply = client.chat(model, conversation)
        seconds = time.time() - started
        cache_after = client.metrics()["conversation_cache"]
        require(bool(reply.get("choices")), f"{name}: missing completion")
        conversation.append({"role": "assistant", "content": reply["choices"][0]["message"].get("content") or "ok"})
        prompt = (reply.get("usage") or {}).get("prompt_tokens", 0)
        reused = int(cache_after.get("reused_tokens", 0)) - int(cache_before.get("reused_tokens", 0))
        # /metrics reports the last request's figures as last_reused / last_prompt (it does not list requests).
        recent = {"reused": cache_after.get("last_reused"), "prompt_tokens": cache_after.get("last_prompt")}
        rows.append({"name": name, "turn": turn, "prompt_tokens": prompt, "reused": reused,
                     "request_reused": recent.get("reused"), "request_prompt": recent.get("prompt_tokens"),
                     "seconds": seconds})
        return reused

    main_step = max(16, main_words // 8) if step_words is None else max(1, step_words)
    for turn in range(1, turns + 1):
        ask("MAIN", turn, mains, main_step)
        if turn > 1:
            require(rows[-1]["reused"] > 0, f"MAIN turn {turn} reused no prompt at all: neither cache restored it")
        ask("SUB", turn, subs, max(8, sub_words // 4))
    last = client.metrics()["conversation_cache"]
    saves = int(last.get("disk_saves", 0)) - int(before.get("disk_saves", 0))
    restores = int(last.get("disk_restores", 0)) - int(before.get("disk_restores", 0))
    require(saves >= 1, "no disk spill was reported (conversation_cache.disk_saves did not move)")
    require(restores >= 1, "no disk restore was reported (conversation_cache.disk_restores did not move)")
    return {"passed": True, "url": client.url, "model": model,
            "engine": {key: info.get(key) for key in INFO_KEYS},
            "cache_before": {key: before.get(key) for key in DISK_KEYS + ("refusals", "last_refusal")},
            "cache_after": {key: last.get(key) for key in DISK_KEYS + ("refusals", "last_refusal")},
            "spilled_this_run": saves, "restored_this_run": restores,
            "pool": {"paged": last.get("disk_paged"), "mib": last.get("disk_pool_mib"),
                     "written_mib": last.get("disk_written_mib"), "reused_mib": last.get("disk_reused_mib"),
                     "last_write_share": last.get("disk_last_write_share"),
                     "compactions": last.get("disk_compactions"),
                     "compacted_mib": last.get("disk_compacted_mib")},
            "requests": rows,
            "note": "Numbers from this run only. The tool's author did not run it against a model; do not quote them "
                    "as Strata's disk-cache speed, write amplification or memory without your own measurement and "
                    "hardware."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="base URL of an idle server with the disk tier on")
    parser.add_argument("model", help="model id, as /v1/models reports it")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report; existing paths refused")
    parser.add_argument("--key", default=None, help="API key, when the server has one")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--main-words", type=int, default=3000,
                        help="words in MAIN's system prompt; raise it for a long-conversation run")
    parser.add_argument("--sub-words", type=int, default=400, help="words in SUB's system prompt")
    parser.add_argument("--step-words", type=int, default=None,
                        help="words in each MAIN user turn (default: main_words // 8, at least 16); use a small "
                             "value when main_words is near the context limit")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.turns < 2:
        parser.error("--turns 2 or more (the first MAIN turn cannot be restored)")
    if not args.run:
        print("Dry run: MAIN/SUB alternation, disk spill and restore counters, prompt reuse. No network traffic.")
        return
    with args.output.open("x", encoding="utf-8") as output:
        try:
            result = run(Client(args.url, args.key), args.model, args.turns, args.main_words, args.sub_words,
                         args.step_words)
        except Exception as error:
            json.dump({"passed": False, "error": str(error)}, output, indent=2)
            output.write("\n")
            raise
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"PASS: disk spill, disk restore and prompt reuse observed. Last write: "
          f"{result['pool']['last_write_share']}% new. Report: {args.output}")


if __name__ == "__main__":
    main()
