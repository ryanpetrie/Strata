"""serve/test_decision_page.py - the Decide tab: the page, the server's flags, the request.

    python -m unittest serve.test_decision_page -v

The decision endpoint (POST /v1/systemone) is a JSON API: a user has to write the
request by hand and read the answer as JSON.  The web app's Decide tab is a client of
it - the page fills in the request, sends it, and shows the answer with the numbers
behind it.  Three things can break without a browser, and they are what these tests
cover:

  * the page's own structure: the tab, the section, and the ids the page uses;
  * the two flags the server adds for the page (/health's "decisions" and
    /v1/status's "systemone");
  * the request the page builds: the example it offers has to be one the endpoint
    accepts, and the limits it checks have to be the endpoint's limits.

No GPU: the engine is the mock.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import systemone as sysone         # noqa: E402
from serve.frontend import ChatTemplate       # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "serve" / "web"


def page(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def example_request() -> dict:
    """The example request the page offers, read out of the page itself."""
    js = page("decide.js")
    match = re.search(r"const DEC_EXAMPLE = `(.*?)`;", js, re.S)
    assert match, "decide.js has no example request"
    return json.loads(match.group(1))


def page_limits() -> dict:
    """The limits the page checks, read out of the page itself."""
    js = page("decide.js")
    out = {}
    for name in ("DEC_MAX_QUESTIONS", "DEC_MAX_OPTIONS", "DEC_MIN_LEVELS",
                 "DEC_MAX_LEVELS", "DEC_MAX_NAME"):
        match = re.search(rf"const {name} = (\d+);", js)
        assert match, f"decide.js has no {name}"
        out[name] = int(match.group(1))
    return out


class TheDecidePage(unittest.TestCase):
    """The tab and the page behind it."""

    def test_the_tab_is_there(self):
        html = page("index.html")
        self.assertIn('id="tab-btn-decide"', html)
        self.assertIn('data-tab="decide"', html)
        self.assertIn('role="tab"', html)
        self.assertIn('aria-controls="view-decide"', html)
        # the tab starts hidden: the endpoint is off by default
        self.assertRegex(html, r'id="tab-btn-decide"[^>]*hidden')

    def test_the_section_is_there(self):
        html = page("index.html")
        self.assertIn('id="view-decide"', html)
        self.assertIn('role="tabpanel"', html)
        self.assertIn('aria-labelledby="tab-btn-decide"', html)

    def test_the_page_is_loaded(self):
        self.assertIn('<script src="web/decide.js"></script>', page("index.html"))

    def test_every_id_the_page_uses_is_on_the_page(self):
        html = page("index.html")
        used = set(re.findall(r'\$\("(dec-[a-z-]+)"\)', page("decide.js")))
        self.assertTrue(used, "the page uses no ids of its own")
        for name in sorted(used):
            self.assertIn(f'id="{name}"', html, f"{name} is used but not on the page")

    def test_the_tab_is_hidden_when_the_endpoint_is_off(self):
        app = page("app.js")
        self.assertIn('tab-btn-decide', app)
        self.assertIn('health.decisions', app)
        # and the tab is one of the app's views
        self.assertIn('"decide"', app)

    def test_the_page_posts_to_the_endpoint(self):
        js = page("decide.js")
        self.assertIn('"v1/systemone"', js)
        self.assertIn('method: "POST"', js)

    def test_the_page_shows_the_servers_own_message(self):
        # a 400 (the request) and a 502 (the model wrote something that did not parse)
        # are shown as the server wrote them: nothing is invented
        js = page("decide.js")
        self.assertIn("error.message", js)
        self.assertIn("The server said", js)


class TheServersFlags(unittest.TestCase):
    """The two flags the server adds for the page."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def test_health_says_the_decisions_are_off_by_default(self):
        self.assertIs(self.get("/health")["decisions"], False)

    def test_health_says_the_decisions_are_on(self):
        self.svc.systemone_enabled = True
        self.assertIs(self.get("/health")["decisions"], True)

    def test_status_says_what_the_endpoint_does(self):
        status = self.get("/v1/status")
        self.assertIn("systemone", status)
        block = status["systemone"]
        self.assertIs(block["enabled"], False)
        self.assertIs(block["diagnostics"], False)
        self.assertEqual(block["multi"], "separate")
        # the temperatures the endpoint is applying: 1.0 is "no temperature"
        self.assertEqual(block["temperature"], {"choice": 1.0, "noul": 1.0, "score": 1.0})

    def test_status_says_the_temperature(self):
        self.svc.systemone_temperature = {"choice": 0.62, "noul": 0.67, "score": 0.77}
        block = self.get("/v1/status")["systemone"]
        self.assertEqual(block["temperature"], {"choice": 0.62, "noul": 0.67, "score": 0.77})

    def test_status_says_the_diagnostics_are_on(self):
        self.svc.systemone_enabled = True
        self.svc.systemone_diagnostics = True
        block = self.get("/v1/status")["systemone"]
        self.assertIs(block["enabled"], True)
        self.assertIs(block["diagnostics"], True)

    def test_the_page_is_served(self):
        with urllib.request.urlopen(self.base + "/web/decide.js", timeout=10) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/javascript", r.headers.get("Content-Type", ""))
            self.assertIn("v1/systemone", r.read().decode())


class TheRequestThePageBuilds(unittest.TestCase):
    """The request: one the endpoint accepts, and the endpoint's own limits."""

    def test_the_example_the_page_offers_is_a_request_the_endpoint_accepts(self):
        request = example_request()
        state, questions = sysone.validate_request(request)   # ValueError: the endpoint's 400
        self.assertTrue(state.strip())
        self.assertEqual(len(questions), 3)
        self.assertEqual([q["type"] for _, q in questions], ["noul", "choice", "score"])

    def test_the_example_has_the_criteria_of_every_type(self):
        criteria = example_request()["questions"]
        self.assertEqual(set(criteria["healthy"]["criteria"]), {"true", "false"})
        self.assertEqual(list(criteria["verdict"]["criteria"]), ["good", "warn", "bad"])
        self.assertEqual(list(criteria["severity"]["criteria"]), ["low", "medium", "high"])

    def test_the_page_checks_the_endpoints_limits(self):
        limits = page_limits()
        self.assertEqual(limits["DEC_MAX_QUESTIONS"], sysone.MAX_QUESTIONS)
        self.assertEqual(limits["DEC_MAX_OPTIONS"], sysone.MAX_OPTIONS)
        self.assertEqual(limits["DEC_MIN_LEVELS"], sysone.MIN_LEVELS)
        self.assertEqual(limits["DEC_MAX_LEVELS"], sysone.MAX_LEVELS)
        self.assertEqual(limits["DEC_MAX_NAME"], sysone.MAX_NAME_LEN)

    def test_a_name_with_a_colon_is_refused_by_both(self):
        # the page checks the names before it sends them: the answers come back as
        # "name: answer" lines, so a name may hold no colon and no line break
        request = example_request()
        request["questions"]["bad:name"] = request["questions"].pop("healthy")
        with self.assertRaises(ValueError):
            sysone.validate_request(request)
        self.assertIn("colon", page("decide.js"))


if __name__ == "__main__":
    unittest.main()
