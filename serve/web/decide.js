// serve/web/decide.js - the Decide tab: POST /v1/systemone, the decision endpoint.
//
// The page is a client of the endpoint.  It fills in a request, sends it, and shows the
// answer: the pick, the distribution over the options, the confidence, and - when the
// server reports them - the numbers behind them.  Plain JavaScript, no framework, no
// network beyond this server.  The helpers ($ , esc, fmt, toast, copyText, store,
// headers) are app.js's.
//
// The rules the endpoint enforces live in serve/systemone.py.  The page checks them
// first, so a mistake is said in the page instead of coming back as a 400.
"use strict";

const DEC_MAX_QUESTIONS = 10;
const DEC_MAX_OPTIONS = 255;
const DEC_MIN_LEVELS = 2;
const DEC_MAX_LEVELS = 10;
const DEC_MAX_NAME = 128;

// An example request, as JSON text, so the page and the tests read the same thing.
const DEC_EXAMPLE = `{
  "state": "The service on port 8080 answered 42 requests in the last minute and none of them failed.",
  "questions": {
    "healthy": {
      "type": "noul",
      "instructions": "Is the service running normally?",
      "criteria": {"true": "the service answers requests without errors", "false": "the service is slow or failing"}
    },
    "verdict": {
      "type": "choice",
      "instructions": "How would you describe the service?",
      "criteria": {"good": "it works as expected", "warn": "it works, but something is off", "bad": "it is broken"}
    },
    "severity": {
      "type": "score",
      "instructions": "How bad is it?",
      "criteria": {"low": "no problem", "medium": "minor", "high": "critical"}
    }
  }
}`;

const decQuestions = [];      // the question blocks on the page
let decSent = [];             // the questions of the last request, in order
let decFlags = null;          // what /v1/status says about the endpoint
let decTried = false;         // has the user pressed Decide yet?
let decTimer = null;

// ------------------------------------------------------------------ the form

// One question block: the name, the type, the instructions, and the criteria the type
// asks for.
function decNewQuestion() {
  const el = document.createElement("div");
  el.className = "dec-question";
  el.innerHTML = `
    <div class="dec-question__head">
      <span class="card-title dec-question__title"></span>
      <button type="button" class="st-btn st-btn--icon dec-remove" title="Remove this question" aria-label="Remove this question">
        <svg class="st-icon"><use href="web/sprite.svg#i-trash"/></svg></button>
    </div>
    <div class="dec-grid">
      <label class="st-field"><span class="st-label">Name <output>no colon, no line break</output></span>
        <input class="st-input dec-name" type="text" placeholder="verdict" autocomplete="off"></label>
      <label class="st-field"><span class="st-label">Type</span>
        <select class="st-input dec-type">
          <option value="noul">noul &mdash; yes or no</option>
          <option value="choice">choice &mdash; one of the options</option>
          <option value="score">score &mdash; one of the levels</option>
        </select></label>
    </div>
    <label class="st-field"><span class="st-label">Instructions <output>the model reads them literally</output></span>
      <textarea class="st-input dec-instructions" rows="2" placeholder="Is the service running normally?"></textarea></label>
    <div class="dec-criteria"></div>`;
  const q = {
    el,
    name: el.querySelector(".dec-name"),
    type: el.querySelector(".dec-type"),
    instructions: el.querySelector(".dec-instructions"),
    criteria: el.querySelector(".dec-criteria"),
  };
  q.type.onchange = () => { decCriteria(q); decRefresh(); };
  q.name.oninput = q.instructions.oninput = decRefresh;
  el.querySelector(".dec-remove").onclick = () => {
    q.el.remove();
    decQuestions.splice(decQuestions.indexOf(q), 1);
    decRefresh();
  };
  decQuestions.push(q);
  $("dec-questions").append(el);
  decCriteria(q);
  return q;
}

// The criteria a type asks for: the two halves of a yes/no, or the options (choice) or
// the levels (score) with a name and an optional description each.
function decCriteria(q, rows = null) {
  q.criteria.replaceChildren();
  if (q.type.value === "noul") {
    q.criteria.innerHTML = `
      <div class="dec-grid">
        <label class="st-field"><span class="st-label">Yes means <output>optional</output></span>
          <input class="st-input dec-yes" type="text" placeholder="the service answers requests without errors"></label>
        <label class="st-field"><span class="st-label">No means <output>optional</output></span>
          <input class="st-input dec-no" type="text" placeholder="the service is slow or failing"></label>
      </div>`;
    q.criteria.querySelector(".dec-yes").oninput = q.criteria.querySelector(".dec-no").oninput = decRefresh;
    return;
  }
  const what = q.type.value === "choice" ? "option" : "level";
  q.criteria.innerHTML = `
    <span class="dec-hint">${q.type.value === "choice"
      ? "The model answers with the option's name."
      : "The levels, lowest first. The model answers with the level's name."}</span>
    <div class="dec-rows"></div>
    <div class="dec-addrow">
      <button type="button" class="st-btn st-btn--secondary dec-add-row">Add an ${what}</button>
      <span class="muted small dec-rows-count"></span>
    </div>`;
  q.criteria.querySelector(".dec-add-row").onclick = () => { decRow(q); decRefresh(); };
  for (const [name, desc] of rows || [[], []]) decRow(q, name, desc);
}

// One option or level: the name the model answers with, and what it means.
function decRow(q, name = "", desc = "") {
  const what = q.type.value === "choice" ? "option" : "level";
  const row = document.createElement("div");
  row.className = "dec-row";
  row.innerHTML = `
    <input class="st-input dec-row__name" type="text" placeholder="${what} name" value="${esc(name)}" autocomplete="off" aria-label="${what} name">
    <input class="st-input dec-row__desc" type="text" placeholder="description (optional)" value="${esc(desc)}" autocomplete="off" aria-label="description (optional)">
    <button type="button" class="st-btn st-btn--icon" title="Remove this ${what}" aria-label="Remove this ${what}">
      <svg class="st-icon"><use href="web/sprite.svg#i-trash"/></svg></button>`;
  row.querySelector(".dec-row__name").oninput = row.querySelector(".dec-row__desc").oninput = decRefresh;
  row.querySelector("button").onclick = () => { row.remove(); decRefresh(); };
  q.criteria.querySelector(".dec-rows").append(row);
  return row;
}

// ------------------------------------------------------------------ the request

// The body the page would send: the state and the questions.  The first problem found
// is the error; the rest is built anyway, so the preview can show it.
function decBuild() {
  let error = "", urgent = false;
  const fail = (message, now = false) => { if (!error) { error = message; urgent = now; } };
  // the state
  const text = $("dec-state").value;
  let state = text;
  if ($("dec-state-json").getAttribute("aria-checked") === "true") {
    if (!text.trim()) {
      fail("The state is empty.");
    } else {
      try {
        state = JSON.parse(text);
      } catch (e) {
        // a broken JSON is said right away: nothing can be sent with it
        fail("The state is not valid JSON: " + e.message, true);
      }
      if (state && typeof state === "object") {
        if (Array.isArray(state) ? !state.length : !Object.keys(state).length) fail("The state is empty.");
      } else if (!error) {
        fail("The state must be a JSON object or an array.");
      }
    }
  } else if (!text.trim()) {
    fail("The state is empty.");
  }
  // the questions
  const questions = [];
  const names = new Set();
  if (!decQuestions.length) fail("There is no question to ask.");
  if (decQuestions.length > DEC_MAX_QUESTIONS)
    fail(`There are ${decQuestions.length} questions; the endpoint takes at most ${DEC_MAX_QUESTIONS}.`);
  decQuestions.forEach((q, i) => {
    const name = q.name.value.trim();
    if (!name) fail(`Question ${i + 1} needs a name.`);
    else if (name.length > DEC_MAX_NAME)
      fail(`The name of question ${i + 1} is longer than ${DEC_MAX_NAME} characters.`);
    else if (name.includes(":") || /[\r\n]/.test(name))
      fail(`The name of question ${i + 1} holds a colon or a line break. The answers come back as "name: answer" lines, so a name may hold neither.`);
    else if (names.has(name))
      fail(`Two questions are called "${name}". Every question needs its own name.`);
    names.add(name);
    const instructions = q.instructions.value.trim();
    if (!instructions) fail(`Question ${i + 1} needs instructions: the model reads them literally.`);
    const definition = {type: q.type.value, instructions};
    const criteria = decCriteriaBody(q, i, fail);
    if (criteria.length) definition.criteria = criteria;
    questions.push([name || `question ${i + 1}`, definition]);
  });
  return {error, urgent, state, questions};
}

// The criteria of one question, as the pairs the endpoint takes: {"true": ..., "false": ...}
// for a yes/no, and {name: description or null} for the options and the levels.
function decCriteriaBody(q, index, fail) {
  if (q.type.value === "noul") {
    const pairs = [];
    const yes = q.criteria.querySelector(".dec-yes").value.trim();
    const no = q.criteria.querySelector(".dec-no").value.trim();
    if (yes) pairs.push(["true", yes]);
    if (no) pairs.push(["false", no]);
    return pairs;
  }
  const what = q.type.value === "choice" ? "option" : "level";
  const pairs = [];
  const seen = new Set();
  for (const row of q.criteria.querySelectorAll(".dec-row")) {
    const name = row.querySelector(".dec-row__name").value.trim();
    const desc = row.querySelector(".dec-row__desc").value.trim();
    if (!name) {
      if (desc) fail(`A row of question ${index + 1} has a description but no ${what} name.`);
      continue;                                   // an empty row is not sent
    }
    if (name.length > DEC_MAX_NAME)
      fail(`A ${what} name of question ${index + 1} is longer than ${DEC_MAX_NAME} characters.`);
    else if (name.includes(":") || /[\r\n]/.test(name))
      fail(`A ${what} name of question ${index + 1} holds a colon or a line break. The answers come back as "name: answer" lines, so a name may hold neither.`);
    else if (seen.has(name)) fail(`Two ${what}s of question ${index + 1} are called "${name}".`);
    seen.add(name);
    pairs.push([name, desc || null]);
  }
  if (q.type.value === "choice") {
    if (!pairs.length) fail(`Question ${index + 1} needs at least one option.`);
    if (pairs.length > DEC_MAX_OPTIONS)
      fail(`Question ${index + 1} has ${pairs.length} options; the endpoint takes at most ${DEC_MAX_OPTIONS}.`);
  } else {
    if (pairs.length < DEC_MIN_LEVELS) fail(`Question ${index + 1} needs at least ${DEC_MIN_LEVELS} levels.`);
    if (pairs.length > DEC_MAX_LEVELS)
      fail(`Question ${index + 1} has ${pairs.length} levels; the endpoint takes at most ${DEC_MAX_LEVELS}.`);
  }
  return pairs;
}

// The body as JSON text.  It is written by hand: JSON.stringify puts the integer-like
// keys first ("2" before "10"), so a map keyed by the user's own names can come out in
// another order than it went in - and the order of the questions, the options and the
// levels is the order the model reads them in.
function decBodyText(state, questions) {
  const qs = questions.map(([name, def]) => {
    const parts = [`      "type": ${JSON.stringify(def.type)}`,
                   `      "instructions": ${JSON.stringify(def.instructions)}`];
    if (def.criteria) parts.push(`      "criteria": ${decMapText(def.criteria, "      ")}`);
    return `    ${JSON.stringify(name)}: {\n${parts.join(",\n")}\n    }`;
  });
  return ["{", `  "state": ${JSON.stringify(state)},`, `  "questions": {`,
          qs.join(",\n"), "  }", "}"].join("\n");
}

function decMapText(pairs, pad) {
  if (!pairs.length) return "{}";
  const inner = pad + "  ";
  const body = pairs.map(([k, v]) => `${inner}${JSON.stringify(k)}: ${v === null ? "null" : JSON.stringify(v)}`);
  return `{\n${body.join(",\n")}\n${pad}}`;
}

// ------------------------------------------------------------------ the page

function decShowError(message) {
  const el = $("dec-error");
  el.textContent = message || "";
  el.hidden = !message;
}

function decPct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : "–";
}

// The preview, the counts and the error line: everything the page shows about the form.
function decRefresh() {
  $("dec-count").textContent = `${decQuestions.length} of ${DEC_MAX_QUESTIONS} questions`;
  $("dec-add").disabled = decQuestions.length >= DEC_MAX_QUESTIONS;
  decQuestions.forEach((q, i) => {
    q.el.querySelector(".dec-question__title").textContent = `Question ${i + 1}`;
    q.el.querySelector(".dec-remove").setAttribute("aria-label", `Remove question ${i + 1}`);
    const count = q.criteria.querySelector(".dec-rows-count");
    if (!count) return;
    const rows = [...q.criteria.querySelectorAll(".dec-row")];
    const what = q.type.value === "choice" ? "options" : "levels";
    const named = rows.filter(r => r.querySelector(".dec-row__name").value.trim()).length;
    const empty = rows.length - named;
    count.textContent = `${named} ${what}` +
                        (empty ? ` (${empty} empty ${empty === 1 ? "row is" : "rows are"} not sent)` : "");
  });
  const built = decBuild();
  $("dec-preview").textContent = decBodyText(built.state, built.questions);
  // the error is said when the user sends the request, and right away when the state
  // cannot be read at all
  if (decTried || built.urgent) decShowError(built.error);
  else decShowError("");
}

// What the server says about the endpoint: the flags the page shows at the top.
function decShowFlags() {
  const el = $("dec-flags");
  if (!el) return;
  if (!decFlags) { el.textContent = ""; return; }
  const parts = [];
  parts.push(decFlags.diagnostics
    ? "This server shows the model's own probability for every option: it runs with \"diagnostics\": true."
    : "This server does not show the model's own probability for every option: it runs without \"diagnostics\": true in its run config.");
  if (decFlags.multi === "shared")
    parts.push("It answers a request with several questions in one prompt, so the answers carry no probability.");
  el.textContent = parts.join(" ");
}

async function decLoadFlags() {
  try {
    const r = await fetch("v1/status", {headers: headers()});
    if (!r.ok) return;
    const j = await r.json();
    decFlags = j.systemone || null;
    decShowFlags();
  } catch (e) { /* the page works without the flags */ }
}

// The example request, filled into the form.
function decExample() {
  let example;
  try { example = JSON.parse(DEC_EXAMPLE); } catch (e) { return; }
  $("dec-state").value = example.state || "";
  $("dec-state-json").setAttribute("aria-checked", "false");
  $("dec-questions").replaceChildren();
  decQuestions.length = 0;
  for (const [name, q] of Object.entries(example.questions || {})) {
    const block = decNewQuestion();
    block.name.value = name;
    block.type.value = q.type;
    block.instructions.value = q.instructions || "";
    const criteria = q.criteria || {};
    if (q.type === "noul") {
      decCriteria(block);
      block.criteria.querySelector(".dec-yes").value = criteria.true || "";
      block.criteria.querySelector(".dec-no").value = criteria.false || "";
    } else {
      decCriteria(block, Object.entries(criteria));
    }
  }
  decRefresh();
}

// ------------------------------------------------------------------ the answer

async function decSend() {
  decTried = true;
  const built = decBuild();
  if (built.error) { decShowError(built.error); return; }
  decShowError("");
  const body = decBodyText(built.state, built.questions);
  const button = $("dec-send");
  button.disabled = true;
  const t0 = performance.now();
  decTimer = setInterval(() => {
    $("dec-hint").textContent = `Deciding… ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  }, 100);
  try {
    const r = await fetch("v1/systemone", {method: "POST", headers: headers(true), body});
    const raw = await r.text();
    let answer = null;
    try { answer = JSON.parse(raw); } catch (e) { answer = null; }
    const seconds = (performance.now() - t0) / 1000;
    if (!r.ok) {
      // the server's own message: a 400 (the request), a 502 (the model wrote
      // something that did not parse), a 404 (the endpoint is off).  Nothing is invented.
      const message = (answer && answer.error && answer.error.message) || raw || r.statusText;
      decShowError(`The server said (${r.status}): ${message}`);
      $("dec-hint").textContent = "";
      return;
    }
    if (!answer) {
      decShowError(`The server's answer was not JSON: ${raw.slice(0, 200) || "(empty)"}`);
      $("dec-hint").textContent = "";
      return;
    }
    decSent = built.questions.map(([name, def]) => ({name, type: def.type}));
    decRender(answer, seconds);
    $("dec-hint").textContent = `Done in ${seconds.toFixed(1)} s`;
    decLoadFlags();                       // the flags may have changed since the page opened
  } catch (e) {
    decShowError(String(e.message || e));
    $("dec-hint").textContent = "";
  } finally {
    clearInterval(decTimer);
    decTimer = null;
    button.disabled = false;
  }
}

function decRender(r, seconds) {
  const out = $("dec-result");
  out.replaceChildren();
  out.append(decRequestCard(r, seconds));
  const answers = r.answers || {};
  const diagnostics = r.diagnostics || {};
  for (const q of decSent) {
    const answer = answers[q.name];
    out.append(answer ? decAnswerCard(q, answer, diagnostics[q.name])
                      : decMissingCard(q, diagnostics[q.name]));
  }
  const raw = document.createElement("details");
  raw.className = "st-collapse";
  raw.innerHTML = `<summary>The raw answer</summary><div class="st-collapse__body"><pre class="dec-json"></pre></div>`;
  raw.querySelector("pre").textContent = JSON.stringify(r, null, 2);
  out.append(raw);
}

// The request as a whole: how long it took, what it cost, and whether the answers came
// from the model's own distribution.
function decRequestCard(r, seconds) {
  const usage = r.usage || {}, timings = r.timings || {};
  const cached = (usage.prompt_tokens_details || {}).cached_tokens;
  const rows = [["Wall-clock", `${seconds.toFixed(1)} s`]];
  rows.push(["Tokens", `${fmt(usage.prompt_tokens)} read, ${fmt(usage.completion_tokens)} written` +
                       (cached ? ` (${fmt(cached)} of the read ones were in the cache)` : "")]);
  if (timings.prompt_ms != null)
    rows.push(["Reading the prompt", `${(timings.prompt_ms / 1000).toFixed(2)} s` +
                                     (timings.prompt_per_second ? ` (${fmt(timings.prompt_per_second, 1)} tokens/s)` : "")]);
  if (timings.predicted_ms)
    rows.push(["Writing the answer", `${(timings.predicted_ms / 1000).toFixed(2)} s` +
                                     (timings.predicted_per_second ? ` (${fmt(timings.predicted_per_second, 1)} tokens/s)` : "")]);
  rows.push(["Where the numbers come from", r.calibrated
    ? "the model's own distribution - every answer is the weight the model gave it"
    : "a guess - the model wrote the answer as text, so the numbers are 1.0 on the answer and 0.0 on the rest"]);
  const card = decCard("The answer", decFacts(rows));
  card.classList.add("dec-request");
  return card;
}

// One question's answer: the pick, the distribution, the confidence, the numbers behind
// them, and what they mean.
function decAnswerCard(q, answer, diag) {
  const type = answer.type || q.type;
  const bars = [];
  let pick = "", note = "";
  if (type === "noul") {
    // the value is the probability of "yes"; the answer is the side the model picked
    const yes = Number(answer.noul);
    pick = yes >= 0.5 ? "Yes" : "No";
    note = `the probability of yes: ${decPct(yes)}`;
    bars.push({name: "yes", value: yes, pick: pick === "Yes"});
    bars.push({name: "no", value: 1 - yes, pick: pick === "No"});
  } else if (type === "choice") {
    pick = answer.choice || "";
    for (const [name, p] of Object.entries(answer.probabilities || {}))
      bars.push({name, value: Number(p), pick: name === pick});
  } else {
    const legend = answer.legend || {}, probs = answer.probabilities || {};
    const index = Number(answer.score);
    pick = legend[String(index)] != null ? legend[String(index)] : String(index);
    // the page counts the levels from one, the answer counts them from zero: say both
    note = `level ${index + 1} of ${Object.keys(legend).length} (index ${index})`;
    for (const [i, name] of Object.entries(legend))
      bars.push({name, value: Number(probs[i] || 0), pick: Number(i) === index});
  }
  const card = document.createElement("div");
  card.className = "st-card dec-answer";
  const head = document.createElement("div");
  head.className = "dec-answer__head";
  const title = document.createElement("span");
  title.className = "card-title";
  title.textContent = q.name;
  const badge = document.createElement("span");
  badge.className = "st-badge";
  badge.textContent = type;
  head.append(title, badge);
  const pickLine = document.createElement("div");
  pickLine.className = "dec-pick";
  pickLine.textContent = pick;
  if (note) {
    const small = document.createElement("span");
    small.className = "dec-pick__note";
    small.textContent = " " + note;
    pickLine.append(small);
  }
  card.append(head, pickLine, decBars(bars));
  if (answer.confidence != null)
    card.append(decLine(type === "noul"
      ? `The model put ${decPct(answer.confidence)} of its own weight on the side it picked.`
      : `The model put ${decPct(answer.confidence)} of its own weight on this answer.`));
  if (diag) {
    card.append(decFacts(decDiagnosticRows(diag)));
    if (diag.open) card.append(decOpenList(diag.open));
  } else {
    card.append(decLine("The model's own probability for every option is not in this answer: the server needs " +
                        "\"diagnostics\": true in its run config."));
  }
  card.append(decDetails("What these numbers mean", decMeaning(type)));
  return card;
}

// A question the server did not answer: it is left out of the answer, and the reason is
// in the log (and in the diagnostics, when they are on).
function decMissingCard(q, diag) {
  const card = document.createElement("div");
  card.className = "st-card dec-answer";
  const head = document.createElement("div");
  head.className = "dec-answer__head";
  const title = document.createElement("span");
  title.className = "card-title";
  title.textContent = q.name;
  const badge = document.createElement("span");
  badge.className = "st-badge st-badge--error";
  badge.textContent = "no answer";
  head.append(title, badge);
  card.append(head, decLine("The server did not answer this question. The other questions kept theirs."));
  if (diag && diag.reason) card.append(decLine(`The server said: ${diag.reason}`));
  return card;
}

// The numbers behind the answer, each with what it means.
function decDiagnosticRows(diag) {
  const rows = [];
  if (diag.open_confidence != null)
    rows.push(["The model's own probability of the pick", `${decPct(diag.open_confidence)} - how much of the model's own weight went on this answer`]);
  if (diag.closed_confidence != null)
    rows.push(["Its share of the option mass", `${decPct(diag.closed_confidence)} - how much of the weight on the options went on this answer`]);
  if (diag.stop != null)
    rows.push(["The weight on ending the turn", `${decPct(diag.stop)} - how much the model wanted to stop instead of answering`]);
  if (diag.outside != null)
    rows.push(["The weight on everything else", `${decPct(diag.outside)} - how much went on words that are not one of the options`]);
  if (diag.model_picked !== undefined)
    rows.push(["Was the model's next word one of the options?", `${diag.model_picked ? "yes" : "no"}` +
               (diag.model_picked ? "" : " - the model was going to write something else, and this is the option it would have picked")]);
  return rows;
}

// The model's own probability of every option, next to the closed one.
function decOpenList(open) {
  const list = document.createElement("dl");
  list.className = "facts dec-open";
  const dt = document.createElement("dt");
  dt.textContent = "The model's own probability of every option";
  const dd = document.createElement("dd");
  dd.textContent = Object.entries(open).map(([name, p]) => `${name} ${decPct(p)}`).join("  ·  ");
  list.append(dt, dd);
  return list;
}

// What the numbers mean, in plain words.
function decMeaning(type) {
  const lines = type === "noul" ? [
    "The bars are the model's share for yes and no. They add up to 100%: the weight the model put on anything else is left out.",
    "The value is the probability of yes, and it always agrees with the answer: a small value with a high confidence means \"no, and I am fairly sure of it\".",
    "The confidence is the model's own probability of the side it picked, out of every word it could have written.",
    "The weight on ending the turn is how much the model wanted to stop instead of answering.",
    "The weight on everything else is how much went on words that are not yes or no.",
    "Was the model's next word yes or no? When it was no, the model was going to write something else, and the answer is the side it would have picked.",
  ] : [
    "The bars are the model's share for each option. They add up to 100%: the weight the model put on anything else is left out.",
    "The confidence is the model's own probability of the answer, out of every word it could have written. It is never larger than the biggest bar.",
    "Its share of the option mass is the answer's part of the bars.",
    "The model's own probability of every option is each option's chance out of the model's whole vocabulary, with nothing left out.",
    "The weight on ending the turn is how much the model wanted to stop instead of answering.",
    "The weight on everything else is how much went on words that are not one of the options.",
    "Was the model's next word one of the options? When it was no, the model was going to write something else, and the answer is the option it would have picked.",
  ];
  const list = document.createElement("ul");
  for (const line of lines) {
    const item = document.createElement("li");
    item.textContent = line;
    list.append(item);
  }
  return list;
}

// ------------------------------------------------------------------ the small helpers

function decCard(title, ...body) {
  const card = document.createElement("div");
  card.className = "st-card";
  const head = document.createElement("span");
  head.className = "card-title";
  head.textContent = title;
  card.append(head, ...body);
  return card;
}

function decFacts(rows) {
  const list = document.createElement("dl");
  list.className = "facts";
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
  return list;
}

function decLine(text) {
  const p = document.createElement("p");
  p.className = "dec-line";
  p.textContent = text;
  return p;
}

function decBars(bars) {
  const box = document.createElement("div");
  box.className = "dec-bars";
  for (const b of bars) {
    const row = document.createElement("div");
    row.className = "dec-bar" + (b.pick ? " dec-bar--pick" : "");
    const name = document.createElement("span");
    name.className = "dec-bar__name";
    name.textContent = b.name;
    const track = document.createElement("div");
    track.className = "st-progress";
    const fill = document.createElement("div");
    fill.className = "st-progress__bar";
    fill.style.width = `${Math.max(0, Math.min(100, Number(b.value) * 100)).toFixed(2)}%`;
    track.append(fill);
    const pct = document.createElement("span");
    pct.className = "dec-bar__pct";
    pct.textContent = decPct(b.value);
    row.append(name, track, pct);
    box.append(row);
  }
  return box;
}

function decDetails(summary, body) {
  const details = document.createElement("details");
  details.className = "st-collapse";
  const head = document.createElement("summary");
  head.textContent = summary;
  const inner = document.createElement("div");
  inner.className = "st-collapse__body";
  inner.append(body);
  details.append(head, inner);
  return details;
}

// the endpoint is off by default: the tab is hidden, and this says so plainly.
// app.js loads the health when the page starts, so the flag is read when it is there.
function decEndpointNote(tries = 20) {
  if (health.decisions === undefined && tries > 0) {
    setTimeout(() => decEndpointNote(tries - 1), 250);
    return;
  }
  $("dec-off").hidden = health.decisions !== false;
}

// ------------------------------------------------------------------ start
decNewQuestion();
decRefresh();
$("dec-add").onclick = () => { decNewQuestion(); decRefresh(); };
$("dec-example").onclick = decExample;
$("dec-send").onclick = decSend;
$("dec-copy").onclick = () => copyText($("dec-preview").textContent, $("dec-copy"));
$("dec-state").oninput = decRefresh;
$("dec-state-json").onclick = () => {
  const button = $("dec-state-json");
  button.setAttribute("aria-checked", String(button.getAttribute("aria-checked") !== "true"));
  decRefresh();
};
decEndpointNote();
decLoadFlags();
