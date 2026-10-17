import { useEffect, useRef, useState } from "react";

// What each ablation adds on top of the one before it. Unknown system names still work, they just show no chips.
const FEATURES = [
  { key: "retrieval", label: "Retrieval" },
  { key: "claims", label: "Claim log" },
  { key: "stance", label: "Stance filter" },
  { key: "rerank", label: "5-way rerank" },
];
const SYSTEMS = {
  A: { blurb: "History only", on: [] },
  B: { blurb: "Plus retrieved evidence", on: ["retrieval"] },
  C1: { blurb: "Plus its own claim log", on: ["retrieval", "claims"] },
  C2: { blurb: "Plus stance-filtered evidence", on: ["retrieval", "claims", "stance"] },
  D: { blurb: "Plus 5-candidate NLI reranking", on: ["retrieval", "claims", "stance", "rerank"] },
};

const STARTERS = [
  "School uniforms should be mandatory.",
  "Social media does more harm than good for teenagers.",
  "Nuclear power is the best way to cut emissions.",
];

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

export default function App() {
  const [info, setInfo] = useState(null);
  const [system, setSystem] = useState(null);
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]); // {role, text, snap?} where snap is the bot turn's inspector data
  const [selected, setSelected] = useState(null); // index of the bot message being inspected, null = latest
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [tab, setTab] = useState("claims");
  const chatRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch("/api/info")
      .then((r) => r.json())
      .then((i) => {
        setInfo(i);
        setSystem(i.systems.includes("C1") ? "C1" : i.systems[0]);
      })
      .catch(() => setInfo({ systems: [], dummy: false, offline: true }));
  }, []);

  useEffect(() => {
    if (!system) return;
    setMessages([]);
    setSelected(null);
    setSession(null);
    post("/api/new", { system }).then((d) => setSession(d.session));
  }, [system]);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const restart = () => {
    setMessages([]);
    setSelected(null);
    post("/api/new", { system }).then((d) => setSession(d.session));
    inputRef.current?.focus();
  };

  const send = async (text) => {
    text = text.trim();
    if (!text || busy || !session) return;
    setDraft("");
    setSelected(null);
    setMessages((m) => [...m, { role: "user", text }]);
    setBusy(true);
    try {
      const out = await post("/api/chat", { session, message: text });
      setMessages((m) => {
        const prev = [...m].reverse().find((x) => x.snap)?.snap.tracked ?? [];
        const snap = { shown: out.claims_shown_to_bot, tracked: out.claim_log_now, passages: out.passages, fresh: out.claim_log_now.filter((c) => !prev.includes(c)) };
        return [...m, { role: "bot", text: out.reply, snap }];
      });
    } catch (err) {
      setMessages((m) => [...m, { role: "error", text: err.message }]);
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  };

  const botTurns = messages.map((m, i) => (m.snap ? i : -1)).filter((i) => i >= 0);
  const inspectIdx = selected ?? botTurns[botTurns.length - 1];
  const snap = inspectIdx != null ? messages[inspectIdx].snap : null;
  const turnNo = inspectIdx != null ? botTurns.indexOf(inspectIdx) + 1 : 0;
  const spec = SYSTEMS[system] ?? { blurb: "", on: [] };

  if (info?.offline) {
    return <div className="offline">Can't reach the ArguBot server. Start it with <code>python scripts/serve_demo.py --dummy</code>.</div>;
  }

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>
            <svg viewBox="0 0 24 24" width="18" height="18"><path d="M4 5h11a3 3 0 0 1 3 3v5a3 3 0 0 1-3 3H9l-4 3v-3H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Zm16 4h0a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-1v2l-3-2h-3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /></svg>
          </span>
          <div>
            <h1>ArguBot</h1>
            <p>Retrieval-augmented, self-consistent debate</p>
          </div>
        </div>
        {info?.dummy && <span className="pill warn">Dummy mode: canned replies</span>}
      </header>

      <section className="systems">
        <div className="seg" role="tablist" aria-label="System">
          {info?.systems.map((s) => (
            <button key={s} role="tab" aria-selected={s === system} className={s === system ? "on" : ""} onClick={() => setSystem(s)}>
              {s}
            </button>
          ))}
        </div>
        <div className="feat">
          <span className="blurb">{spec.blurb}</span>
          {FEATURES.map((f) => (
            <span key={f.key} className={"chip" + (spec.on.includes(f.key) ? " lit" : "")}>
              <i />
              {f.label}
            </span>
          ))}
        </div>
      </section>

      <main className="grid">
        <section className="panel chat-panel">
          <div className="chat" ref={chatRef}>
            {messages.length === 0 && (
              <div className="empty-chat">
                <h2>Pick a side and start arguing</h2>
                <p>Make a claim and ArguBot will push back. Click any reply to see what it saw on that turn.</p>
                <div className="starters">
                  {STARTERS.map((s) => (
                    <button key={s} onClick={() => send(s)} disabled={!session}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                className={`msg ${m.role}` + (m.snap && i === inspectIdx ? " inspected" : "")}
                onClick={m.snap ? () => setSelected(i) : undefined}
                title={m.snap ? "Inspect this turn" : undefined}
              >
                {m.role === "bot" && <span className="who">ArguBot · turn {botTurns.indexOf(i) + 1}</span>}
                {m.role === "error" && <span className="who">Error</span>}
                {m.text}
              </div>
            ))}
            {busy && (
              <div className="msg bot typing" aria-label="ArguBot is thinking">
                <span /><span /><span />
              </div>
            )}
          </div>
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              send(draft);
            }}
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={draft}
              placeholder="Make your argument…"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(draft);
                }
              }}
            />
            <button type="button" className="ghost" onClick={restart} title="New debate">
              Reset
            </button>
            <button className="primary" disabled={busy || !draft.trim()}>
              Send
            </button>
          </form>
        </section>

        <aside className="panel inspector">
          <div className="insp-head">
            <h2>Inspector</h2>
            <span className="muted">{snap ? `Turn ${turnNo}${selected == null ? " (latest)" : ""}` : "No turns yet"}</span>
            {selected != null && (
              <button className="link" onClick={() => setSelected(null)}>
                Back to latest
              </button>
            )}
          </div>
          <div className="tabs">
            <button className={tab === "claims" ? "on" : ""} onClick={() => setTab("claims")}>
              Claim log <b>{snap?.tracked.length ?? 0}</b>
            </button>
            <button className={tab === "evidence" ? "on" : ""} onClick={() => setTab("evidence")}>
              Evidence <b>{snap?.passages.length ?? 0}</b>
            </button>
          </div>
          {tab === "claims" ? <Claims snap={snap} usesLog={spec.on.includes("claims")} /> : <Evidence snap={snap} />}
        </aside>
      </main>
    </div>
  );
}

function Claims({ snap, usesLog }) {
  if (!snap) return <p className="placeholder">Claims the bot commits to will collect here.</p>;
  if (!snap.tracked.length) return <p className="placeholder">No claims tracked yet.</p>;
  return (
    <>
      <p className="legend">
        <span className="tag in">in prompt</span> the bot was reminded of this claim
        {!usesLog && <><br />This system doesn't read its claim log, so it can contradict itself.</>}
      </p>
      <ol className="claims">
        {snap.tracked.map((c, i) => (
          <li key={c + i} className={snap.fresh.includes(c) ? "fresh" : ""}>
            <span className="num">{i + 1}</span>
            <span className="txt">{c}</span>
            <span className="tags">
              {snap.shown.includes(c) && <span className="tag in">in prompt</span>}
              {snap.fresh.includes(c) && <span className="tag new">new</span>}
            </span>
          </li>
        ))}
      </ol>
    </>
  );
}

function Evidence({ snap }) {
  if (!snap) return <p className="placeholder">Retrieved passages for each reply show up here.</p>;
  if (!snap.passages.length) return <p className="placeholder">This system doesn't retrieve evidence.</p>;
  return (
    <ol className="passages">
      {snap.passages.map((p, i) => (
        <li key={p + i} style={{ animationDelay: `${i * 60}ms` }}>
          <span className="rank">#{i + 1}</span>
          {p}
        </li>
      ))}
    </ol>
  );
}
