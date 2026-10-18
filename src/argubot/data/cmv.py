"""CMV Winning Arguments data (owner: Tricia).

Comment cleaning, the thread-level split (a leak between train and test would quietly
invalidate every result) and the persona chain extraction.
"""
from __future__ import annotations

import html
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from argubot.text import split_sentences, truncate_words

_URL = re.compile(r"https?://\S+|www\.\S+")
_LINK_LABEL = re.compile(r"\[([^\]]+)\]\((?:\s*\))?")
_MARKUP = re.compile(r"[*^]|~~")  # bold/italic asterisks, superscript carets, strikethrough
_REMOVED = {"", "[deleted]", "[removed]"}
_AUTOMATED_USERS = {"deltabot", "automoderator"}


def is_removed(text: str) -> bool:
    return text.strip().lower() in _REMOVED


def is_automated(speaker: str) -> bool:
    return speaker.strip().lower() in _AUTOMATED_USERS


def clean_comment(text: str) -> str:
    """Drop quoted lines and URLs, collapse whitespace.

    Reddit marks a quote with a leading ">" (sometimes escaped as "&gt;"), so those
    lines go first. Otherwise the bot would be trained to repeat the opponent's own words.
    """
    kept = [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith((">", "&gt;"))
    ]
    text = _URL.sub("", " ".join(kept))
    return " ".join(text.split())


def split_by_thread(
    thread_ids: Sequence[str],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, str]:
    """Map each thread id to "train", "val" or "test". A thread never crosses a split.

    Always split threads, not chains. Two chains from one thread share the original post,
    so splitting chains would leak the topic across splits.
    """
    unique = sorted(set(thread_ids))
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    mapping: Dict[str, str] = {}
    for i, tid in enumerate(unique):
        if i < n_train:
            mapping[tid] = "train"
        elif i < n_train + n_val:
            mapping[tid] = "val"
        else:
            mapping[tid] = "test"
    return mapping


MIN_BOT_WORDS = 10  # a bot turn shorter than this ("lol", "this.") teaches nothing


def prepare_text(text: str) -> str:
    """clean_comment plus HTML entities and Reddit markdown.

    The corpus stores "&gt;" for quotes and "&amp;" for "&". clean_comment strips the URL inside
    "[label](url)" and leaves "[label](", so the label is kept and the brackets removed here.
    """
    text = clean_comment(text)
    for _ in range(3):  # some comments are encoded twice ("&amp;#8710;" for the delta sign)
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = _LINK_LABEL.sub(r"\1", text)
    return " ".join(_MARKUP.sub("", text).split())


def cap_words(text: str, max_words: int, *, whole_sentences: bool) -> Optional[str]:
    """Shorten text to at most max_words words (whitespace tokens stand in for model tokens).

    With whole_sentences the cut falls between sentences, so a bot turn never ends mid-thought.
    If even the first sentence is too long it returns None and the caller drops the chain.
    Without it the text is cut hard, which is what we want for the poster's long opening post.
    """
    if len(text.split()) <= max_words:
        return text
    if not whole_sentences:
        return truncate_words(text, max_words)
    kept, used = [], 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if used + n > max_words:
            break
        kept.append(sent)
        used += n
    return " ".join(kept) if kept else None


def _sort_key(post: Dict[str, Any]):
    return (post["ts"] if post["ts"] is not None else 0, post["id"])


def chains_for_thread(
    posts: Sequence[Dict[str, Any]],
    op: str,
    *,
    min_turns: int = 3,
    max_turns: int = 6,
    bot_max_words: int = 150,
    user_max_words: int = 200,
) -> List[List[Dict[str, str]]]:
    """Persona dialogues for one thread. Plain dicts in, so it is testable without ConvoKit.

    posts: dicts with id, parent (None for the opening post), author, text, ts (may be None),
    distinguished (e.g. "moderator", may be None). The first post must be the poster's own.

    Reddit threads are trees, so a dialogue is a path that starts at the poster's opening post
    and alternates poster -> persona -> poster -> persona. Each persona gets at most one dialogue
    per thread, the longest alternating path (ties go to the earliest comments).
    """
    by_id = {p["id"]: p for p in posts}
    roots = [p for p in posts if p["parent"] is None]
    if len(roots) != 1 or roots[0]["author"] != op or is_removed(op):
        return []
    root = roots[0]

    children: Dict[str, List[Dict[str, Any]]] = {}
    for p in posts:
        if p["parent"] in by_id:
            children.setdefault(p["parent"], []).append(p)
    for kids in children.values():
        kids.sort(key=_sort_key)

    def usable(p: Dict[str, Any]) -> bool:
        return not (
            is_removed(p["text"])
            or is_removed(p["author"])
            or is_automated(p["author"])
            or (p.get("distinguished") or "").lower() == "moderator"
        )

    prepared: Dict[str, Optional[str]] = {}

    def text_of(p: Dict[str, Any], is_bot: bool) -> Optional[str]:
        if p["id"] not in prepared:
            clean = prepare_text(p["text"])
            capped = cap_words(clean, bot_max_words if is_bot else user_max_words, whole_sentences=is_bot) if clean else None
            if capped is not None and is_bot and len(capped.split()) < MIN_BOT_WORDS:
                capped = None
            prepared[p["id"]] = capped
        return prepared[p["id"]]

    root_text = text_of(root, is_bot=False) if usable(root) else None
    if root_text is None:
        return []

    persona_candidates = {
        c["author"]
        for c in children.get(root["id"], [])
        if c["author"] != op and usable(c)
    }
    # replies to the poster's later comments count too, so collect those authors as well
    for p in posts:
        if p["author"] == op and p["id"] != root["id"]:
            persona_candidates.update(
                c["author"] for c in children.get(p["id"], []) if c["author"] != op and usable(c)
            )

    def longest(node: Dict[str, Any], persona: str, depth: int) -> List[Dict[str, Any]]:
        want_bot = node["author"] == op
        want = persona if want_bot else op
        tail: List[Dict[str, Any]] = []
        if depth < max_turns:
            for child in children.get(node["id"], []):
                if child["author"] != want or not usable(child):
                    continue
                if text_of(child, is_bot=want_bot) is None:
                    continue
                candidate = longest(child, persona, depth + 1)
                if len(candidate) > len(tail):
                    tail = candidate
        return [node] + tail

    dialogues: List[List[Dict[str, str]]] = []
    for persona in sorted(persona_candidates):
        path = longest(root, persona, 1)
        bot_turns = sum(1 for p in path if p["author"] == persona)
        if len(path) < min_turns or bot_turns < 2:
            continue
        dialogues.append(
            [
                {
                    "speaker": "bot" if p["author"] == persona else "user",
                    "text": root_text if p is root else text_of(p, is_bot=p["author"] == persona),
                }
                for p in path
            ]
        )
    return dialogues


def extract_persona_chains(corpus, cfg) -> List[Dict[str, Any]]:
    """Persona dialogues from the ConvoKit Winning Arguments corpus, split by thread.

    A persona is a non-poster who replies to the poster at least twice in one thread. Their
    replies are the bot's turns and the poster's comments are the user turns. Quotes, URLs,
    deleted/removed comments, moderators and bots are dropped. Dialogues have 3 to 6 turns and
    bot turns are cut to cfg["data"]["bot_reply_max_tokens"] words at a sentence boundary.

    Every thread gets its split first (split_by_thread over ALL threads, so the assignment does
    not change if the filters above change), then each dialogue inherits its thread's split.
    ConvoKit field names, checked against the real corpus: conversation id is the opening post
    id, conversation.meta["op-userID"] is the poster, utterance.reply_to is the parent id (None
    for the opening post), utterance.speaker.id is the author, utterance.meta["distinguished"]
    is "moderator" for mods. Timestamps are None for the opening post and some others, so
    conversation order comes from reply_to, never from sorting by timestamp.
    """
    d = cfg["data"]
    threads = list(corpus.iter_conversations())
    split_of = split_by_thread([t.id for t in threads], tuple(d["split_ratios"]), cfg.get("seed", 42))

    rows: List[Dict[str, Any]] = []
    for thread in sorted(threads, key=lambda t: t.id):
        posts = [
            {
                "id": u.id,
                "parent": u.reply_to,
                "author": u.speaker.id,
                "text": u.text,
                "ts": u.timestamp,
                "distinguished": u.meta.get("distinguished"),
            }
            for u in thread.iter_utterances()
        ]
        dialogues = chains_for_thread(
            posts,
            thread.meta["op-userID"],
            min_turns=d["min_turns"],
            max_turns=d["max_turns"],
            bot_max_words=d["bot_reply_max_tokens"],
            user_max_words=d.get("user_turn_max_words", 200),
        )
        for i, turns in enumerate(dialogues):
            rows.append(
                {
                    "dialogue_id": f"{thread.id}_{i}",
                    "split": split_of[thread.id],
                    "turns": turns,
                }
            )
    return rows
