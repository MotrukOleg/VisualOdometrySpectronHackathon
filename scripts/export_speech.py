"""Export the speaker text of the reveal.js slides to a Markdown script (docs/presentation/speech.md).

The text lives in the slides themselves (<aside class="notes" data-speaker=".." data-time="..">), with numbers as
<span data-k="path"> placeholders; this script fills them from docs/presentation/data.js (scripts/make_slides_data.py),
so the slides, the speaker view (key S) and the printed script always say the same numbers.

    python scripts/make_slides_data.py && python scripts/export_speech.py
"""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRES = ROOT / "docs" / "presentation"


class Slides(HTMLParser):
    """Collects, per leaf <section>: title (h1/h2), speaker, time and the notes text (data-k filled)."""

    def __init__(self, data):
        super().__init__(convert_charrefs=True)
        self.data, self.slides, self.stack = data, [], []
        self.cur = None          # current leaf slide dict
        self.mode = None         # "title" | "notes" | None
        self.skip_k = 0          # depth inside a data-k span (its own text is replaced by the value)

    def value(self, path):
        v = self.data
        for k in path.split("."):
            v = v[int(k)] if isinstance(v, list) else v.get(k) if isinstance(v, dict) else None
            if v is None:
                return "–"
        return str(v)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "section":
            self.stack.append({"title": "", "speaker": "", "time": "", "notes": "", "leaf": True})
            if len(self.stack) > 1:
                self.stack[-2]["leaf"] = False
        elif self.stack and tag in ("h1", "h2") and not self.stack[-1]["title"]:
            self.mode = "title"
        elif self.stack and tag == "aside" and "notes" in a.get("class", ""):
            self.mode = "notes"
            self.stack[-1]["speaker"], self.stack[-1]["time"] = a.get("data-speaker", ""), a.get("data-time", "")
        if tag == "span" and "data-k" in a and self.mode:
            self._add(self.value(a["data-k"]))
            self.skip_k += 1
        elif tag == "br" and self.mode == "title":
            self._add(" ")

    def handle_endtag(self, tag):
        if tag == "span" and self.skip_k:
            self.skip_k -= 1
        elif tag in ("h1", "h2") and self.mode == "title":
            self.mode = None
        elif tag == "aside" and self.mode == "notes":
            self.mode = None
        elif tag == "section" and self.stack:
            s = self.stack.pop()
            if s["leaf"]:
                self.slides.append(s)

    def handle_data(self, text):
        if self.mode and not self.skip_k:
            self._add(text)

    def _add(self, text):
        key = "title" if self.mode == "title" else "notes"
        self.stack[-1][key] += text


def main():
    js = (PRES / "data.js").read_text(encoding="utf-8")
    data = json.loads(js[js.index("{"): js.rstrip().rindex("}") + 1])
    p = Slides(data)
    p.feed((PRES / "index.html").read_text(encoding="utf-8"))
    clean = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    slides = [{k: clean(v) if isinstance(v, str) else v for k, v in s.items()} for s in p.slides]

    def secs(t):
        m = re.fullmatch(r"(\d+):(\d\d)", t)
        return int(m[1]) * 60 + int(m[2]) if m else 0

    main_slides = [s for s in slides if secs(s["time"])]
    total = sum(secs(s["time"]) for s in main_slides)
    by_speaker = {}
    for s in main_slides:
        by_speaker[s["speaker"]] = by_speaker.get(s["speaker"], 0) + secs(s["time"])
    md = ["# Текст виступу\n",
          "Згенеровано `scripts/export_speech.py` з нотаток слайдів (`docs/presentation/index.html`, клавіша **S** — режим "
          "доповідача); числа — з `data.js`. Редагувати текст треба в нотатках слайдів, потім перезапустити скрипт.\n",
          f"**Тривалість основної частини:** ≈ {total // 60}:{total % 60:02d}. "
          "Слайди з позначкою «запасний» — вертикальні, показувати лише за питаннями.\n",
          "| Доповідач | Роль | Час |\n|---|---|---|"]
    roles = {"Людина 1": "ядро конвеєра, геометрія", "Людина 2": "оцінювання і бенчмарк",
             "Людина 3": "навчені методи, GPU", "Людина 4": "відео, візуалізація, презентація"}
    for sp in sorted(by_speaker):
        md.append(f"| {sp} | {roles.get(sp, '')} | {by_speaker[sp] // 60}:{by_speaker[sp] % 60:02d} |")
    md.append("")
    n = 0
    for s in slides:
        main_one = bool(secs(s["time"]))
        if main_one:
            n += 1
        head = f"## {n}. {s['title']}" if main_one else f"### Запасний: {s['title']}"
        md += [head, f"*{s['speaker']} · {s['time']}*\n", s["notes"] + "\n"]
    out = PRES / "speech.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}: {len(main_slides)} main + {len(slides) - len(main_slides)} backup slides, "
          f"{total // 60}:{total % 60:02d}; per speaker " + ", ".join(f"{k} {v // 60}:{v % 60:02d}"
                                                                   for k, v in sorted(by_speaker.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
