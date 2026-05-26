"""Build docs/study/guide.html — a SINGLE self-contained interactive HTML."""

from __future__ import annotations

import json
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

STUDY_DIR = Path(__file__).resolve().parent.parent / "docs" / "study"

CHAPTERS: list[tuple[str, str, str, str, str, int]] = [
    ("00-getting-started.md",          "Phase 1. 이해하기", "00", "시작하기 — 환경 셋업",        "환경 셋업 + ASR 한 줄 정리",            15),
    ("01-asr-fundamentals.md",         "Phase 1. 이해하기", "01", "ASR 기본 — 음성→텍스트",       "ASR 동작 원리 (mel, encoder, decoder)", 25),
    ("02-korean-asr.md",               "Phase 1. 이해하기", "02", "한국어 ASR 의 특수성",          "띄어쓰기/외래어/사투리 등",           20),
    ("03-models-whisper-sensevoice.md","Phase 1. 이해하기", "03", "Whisper vs. SenseVoice",        "두 모델의 같은 점·다른 점",             20),
    ("04-data-pipeline.md",            "Phase 1. 이해하기", "04", "데이터 파이프라인",             "RAW → SILVER → GOLD",                 25),
    ("05-training-deepdive.md",        "Phase 2. 학습/평가 깊이", "05", "학습 내부",              "loss / optimizer / scheduler / batch",  30),
    ("06-evaluation-metrics.md",       "Phase 2. 학습/평가 깊이", "06", "평가 메트릭",            "CER · WER · 한국어 정규화",           25),
    ("07-finetuning-strategies.md",    "Phase 2. 학습/평가 깊이", "07", "파인튜닝 전략",          "Continual / LoRA / PEFT",             20),
    ("08-benchmark-building.md",       "Phase 2. 학습/평가 깊이", "08", "벤치마크 구축",          "도메인·화자별 평가셋 설계",            20),
    ("09-error-analysis.md",           "Phase 2. 학습/평가 깊이", "09", "오류 분석",              "숫자 뒤의 이야기",                   20),
    ("glossary.md",                    "참고",                  "—",  "용어집",                    "ASR / 한국어 ASR 용어",               10),
]

CER_WIDGET_PAGES = {"06-evaluation-metrics.md", "02-korean-asr.md"}


def chapter_slug(fname: str) -> str:
    return fname.removesuffix(".md")


def md_to_html_with_ids(raw_md: str, prefix: str) -> tuple[str, list[dict]]:
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "codehilite", "sane_lists", "toc"],
        extension_configs={
            "codehilite": {"css_class": "codehilite", "guess_lang": False},
            "toc": {"slugify": slugify_unicode, "permalink": False},
        },
    )
    body = md.convert(raw_md)
    headings: list[dict] = []

    def replace_heading(m: re.Match) -> str:
        tag = m.group(1)
        attrs = m.group(2)
        inner = m.group(3)
        id_m = re.search(r'id="([^"]+)"', attrs)
        if id_m:
            orig_id = id_m.group(1)
            new_id = f"{prefix}--{orig_id}"
            attrs = attrs.replace(f'id="{orig_id}"', f'id="{new_id}"')
            text = re.sub(r"<[^>]+>", "", inner).strip()
            headings.append({"level": int(tag[1]), "text": text, "id": new_id})
        return f"<{tag}{attrs}>{inner}</{tag}>"

    body = re.sub(r"<(h[1-6])([^>]*)>(.+?)</\1>", replace_heading, body, flags=re.S)
    body = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{prefix}--{m.group(1)}"', body)

    def replace_link(m: re.Match) -> str:
        href = m.group(2)
        if href.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        if href.startswith("../"):
            return m.group(0)
        if ".md#" in href and "/" not in href:
            base, anchor = href.split(".md#", 1)
            return f'{m.group(1)}="#{base}--{anchor}"'
        if href.endswith(".md") and "/" not in href:
            return f'{m.group(1)}="#{href[:-3]}"'
        return m.group(0)

    body = re.sub(r'(href)="([^"]+)"', replace_link, body)
    return body, headings


CER_WIDGET = """
<div class="widget" id="cer-widget">
  <div class="widget-header">
    <span class="widget-tag">인터랙티브</span>
    <span class="widget-title">한국어 CER · WER 계산기</span>
  </div>
  <div class="widget-body">
    <p class="widget-desc">아래 두 입력에 <strong>정답(reference)</strong>과 <strong>예측(hypothesis)</strong> 을 입력하면 CER / sCER / WER 이 즉시 계산됩니다. 띄어쓰기만 바꿔서 WER 과 CER 의 변동 폭을 비교해보세요.</p>
    <div class="cer-grid">
      <label>정답 (reference)
        <textarea class="cer-ref" rows="2">아메리카노 한 잔이요</textarea>
      </label>
      <label>예측 (hypothesis)
        <textarea class="cer-hyp" rows="2">아메리카노 한잔이요</textarea>
      </label>
    </div>
    <div class="cer-options">
      <label><input type="checkbox" class="cer-normalize" checked> 한국어 정규화 적용 (구두점/대소문자/공백)</label>
    </div>
    <div class="cer-results">
      <div class="cer-card"><div class="cer-card-label">CER</div><div class="cer-card-value cer-out">–</div><div class="cer-card-sub">한국어 표준</div></div>
      <div class="cer-card"><div class="cer-card-label">sCER</div><div class="cer-card-value scer-out">–</div><div class="cer-card-sub">공백 무시</div></div>
      <div class="cer-card"><div class="cer-card-label">WER</div><div class="cer-card-value wer-out">–</div><div class="cer-card-sub">참고치</div></div>
      <div class="cer-card"><div class="cer-card-label">편집</div><div class="cer-card-value cer-small ops-out">–</div><div class="cer-card-sub">S / I / D</div></div>
    </div>
    <div class="cer-diff" aria-live="polite"></div>
    <div class="cer-presets">
      <span class="cer-presets-label">프리셋 시도:</span>
      <button data-ref="아메리카노 한 잔이요" data-hyp="아메리카노 한잔이요">띄어쓰기만 다름</button>
      <button data-ref="안녕하세요" data-hyp="안녕하시요">받침 오류</button>
      <button data-ref="좋은 하루 되세요" data-hyp="좋은 하루 되세요!">구두점 차이</button>
      <button data-ref="아이스 아메리카노 톨 사이즈" data-hyp="아이스 아메리카노 톨사이로">외래어 패턴</button>
      <button data-ref="3개 주세요" data-hyp="세 개 주세요">숫자 표기</button>
    </div>
  </div>
</div>
"""


def build_chapter_section(fname: str) -> str:
    slug = chapter_slug(fname)
    raw = (STUDY_DIR / fname).read_text(encoding="utf-8")
    body_html, _ = md_to_html_with_ids(raw, slug)
    meta = next(c for c in CHAPTERS if c[0] == fname)
    _f, _g, num, _title, _d, mins = meta

    meta_bar = (
        '<div class="chapter-meta">'
        f'<span class="badge"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> 약 {mins}분</span>'
        f'<span class="badge ghost">Chapter {num}</span>'
        '<label class="read-toggle">'
        f'<input type="checkbox" class="read-toggle-input" data-chapter="{fname}">'
        '<span>읽음 표시</span>'
        '</label>'
        '</div>'
    )

    idx = next(i for i, c in enumerate(CHAPTERS) if c[0] == fname)
    if idx > 0:
        pf, _pg, pnum, ptitle, _pd, _pm = CHAPTERS[idx - 1]
        prev_html = (
            f'<a class="prev" href="#{chapter_slug(pf)}">'
            f'<span class="label">← 이전 ({pnum})</span>'
            f'<span class="title">{ptitle}</span></a>'
        )
    else:
        prev_html = '<a class="prev" href="#home"><span class="label">← 처음</span><span class="title">학습 로드맵</span></a>'
    if idx + 1 < len(CHAPTERS):
        nf, _ng, nnum, ntitle, _nd, _nm = CHAPTERS[idx + 1]
        next_html = (
            f'<a class="next" href="#{chapter_slug(nf)}">'
            f'<span class="label">다음 ({nnum}) →</span>'
            f'<span class="title">{ntitle}</span></a>'
        )
    else:
        next_html = '<a class="next" href="#home"><span class="label">처음으로 →</span><span class="title">학습 로드맵</span></a>'
    pagenav = f'<nav class="page-nav">{prev_html}{next_html}</nav>'

    cer = CER_WIDGET if fname in CER_WIDGET_PAGES else ""

    return (
        f'<section class="page" data-route="{slug}" data-chapter="{fname}" hidden>'
        f'<div class="crumbs"><a href="#home">학습 로드맵</a> / Chapter {num}</div>'
        f'{meta_bar}'
        f'<article class="article">{body_html}{cer}</article>'
        f'{pagenav}'
        f'<div class="footer">© TRAIN-ASR · 한국어 ASR 학습·평가 프로젝트</div>'
        f'</section>'
    )


def build_sidebar() -> str:
    parts: list[str] = [
        '<a class="brand" href="#home"><span>TRAIN-ASR</span><small>study guide for interns</small></a>',
        '<div class="sidebar-controls">',
        '<button class="btn-icon" id="btn-search" title="검색 (Ctrl/Cmd + K)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg><span>검색</span><kbd>⌘K</kbd></button>',
        '<button class="btn-icon" id="btn-theme" title="다크/라이트 토글 (T)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg><span>테마</span></button>',
        '<button class="btn-icon" id="btn-progress-reset" title="진행률 초기화"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg><span>진행률 초기화</span></button>',
        '</div>',
        '<div class="progress-bar"><div class="progress-bar-fill" id="progress-fill"></div></div>',
        '<div class="progress-text"><span id="progress-text">0 / 14 완료</span></div>',
        '<nav>',
        '<h3>시작</h3>',
        '<ol>',
        '<li><a href="#home" data-route="home"><span class="num">★</span>학습 로드맵</a></li>',
        '</ol>',
    ]
    current_group = None
    for fname, group, num, title, _desc, _mins in CHAPTERS:
        if group != current_group:
            if current_group is not None:
                parts.append('</ol>')
            parts.append(f'<h3>{group}</h3>')
            parts.append('<ol>')
            current_group = group
        slug = chapter_slug(fname)
        parts.append(
            f'<li><a href="#{slug}" data-route="{slug}" data-chapter="{fname}">'
            f'<span class="num">{num}</span>{title}'
            f'<span class="check" aria-hidden="true">✓</span></a></li>'
        )
    parts.append('</ol></nav>')
    return "\n".join(parts)


def build_home_section() -> str:
    hero = (
        '<div class="hero">'
        '<h1>한국어 ASR 인턴 학습 로드맵</h1>'
        '<p>이론부터 핸즈온까지. 8~10일 안에 첫 학습 사이클까지 돌릴 수 있게.</p>'
        '<div class="hero-actions">'
        '<a href="#00-getting-started" class="hero-btn primary">처음부터 시작 →</a>'
        '<button class="hero-btn ghost" id="hero-search">키워드로 검색</button>'
        '</div>'
        '</div>'
    )
    intro = (
        '<p>이 가이드는 한국어 음성인식(ASR) 프로젝트에 합류한 인턴을 위한 자료입니다. '
        '이론(Phase 1) → 학습/평가 깊이(Phase 2) → 손으로 굴려보기(Phase 3) 순서로 따라가면 '
        '1~2주 안에 모델을 학습·평가하고 결과를 분석할 수 있게 됩니다.</p>'
        '<blockquote><p>읽기만 하지 말고 손을 움직이세요. 각 챕터의 <strong>"직접 해보기"</strong> 박스가 이론을 손에 박히게 만듭니다.</p></blockquote>'
    )
    cards: list[str] = []
    current_group = None
    for fname, group, num, title, desc, mins in CHAPTERS:
        if group != current_group:
            if current_group is not None:
                cards.append('</div>')
            cards.append(f'<h2>{group}</h2>')
            cards.append('<div class="card-grid">')
            current_group = group
        slug = chapter_slug(fname)
        cards.append(
            f'<a class="card" href="#{slug}" data-chapter="{fname}">'
            f'<div class="ch-row"><span class="ch">Chapter {num}</span><span class="mins">{mins}분</span></div>'
            f'<div class="title">{title}</div>'
            f'<div class="desc">{desc}</div>'
            f'<div class="card-check">읽음 ✓</div>'
            f'</a>'
        )
    cards.append('</div>')
    extra = """
<h2>학습 방식 가이드</h2>
<ol>
  <li><strong>읽기만 하지 말고 손을 움직인다</strong></li>
  <li><strong>모르면 물어본다, 단 5분 자가 해결 후</strong></li>
  <li><strong>노트북에 메모하면서 본다</strong></li>
  <li><strong>모르는 용어는 즉시 찾는다</strong> — <a href="#glossary">용어집</a></li>
  <li><strong>모든 실험은 가설 → 실행 → 결과 → 해석</strong></li>
</ol>
<h2>키보드 단축키</h2>
<table>
  <thead><tr><th>키</th><th>동작</th></tr></thead>
  <tbody>
    <tr><td><kbd>Ctrl/Cmd</kbd> + <kbd>K</kbd></td><td>검색 열기</td></tr>
    <tr><td><kbd>J</kbd> / <kbd>K</kbd></td><td>다음 / 이전 챕터</td></tr>
    <tr><td><kbd>M</kbd></td><td>현재 챕터 읽음 표시 토글</td></tr>
    <tr><td><kbd>T</kbd></td><td>다크 / 라이트 테마 토글</td></tr>
    <tr><td><kbd>?</kbd></td><td>이 단축키 도움말</td></tr>
  </tbody>
</table>
<h2>인턴 종료 시점 체크리스트</h2>
<ul class="goal-checklist">
  <li><label><input type="checkbox" data-goal="cer-explain"> 한국어 ASR 의 평가 지표(CER) 를 자신 있게 설명할 수 있다</label></li>
  <li><label><input type="checkbox" data-goal="train-once"> Whisper 또는 SenseVoice 를 직접 학습시켜본 적이 있다</label></li>
  <li><label><input type="checkbox" data-goal="pipeline"> 데이터 파이프라인의 의미를 설명할 수 있다</label></li>
  <li><label><input type="checkbox" data-goal="slice"> 자신만의 평가 슬라이스로 모델 약점을 찾아본 적이 있다</label></li>
  <li><label><input type="checkbox" data-goal="pr"> PR 을 하나 이상 작성해 머지했다</label></li>
  <li><label><input type="checkbox" data-goal="cycle"> "검색 → 시도 → 질문" 의 사이클을 운영할 수 있다</label></li>
</ul>
"""
    body = hero + intro + "\n".join(cards) + extra
    return (
        '<section class="page home" data-route="home" hidden>'
        '<div class="crumbs">study / 학습 로드맵</div>'
        f'<article class="article">{body}</article>'
        '<div class="footer">© TRAIN-ASR · 한국어 ASR 학습·평가 프로젝트</div>'
        '</section>'
    )


def build_search_index(chapter_data: list[dict]) -> list[dict]:
    index: list[dict] = []
    for ch in chapter_data:
        text = (STUDY_DIR / ch["src"]).read_text(encoding="utf-8")
        cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
        index.append({
            "route": ch["slug"],
            "chapter": f'{ch["num"]}. {ch["title"]}',
            "heading": "",
            "id": "",
            "snippet": re.sub(r"\s+", " ", cleaned).strip()[:240],
        })
        sections = list(re.finditer(r"^##\s+(.+)$", cleaned, flags=re.M))
        for i, s in enumerate(sections):
            heading = s.group(1).strip()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(cleaned)
            chunk = cleaned[s.start(): end]
            body_text = re.sub(r"^##\s+.+\n?", "", chunk, count=1)
            body_text = re.sub(r"\s+", " ", body_text).strip()
            anchor_id = f'{ch["slug"]}--{slugify_unicode(heading, "-")}'
            index.append({
                "route": ch["slug"],
                "chapter": f'{ch["num"]}. {ch["title"]}',
                "heading": heading,
                "id": anchor_id,
                "snippet": body_text[:240],
            })
    return index


SEARCH_MODAL = """
<div class="modal" id="search-modal" aria-hidden="true">
  <div class="modal-backdrop"></div>
  <div class="modal-card" role="dialog" aria-label="검색">
    <div class="search-bar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
      <input type="search" id="search-input" placeholder="키워드 검색..." autocomplete="off" spellcheck="false">
      <kbd>Esc</kbd>
    </div>
    <div class="search-results" id="search-results"></div>
    <div class="search-tips"><kbd>↑</kbd><kbd>↓</kbd> 이동 &nbsp; <kbd>Enter</kbd> 열기 &nbsp; <kbd>Esc</kbd> 닫기</div>
  </div>
</div>
"""


def _load_inline_css_and_js() -> tuple[str, str]:
    """Extract current CSS and JS from existing index.html to avoid duplicating them in this file."""
    index_path = STUDY_DIR / "guide.html"
    if not index_path.exists():
        raise SystemExit(
            "docs/study/guide.html이 없으면 CSS/JS를 추출할 수 없습니다. "
            "현재 빌드 스크립트는 기존 guide.html의 CSS/JS를 그대로 재사용합니다."
        )
    html = index_path.read_text(encoding="utf-8")
    css_m = re.search(r"<style>\s*(.*?)\s*</style>", html, re.S)
    if not css_m:
        raise SystemExit("기존 guide.html에서 <style>을 찾지 못했습니다.")
    css = css_m.group(1)
    # JS: the last <script> block contains the SPA logic
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    js = next((s for s in scripts if "TRAIN-ASR Study" in s or "STUDY_CHAPTERS" in s and "function" in s), "")
    if not js:
        # fallback: last non-data script
        for s in reversed(scripts):
            if "window.STUDY_CHAPTERS" not in s.split("\n", 1)[0]:
                js = s
                break
    return css, js


def main() -> None:
    print("Building single-file study/guide.html ...")
    css, js = _load_inline_css_and_js()

    chapter_data: list[dict] = []
    sections_html: list[str] = [build_home_section()]
    for f, g, num, title, desc, mins in CHAPTERS:
        sections_html.append(build_chapter_section(f))
        chapter_data.append({
            "src": f, "slug": chapter_slug(f), "group": g,
            "num": num, "title": title, "desc": desc, "minutes": mins,
        })

    search_index = build_search_index(chapter_data)
    sidebar = build_sidebar()
    chapters_json = json.dumps(chapter_data, ensure_ascii=False)
    index_json = json.dumps(search_index, ensure_ascii=False)

    out = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>학습 로드맵 · TRAIN-ASR 학습 자료</title>
<style>
{css}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar" id="sidebar">
{sidebar}
</aside>
<main class="content" id="content">
{"".join(sections_html)}
</main>
<aside class="toc-pane" id="toc-pane" aria-label="이 페이지 목차">
<h4>이 페이지</h4>
<nav id="page-toc"></nav>
</aside>
</div>

{SEARCH_MODAL}

<script>
window.STUDY_CHAPTERS = {chapters_json};
window.STUDY_SEARCH_INDEX = {index_json};
</script>
<script>{js}</script>
</body>
</html>
"""
    (STUDY_DIR / "guide.html").write_text(out, encoding="utf-8")
    print(f"  wrote {(STUDY_DIR / 'guide.html').relative_to(STUDY_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
