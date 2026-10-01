# ============================================================
# allganize/RAG-Evaluation-Dataset-KO 의 documents.csv 를 읽고
# 원문 PDF를 data/allganize_rag_ko/<domain>/<file_name> 에 저장합니다.
#
# - URL이 PDF면 그대로 저장합니다.
# - URL이 게시판 페이지면 첨부파일 링크를 찾아 PDF를 받습니다.
#   첨부가 여러 개면 CSV의 pages(쪽수)와 일치하는 파일을 고릅니다.
# - 원 사이트에서 사라진 문서는 Wayback Machine 보관본을 찾아봅니다.
# - 이미 받은 파일은 쪽수가 맞으면 건너뜁니다.
#
# 실행: python allganize_download.py
# ============================================================

import csv
import difflib
import html
import io
import logging
import re
import ssl
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin

import requests
import urllib3
from pypdf import PdfReader
from requests.adapters import HTTPAdapter

urllib3.disable_warnings()
logging.getLogger("pypdf").setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "allganize_rag_ko"
CSV_URL = ("https://huggingface.co/datasets/allganize/"
           "RAG-Evaluation-Dataset-KO/resolve/main/documents.csv")
CSV_PATH = DATA_DIR / "documents.csv"

# 데이터셋이 만들어진 시점에 가까운 Wayback 보관본을 찾습니다.
WAYBACK_TIMESTAMP = "20240601"
TIMEOUT = 120
MAX_CANDIDATES = 8

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 첨부파일 다운로드로 보이는 링크 패턴
DOWNLOAD_HINT = re.compile(
    r"\.pdf|download|filedown|file_down|atchfile|getfile|downfile|"
    r"fileid|attach|다운로드", re.I)


# ============================================================
# HTTP 세션
# ============================================================
class LegacySSLAdapter(HTTPAdapter):
    """오래된 TLS 설정을 쓰는 공공기관 서버에도 접속합니다."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def make_session():
    session = requests.Session()
    session.mount("https://", LegacySSLAdapter())
    session.headers.update(HEADERS)
    return session


# ============================================================
# 공통 유틸
# ============================================================
def _norm(text):
    """비교용 정규화: NFC, 소문자, 확장자·공백·기호 제거"""
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = re.sub(r"\.pdf$", "", text.strip(), flags=re.I)
    return re.sub(r"[\s+_\-\.\(\)\[\]★·ᆞ,]", "", text).lower()


def _strip_tags(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def _attr(attrs, name):
    """태그 속성값. "...'...'..." 처럼 다른 따옴표가 섞인 값도 읽습니다."""
    m = re.search(name + r"""\s*=\s*(?:"([^"]*)"|'([^']*)')""", attrs, re.I)
    return html.unescape(m.group(1) or m.group(2) or "") if m else ""


def _decode(resp):
    """한국 공공기관 사이트는 EUC-KR이 섞여 있어 인코딩을 추정합니다."""
    enc = resp.encoding or ""
    if enc.lower() in ("", "iso-8859-1"):
        enc = resp.apparent_encoding or "utf-8"
    return resp.content.decode(enc, errors="replace")


def page_count(data):
    try:
        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return -1


# ============================================================
# 요청 하나 (GET 또는 POST) → PDF 바이트 / None
# ============================================================
def _request(session, req, referer=None, retries=3):
    method, url, form = req
    headers = {"Referer": referer} if referer else {}
    for attempt in range(retries):
        try:
            if method == "POST":
                resp = session.post(url, data=form, headers=headers,
                                    timeout=TIMEOUT, verify=False)
            else:
                resp = session.get(url, headers=headers,
                                   timeout=TIMEOUT, verify=False)
        except requests.RequestException:
            return None
        # archive.org 요청 제한(429)은 잠시 기다렸다가 다시 시도합니다.
        if resp.status_code == 429 and "archive.org" in url:
            time.sleep(30 * (attempt + 1))
            continue
        return resp
    return resp


def _pdf_or_none(resp):
    if resp is not None and resp.ok and resp.content[:5] == b"%PDF-":
        return resp.content
    return None


# ============================================================
# HTML에서 첨부파일 후보 요청 찾기
# ============================================================
def _candidates(page_html, page_url, file_name):
    target = _norm(file_name)
    found = []

    def add(score, req):
        found.append((score, req))

    # 닫히지 않은 <a>가 뒤의 링크를 삼키지 않도록 안쪽에 같은 태그를 허용하지 않습니다.
    for m in re.finditer(r"<(a|button)\b([^>]*)>((?:(?!<(?:a|button)\b).)*?)</\1>",
                         page_html, re.I | re.S):
        attrs, inner = m.group(2), m.group(3)
        # 링크 텍스트 + title + 바로 앞 텍스트(파일명이 링크 밖에 있는 경우)
        before = _strip_tags(page_html[max(0, m.start() - 300):m.start()])
        label = " ".join([_strip_tags(inner), _attr(attrs, "title"),
                          before[-120:]])

        href = _attr(attrs, "href")
        onclick = _attr(attrs, "onclick")

        reqs = []
        # 대법원: javascript:download('저장명','원본명') → POST
        sc = re.search(r"download\('([^']+)','([^']+)'\)", href)
        if sc:
            path = re.search(r"form\.path\.value\s*=\s*'(\d+)'", page_html)
            reqs.append(("POST", "https://file.scourt.go.kr//AttachDownload",
                         {"file": sc.group(1),
                          "path": path.group(1) if path else "003",
                          "downFile": sc.group(2).encode("euc-kr", "ignore")}))
            label += " " + sc.group(2)
        elif href and not href.lower().startswith(("javascript", "#")):
            reqs.append(("GET", urljoin(page_url, href), None))
        # 버튼: onclick="location.href='...'"
        loc = re.search(r"location\.href\s*=\s*'([^']+)'", onclick)
        if loc:
            reqs.append(("GET", urljoin(page_url, loc.group(1)), None))
        if not reqs:
            continue

        score = difflib.SequenceMatcher(None, target, _norm(label)).ratio()
        if target and target in _norm(label):
            score += 1.0
        # 다운로드 링크는 첨부 이름이 제목과 달라도(사건번호 등) 후보로 둡니다.
        # 잘못된 파일은 쪽수 검증에서 걸러집니다.
        if DOWNLOAD_HINT.search(href + onclick) or DOWNLOAD_HINT.search(label):
            score = max(score, 0.2) + 0.3
        if score >= 0.5:
            for req in reqs:
                add(score, req)

    found.sort(key=lambda x: -x[0])
    seen, result = set(), []
    for _, req in found:
        key = (req[0], req[1], str(req[2]))
        if key not in seen:
            seen.add(key)
            result.append(req)
    return result[:MAX_CANDIDATES]


# ============================================================
# URL 하나에서 PDF 후보들을 모읍니다.
# 쪽수가 정확히 맞는 PDF를 찾으면 바로 반환합니다.
# ============================================================
def _collect(session, url, file_name, pages):
    resp = _request(session, ("GET", url, None))
    if resp is None:
        return []
    data = _pdf_or_none(resp)
    if data:
        return [data]

    page_html = _decode(resp)
    pdfs = []
    for req in _candidates(page_html, resp.url, file_name):
        data = _pdf_or_none(_request(session, req, referer=resp.url))
        if data:
            pdfs.append(data)
            if page_count(data) == pages:
                break
    return pdfs


def _wayback_url(session, url):
    """200 응답으로 보관된 캡처 중 WAYBACK_TIMESTAMP에 가장 가까운 것"""
    cdx = requests.Request(
        "GET", "https://web.archive.org/cdx/search/cdx",
        params={"url": url, "output": "json", "filter": "statuscode:200",
                "fl": "timestamp,original"}).prepare().url
    resp = _request(session, ("GET", cdx, None))
    try:
        rows = resp.json()[1:] if resp is not None and resp.ok else []
    except ValueError:
        rows = []

    if not rows:
        # CDX가 응답하지 않으면 가용성 API로 대체합니다(가장 가까운 캡처 1건).
        api = requests.Request(
            "GET", "https://archive.org/wayback/available",
            params={"url": url}).prepare().url
        resp = _request(session, ("GET", api, None))
        try:
            snap = resp.json()["archived_snapshots"]["closest"]
        except (AttributeError, ValueError, KeyError, TypeError):
            return None
        if snap.get("status") != "200":
            return None
        rows = [[snap["timestamp"], url]]

    ts, original = min(rows, key=lambda r: abs(
        int(r[0][:8]) - int(WAYBACK_TIMESTAMP)))
    # PDF는 원본 그대로(id_), HTML은 링크가 보관본으로 바뀐 페이지를 받습니다.
    if original.lower().split("?")[0].endswith(".pdf"):
        return f"https://web.archive.org/web/{ts}id_/{original}"
    return f"https://web.archive.org/web/{ts}/{original}"


def _best(pdfs, pages):
    """쪽수가 가장 가까운 PDF를 고릅니다. (데이터, 쪽수)"""
    best = None
    for data in pdfs:
        n = page_count(data)
        if n <= 0:
            continue
        if best is None or abs(n - pages) < abs(best[1] - pages):
            best = (data, n)
    return best


def fetch(session, url, file_name, pages):
    """(PDF 바이트, 쪽수, 출처) 를 반환하고, 실패하면 None을 반환합니다."""
    best = _best(_collect(session, url, file_name, pages), pages)
    if best and best[1] == pages:
        return best + ("live",)

    snap = _wayback_url(session, url)
    if snap:
        wb = _best(_collect(session, snap, file_name, pages), pages)
        if wb and (best is None or abs(wb[1] - pages) < abs(best[1] - pages)):
            return wb + ("wayback",)

    return best + ("live",) if best else None


# ============================================================
# 실행
# ============================================================
def load_rows():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        r = requests.get(CSV_URL, timeout=60)
        r.raise_for_status()
        CSV_PATH.write_bytes(r.content)
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    session = make_session()
    ok, skipped, mismatched, failed = 0, 0, [], []

    for i, row in enumerate(rows, 1):
        dest = DATA_DIR / row["domain"] / row["file_name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        pages = int(row["pages"])
        tag = f"[{i:2}/{len(rows)}] {row['domain']:8} {row['file_name'][:45]}"

        # 이미 있고 쪽수가 맞으면 건너뜁니다.
        old_n = page_count(dest.read_bytes()) if dest.exists() else -1
        if old_n == pages:
            skipped += 1
            print(f"{tag}  (이미 있음)")
            continue

        result = fetch(session, row["url"], row["file_name"], pages)

        # 새로 받은 것이 기존 파일보다 쪽수가 가깝지 않으면 기존 파일 유지
        if result and old_n > 0 and abs(result[1] - pages) >= abs(old_n - pages):
            result = None
        if result:
            data, n, origin = result
            dest.write_bytes(data)
            ok += 1
            note = "" if n == pages else f"  ⚠ 쪽수 {n} (기대 {pages})"
            print(f"{tag}  ✓ {origin} {len(data) // 1024} KB{note}")
            if n != pages:
                mismatched.append((row, n))
        elif old_n > 0:
            skipped += 1
            mismatched.append((row, old_n))
            print(f"{tag}  (기존 파일 유지, 쪽수 {old_n} / 기대 {pages})")
        else:
            failed.append(row)
            print(f"{tag}  ✗ PDF를 찾지 못함")

    print(f"\n완료: 신규 {ok}, 기존 {skipped}, 실패 {len(failed)} / 전체 {len(rows)}")
    if mismatched:
        print("\n쪽수 불일치:")
        for row, n in mismatched:
            print(f"  - {row['domain']} | {row['file_name']} | {n}/{row['pages']}")
    if failed:
        print("\n실패:")
        for row in failed:
            print(f"  - {row['domain']} | {row['file_name']} | {row['url']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
