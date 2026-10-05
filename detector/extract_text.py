"""
Text extraction for the 4 supported input types: TEXT, PDF, TXT, URL.

Every function returns (text, error). `error` is a short user-safe message;
`text` is None when extraction failed. No raw exceptions/stack traces are
ever surfaced to the caller.

extract_from_url() also returns a `metadata` dict (domain, title, author,
published date, HTTPS flag) - this feeds Layer 3 (source analysis) in
detector/verification/. Extraction stays here rather than duplicated
elsewhere, per the project's "don't duplicate existing functionality" rule.
"""
import ipaddress
import re
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_TEXT_LENGTH = 30
MAX_TEXT_LENGTH = 50000
URL_TIMEOUT_SECONDS = 10
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB - cap how much of a page we download
USER_AGENT = "TruthLensAI-Bot/1.0 (+https://truthlens.ai)"


def validate_pasted_text(text: str):
    text = (text or "").strip()
    if not text:
        return None, "Please paste some article text."
    if len(text) < MIN_TEXT_LENGTH:
        return None, f"The text is too short to analyze (minimum {MIN_TEXT_LENGTH} characters)."
    if len(text) > MAX_TEXT_LENGTH:
        return None, f"The text is too long (maximum {MAX_TEXT_LENGTH} characters). Please shorten it."
    return text, None


def extract_from_pdf(uploaded_file):
    try:
        if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
            return None, "PDF file is too large (maximum 10 MB)."

        reader = PdfReader(uploaded_file)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return None, "This PDF is password-protected and cannot be read."

        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                continue

        text = "\n".join(pages_text).strip()
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return None, "No extractable text was found in this PDF (it may be a scanned image)."
        if len(text) < MIN_TEXT_LENGTH:
            return None, "The PDF does not contain enough readable text to analyze."

        return text[:MAX_TEXT_LENGTH], None
    except Exception:
        return None, "This PDF could not be read. It may be corrupted or in an unsupported format."


def extract_from_txt(uploaded_file):
    try:
        if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
            return None, "Text file is too large (maximum 10 MB)."
        if not uploaded_file.name.lower().endswith(".txt"):
            return None, "Only .txt files are supported for text upload."

        raw = uploaded_file.read()
        text = None
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if text is None:
            return None, "The file's text encoding could not be recognized."

        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < MIN_TEXT_LENGTH:
            return None, "The uploaded file does not contain enough text to analyze."

        return text[:MAX_TEXT_LENGTH], None
    except Exception:
        return None, "This text file could not be read."


def _is_private_or_reserved(hostname: str) -> bool:
    """SSRF guard: resolve the hostname and reject anything pointing at
    loopback/private/link-local/reserved address space, so a URL can't be
    used to make the server fetch its own internal network."""
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True  # can't resolve -> treat as unsafe rather than risk it
    for info in addr_infos:
        ip = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
                or ip_obj.is_reserved or ip_obj.is_multicast or ip_obj.is_unspecified):
            return True
    return False


def _extract_metadata(soup: BeautifulSoup, url: str, is_https: bool) -> dict:
    """Layer 3 groundwork: pull whatever source/publication signals are
    actually present in the page. Missing fields stay None/empty - nothing
    here is guessed or fabricated."""
    domain = urlparse(url).netloc.lower().removeprefix("www.")

    def meta(*names):
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = meta("og:title", "twitter:title")
    if not title and soup.title:
        title = soup.title.get_text(strip=True)

    author = meta("author", "article:author", "og:author", "twitter:creator")
    if not author:
        byline = soup.find(attrs={"class": re.compile(r"byline|author", re.I)})
        if byline:
            author = byline.get_text(" ", strip=True)[:150]

    published = meta("article:published_time", "og:article:published_time",
                      "datePublished", "publish-date", "date")
    if not published:
        time_tag = soup.find("time")
        if time_tag:
            published = time_tag.get("datetime") or time_tag.get_text(strip=True)

    site_name = meta("og:site_name")

    return {
        "domain": domain,
        "url": url,
        "is_https": is_https,
        "title": title,
        "author": author,
        "published_date": published,
        "site_name": site_name,
    }


def extract_from_url(url: str, timeout: int = URL_TIMEOUT_SECONDS):
    """Returns (text, error, metadata). metadata is a dict (possibly with
    None fields) when text extraction succeeds, else {}."""
    url = (url or "").strip()
    if not url:
        return None, "Please enter a URL.", {}

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None, "Please enter a valid URL starting with http:// or https://.", {}

    hostname = parsed.hostname or ""
    if _is_private_or_reserved(hostname):
        return None, "This URL points to a restricted or unreachable address and cannot be fetched.", {}

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
    except requests.exceptions.Timeout:
        return None, "The request to this URL timed out. Please try again or paste the text directly.", {}
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to this website. It may be down or blocking requests.", {}
    except requests.exceptions.RequestException:
        return None, "This URL could not be reached.", {}

    # Re-check the FINAL (post-redirect) host too, in case a redirect
    # points at an internal address.
    final_host = urlparse(resp.url).hostname or ""
    if final_host and _is_private_or_reserved(final_host):
        return None, "This URL redirects to a restricted address and cannot be fetched.", {}

    if resp.status_code == 403:
        return None, "This website is blocking automated access. Please paste the article text directly.", {}
    if resp.status_code == 404:
        return None, "This page could not be found (404).", {}
    if resp.status_code >= 400:
        return None, f"This website returned an error (HTTP {resp.status_code}).", {}

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None, "This URL does not point to a readable web page.", {}

    # Cap how much we read, regardless of what Content-Length claims.
    raw_bytes = b""
    for chunk in resp.iter_content(chunk_size=65536):
        raw_bytes += chunk
        if len(raw_bytes) > MAX_RESPONSE_BYTES:
            break
    html = raw_bytes.decode(resp.encoding or "utf-8", errors="replace")

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None, "This page's content could not be parsed.", {}

    metadata = _extract_metadata(soup, resp.url, is_https=urlparse(resp.url).scheme == "https")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # Prefer <article>, fall back to largest cluster of <p> tags.
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = article_tag.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = re.sub(r"\s+", " ", text).strip()

    if not text or len(text) < MIN_TEXT_LENGTH:
        return None, "No readable article content could be extracted from this page.", {}

    return text[:MAX_TEXT_LENGTH], None, metadata
