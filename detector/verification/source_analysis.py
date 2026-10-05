"""
Layer 3 - Source / domain analysis.

IMPORTANT: this produces SIGNALS, not a verdict. "Unknown domain" does NOT
mean fake; "well-known domain" does NOT mean real. A reputable outlet can
publish something wrong, and an unfamiliar site can report something true.
This module's output is one input to detector/verification/aggregation.py,
weighted deliberately low relative to evidence-based signals.
"""

from urllib.parse import urlparse

# A small, explicitly-labeled reference list used ONLY as a weak signal
# ("commonly cited as an established news organization") - never as a
# REAL/FAKE determination by itself. Absence from this list is NOT a
# negative signal; it just means "no prior signal available."
ESTABLISHED_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "wsj.com", "nytimes.com", "washingtonpost.com", "theguardian.com",
    "bloomberg.com", "afp.com", "cnn.com", "abcnews.go.com", "cbsnews.com",
    "nbcnews.com", "pbs.org", "usatoday.com", "time.com", "economist.com",
}

# Domain patterns historically associated with satire (explicitly NOT a
# fake-news accusation - satire is a legitimate genre, just not literal news).
KNOWN_SATIRE_DOMAINS = {
    "theonion.com", "babylonbee.com", "clickhole.com", "newsthump.com", "thebeaverton.com",
}

# Tier 1 (Very High, Weight ~2.0): Official government & international institutional bodies
VERY_HIGH_PRIMARY_HINTS = (
    ".gov", ".mil", ".gov.in", ".gov.uk", ".gov.au", ".gov.ca", ".gc.ca",
    ".nic.in", ".gov.sg", ".europa.eu",
)

VERY_HIGH_PRIMARY_DOMAINS = {
    "who.int", "un.org", "unicef.org", "courtlistener.com", "sec.gov", "nih.gov",
    "cdc.gov", "supremecourt.gov", "federalregister.gov", "nasa.gov", "isro.gov.in",
    "pib.gov.in", "whitehouse.gov", "parliament.uk", "rbi.org.in", "worldbank.org",
    "imf.org", "wto.org", "fbi.gov", "justice.gov", "defense.gov", "epa.gov",
    "fda.gov", "weather.gov", "state.gov", "treasury.gov", "hhs.gov", "unesco.org",
}

# Tier 2 (High, Weight ~1.2): Major wire services, top-tier global news, and accredited fact-checkers
HIGH_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "afp.com", "upi.com", "bloomberg.com",
    "bbc.com", "bbc.co.uk", "npr.org", "wsj.com", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "ft.com", "pbs.org", "economist.com", "time.com", "usatoday.com",
    "abcnews.go.com", "cbsnews.com", "nbcnews.com", "cnn.com", "dw.com", "aljazeera.com",
    "france24.com", "kyodonews.net", "lemonde.fr", "spiegel.de", "theatlantic.com",
    "nature.com", "science.org", "thelancet.com", "nejm.org",
}

HIGH_FACTCHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
    "checkyourfact.com", "boomlive.in", "altnews.in", "vishvasnews.com",
    "afpcheck.afp.com", "factcheck.afp.com", "factly.in",
}

# Tier 3 (Medium, Weight ~0.75): Established regional & national news publications
MEDIUM_REGIONAL_DOMAINS = {
    "thehindu.com", "indianexpress.com", "hindustantimes.com", "timesofindia.com",
    "ndtv.com", "indiatoday.in", "livemint.com", "business-standard.com", "theprint.in",
    "thewire.in", "deccanherald.com", "tribuneindia.com", "scmp.com", "japantimes.co.jp",
    "smh.com.au", "theage.com.au", "cbc.ca", "theglobeandmail.com", "telegraph.co.uk",
    "independent.co.uk", "latimes.com", "chicagotribune.com", "bostonglobe.com",
    "politico.com", "axios.com", "propublica.org", "thehill.com", "newsweek.com",
    "vox.com", "slate.com", "marketwatch.com", "fortune.com", "forbes.com",
    "businessinsider.com",
}


def normalize_domain(domain_or_url: str) -> str:
    """Extracts a clean, normalized domain string from a domain or full URL."""
    if not domain_or_url:
        return ""
    val = domain_or_url.strip().lower()
    if "://" in val:
        parsed = urlparse(val).netloc.lower()
        if parsed:
            val = parsed
    val = val.split("/")[0].split(":")[0]
    if val.startswith("www."):
        val = val[4:]
    return val


USER_GENERATED_DOMAINS = {
    "twitter.com", "x.com", "reddit.com", "facebook.com", "instagram.com",
    "t.me", "telegram.org", "tiktok.com", "threads.net", "youtube.com",
}

BLOG_DOMAINS = {
    "medium.com", "substack.com", "wordpress.com", "blogspot.com", "tumblr.com",
}

AGGREGATOR_DOMAINS = {
    "news.google.com", "news.yahoo.com", "flipboard.com", "msn.com",
}

PRIMARY_DOC_DOMAINS = {
    "courtlistener.com", "sec.gov", "federalregister.gov", "supremecourt.gov",
    "supremecourtofindia.nic.in", "ecf.uscourts.gov",
}


def classify_source_reliability(domain_or_url: str) -> dict:
    """
    Classifies source quality and reliability into explicit hierarchy (Section 8):
    - OFFICIAL_PRIMARY / GOVERNMENT / PRIMARY_DOCUMENT (weight ~2.0)
    - MAJOR_NEWS / FACT_CHECKER (weight ~1.2)
    - REPUTABLE_SECONDARY (weight ~0.85)
    - GENERAL_WEB (weight ~0.5)
    - BLOG / AGGREGATOR (weight ~0.3)
    - USER_GENERATED (weight ~0.15)
    - SATIRE (weight 0.0)
    """
    domain = normalize_domain(domain_or_url)
    if not domain:
        return {
            "category": "GENERAL_WEB",
            "tier": "LOW",
            "tier_label": "Low (Unspecified / Unknown Source)",
            "weight": 0.3,
            "is_primary_source": False,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": "",
        }

    # Satire check
    if domain in KNOWN_SATIRE_DOMAINS or any(domain.endswith("." + s) for s in KNOWN_SATIRE_DOMAINS):
        return {
            "category": "SATIRE",
            "tier": "SATIRE",
            "tier_label": "Satire Publication",
            "weight": 0.0,
            "is_primary_source": False,
            "is_fact_checker": False,
            "is_satire": True,
            "domain": domain,
        }

    # Primary Document check
    if domain in PRIMARY_DOC_DOMAINS or any(domain.endswith("." + d) for d in PRIMARY_DOC_DOMAINS):
        return {
            "category": "PRIMARY_DOCUMENT",
            "tier": "VERY_HIGH",
            "tier_label": "Very High (Primary Legal/Regulatory Document)",
            "weight": 2.0,
            "is_primary_source": True,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": domain,
        }

    # Very High: Official government and primary institutional bodies
    is_gov_or_mil = any(domain.endswith(hint) for hint in VERY_HIGH_PRIMARY_HINTS)
    is_inst = domain in VERY_HIGH_PRIMARY_DOMAINS or any(domain.endswith("." + d) for d in VERY_HIGH_PRIMARY_DOMAINS)
    if is_gov_or_mil or is_inst:
        category = "GOVERNMENT" if is_gov_or_mil else "OFFICIAL_PRIMARY"
        return {
            "category": category,
            "tier": "VERY_HIGH",
            "tier_label": "Very High (Official / Primary Source)",
            "weight": 2.0,
            "is_primary_source": True,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": domain,
        }

    # High: Accredited Fact-Checking Organizations
    if domain in HIGH_FACTCHECK_DOMAINS or any(domain.endswith("." + fc) for fc in HIGH_FACTCHECK_DOMAINS):
        return {
            "category": "FACT_CHECKER",
            "tier": "HIGH",
            "tier_label": "High (Verified Fact-Checker)",
            "weight": 1.2,
            "is_primary_source": False,
            "is_fact_checker": True,
            "is_satire": False,
            "domain": domain,
        }

    # High: Wire Services and Global Reputable Outlets, or Academic (.edu / .ac.uk)
    is_high_news = domain in HIGH_NEWS_DOMAINS or any(domain.endswith("." + n) for n in HIGH_NEWS_DOMAINS)
    is_academic = domain.endswith(".edu") or domain.endswith(".ac.uk")
    if is_high_news or is_academic:
        return {
            "category": "MAJOR_NEWS",
            "tier": "HIGH",
            "tier_label": "High (Major News / Wire Service)",
            "weight": 1.2,
            "is_primary_source": False,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": domain,
        }

    # Medium: Regional Publications
    is_med_news = domain in MEDIUM_REGIONAL_DOMAINS or any(domain.endswith("." + m) for m in MEDIUM_REGIONAL_DOMAINS)
    if is_med_news:
        return {
            "category": "REPUTABLE_SECONDARY",
            "tier": "MEDIUM",
            "tier_label": "Medium (Reputable Secondary / Regional News)",
            "weight": 0.85,
            "is_primary_source": False,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": domain,
        }

    # User Generated / Social Media
    if domain in USER_GENERATED_DOMAINS or any(domain.endswith("." + u) for u in USER_GENERATED_DOMAINS):
        return {
            "category": "USER_GENERATED",
            "tier": "LOW",
            "tier_label": "Low (User-Generated Content / Social Media)",
            "weight": 0.15,
            "is_primary_source": False,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": domain,
        }

    # Blog / Aggregator
    if domain in BLOG_DOMAINS or any(domain.endswith("." + b) for b in BLOG_DOMAINS):
        return {
            "category": "BLOG",
            "tier": "LOW",
            "tier_label": "Low (Blog / Independent Commentary)",
            "weight": 0.3,
            "is_primary_source": False,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": domain,
        }
    if domain in AGGREGATOR_DOMAINS or any(domain.endswith("." + a) for a in AGGREGATOR_DOMAINS):
        return {
            "category": "AGGREGATOR",
            "tier": "LOW",
            "tier_label": "Low (Content Aggregator)",
            "weight": 0.25,
            "is_primary_source": False,
            "is_fact_checker": False,
            "is_satire": False,
            "domain": domain,
        }

    # Default General Web
    return {
        "category": "GENERAL_WEB",
        "tier": "LOW",
        "tier_label": "Low (General Web Source)",
        "weight": 0.5,
        "is_primary_source": False,
        "is_fact_checker": False,
        "is_satire": False,
        "domain": domain,
    }


def analyze_source(metadata: dict) -> dict:
    """metadata comes from detector.extract_text._extract_metadata (empty
    dict for pasted-text input, where there is no source to analyze)."""
    if not metadata or not metadata.get("domain"):
        return {
            "available": False,
            "domain": None,
            "signals": ["No URL was supplied, so no source/domain analysis is available for this input."],
            "reference_list_match": None,
            "reliability": classify_source_reliability(""),
        }

    domain = normalize_domain(metadata["domain"])
    signals = []

    rel = classify_source_reliability(domain)
    reference_match = None

    if rel["is_satire"]:
        reference_match = "satire"
        signals.append(
            f"'{domain}' is an identified satire publication. Content here is intentionally not "
            f"literal news, which is a different category from 'fake news'."
        )
    elif rel["is_primary_source"]:
        reference_match = "established_news_organization"
        signals.append(
            f"'{domain}' is an official government or institutional primary source ({rel['tier_label']}). "
            f"Primary announcements carry very high evidentiary weight."
        )
    elif rel["tier"] == "HIGH":
        reference_match = "established_news_organization"
        signals.append(
            f"'{domain}' is commonly cited as an established reputable news organization or fact-checker. "
            f"This is a strong positive signal for reporting standards."
        )
    elif rel["tier"] == "MEDIUM":
        signals.append(
            f"'{domain}' is an established regional or digital news publication ({rel['tier_label']})."
        )
    else:
        signals.append(
            f"'{domain}' is not in TruthLens AI's reference list of major established outlets. "
            f"This is NOT evidence the article is fake - most legitimate publishers "
            f"worldwide simply aren't on any such list."
        )

    if metadata.get("is_https"):
        signals.append("The source uses HTTPS.")
    else:
        signals.append("The source does not use HTTPS, which is a minor transparency concern.")

    if metadata.get("author"):
        signals.append(f"Author information is available: {metadata['author']}.")
    else:
        signals.append("No author information could be detected on this page.")

    if metadata.get("published_date"):
        signals.append(f"Publication date is available: {metadata['published_date']}.")
    else:
        signals.append("No publication date could be detected on this page.")

    return {
        "available": True,
        "domain": domain,
        "site_name": metadata.get("site_name"),
        "title": metadata.get("title"),
        "author": metadata.get("author"),
        "published_date": metadata.get("published_date"),
        "is_https": metadata.get("is_https"),
        "reference_list_match": reference_match,
        "reliability": rel,
        "signals": signals,
    }
