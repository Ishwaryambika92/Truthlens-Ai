"""
TruthLens AI test suite.

Covers:
  - Layer 2 (quality gate), Layer 3 (source analysis) - pure logic, no mocking needed
  - Layer 5 (claims) and Layers 7-8 (web evidence) - mocked Anthropic API calls,
    since these require ANTHROPIC_API_KEY to hit the real API. Mocking lets us
    verify parsing/error-handling correctness (malformed JSON, timeouts, missing
    URLs, invalid status values) without a live key or network access.
  - Aggregation (Layer 9-10) - verifies the documented scoring policy, in
    particular that independent contradicting evidence overrides a
    high-confidence ML prediction (the core "ML says FAKE != fake" principle).
  - Full end-to-end views: auth, all 4 input types, history isolation, PDF
    report, chatbot - via Django's test Client.

Run with:  python manage.py test detector
"""
import io
import json
import os
from unittest.mock import patch, MagicMock

import requests
from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase, Client

from .verification import quality, temporal, source_analysis, claims as claims_module, web_evidence, aggregation
from .verification.canonical import build_canonical_verification_result
from .models import Analysis


class QualityCheckTests(TestCase):
    def test_empty_text_is_unusable(self):
        result = quality.check_article_quality("")
        self.assertFalse(result["usable"])

    def test_short_text_is_unusable(self):
        result = quality.check_article_quality("Short update today.")
        self.assertFalse(result["usable"])
        self.assertTrue(any("too short" in r for r in result["reasons"]))

    def test_normal_length_text_is_usable(self):
        text = ("The city council voted 6-3 on Tuesday to approve a new budget for road "
                "maintenance in the coming fiscal year. Officials said work would begin in spring.")
        result = quality.check_article_quality(text)
        self.assertTrue(result["usable"])

    def test_repeated_line_spam_is_unusable(self):
        text = "\n".join(["Buy now buy now buy now click here"] * 10)
        result = quality.check_article_quality(text)
        self.assertFalse(result["usable"])

    def test_missing_metadata_produces_warnings_not_failure(self):
        text = "The mayor spoke today about the new policy plans for the city budget review process this year."
        result = quality.check_article_quality(text, metadata={"title": None, "author": None,
                                                                  "published_date": None, "is_https": False})
        self.assertGreater(len(result["warnings"]), 0)


class SourceAnalysisTests(TestCase):
    def test_no_metadata_reports_unavailable_not_fake(self):
        result = source_analysis.analyze_source({})
        self.assertFalse(result["available"])

    def test_unknown_domain_is_not_treated_as_fake_signal(self):
        result = source_analysis.analyze_source({"domain": "some-random-blog.example", "is_https": True})
        self.assertIsNone(result["reference_list_match"])
        self.assertTrue(any("NOT evidence the article is fake" in s for s in result["signals"]))

    def test_known_outlet_is_only_a_weak_positive_signal(self):
        result = source_analysis.analyze_source({"domain": "reuters.com", "is_https": True})
        self.assertEqual(result["reference_list_match"], "established_news_organization")


class ClaimExtractionMockedTests(TestCase):
    """Layer 5. ANTHROPIC_API_KEY is not available in this environment, so
    these tests mock the HTTP call to verify parsing/degradation logic."""

    def setUp(self):
        self._orig_gemini = getattr(claims_module, "GEMINI_API_KEY", "")
        self._orig_anthropic = getattr(claims_module, "ANTHROPIC_API_KEY", "")
        claims_module.GEMINI_API_KEY = ""

    def tearDown(self):
        claims_module.GEMINI_API_KEY = self._orig_gemini
        claims_module.ANTHROPIC_API_KEY = self._orig_anthropic

    def test_no_api_key_degrades_gracefully(self):
        claims_module.ANTHROPIC_API_KEY = ""
        result = claims_module.extract_claims("Some article text about a real event.")
        self.assertFalse(result["available"])
        self.assertEqual(result["claims"], [])

    def test_well_formed_response_is_parsed(self):
        claims_module.ANTHROPIC_API_KEY = "fake-test-key"
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": '["Claim A.", "Claim B."]'}]
        }
        with patch("detector.verification.claims.requests.post", return_value=mock_resp):
            result = claims_module.extract_claims("Article text.")
        self.assertTrue(result["available"])
        self.assertEqual(result["claims"], ["Claim A.", "Claim B."])

    def test_malformed_json_does_not_crash(self):
        claims_module.ANTHROPIC_API_KEY = "fake-test-key"
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "not json {{{"}]}
        with patch("detector.verification.claims.requests.post", return_value=mock_resp):
            result = claims_module.extract_claims("Article text.")
        self.assertFalse(result["available"])
        self.assertEqual(result["claims"], [])

    def test_timeout_is_handled(self):
        claims_module.ANTHROPIC_API_KEY = "fake-test-key"
        with patch("detector.verification.claims.requests.post", side_effect=requests.exceptions.Timeout):
            result = claims_module.extract_claims("Article text.")
        self.assertFalse(result["available"])

    def test_generic_request_exception_is_handled(self):
        claims_module.ANTHROPIC_API_KEY = "fake-test-key"
        with patch("detector.verification.claims.requests.post", side_effect=requests.exceptions.ConnectionError):
            result = claims_module.extract_claims("Article text.")
        self.assertFalse(result["available"])
        self.assertEqual(result["claims"], [])


class WebEvidenceHardenedTests(TestCase):
    """Layers 7-8, hardened version. ANTHROPIC_API_KEY is not available in
    this environment, so these tests mock the two-stage LLM calls (search,
    then content-grounded judgment) plus the independent source-fetching
    step, to verify the hardening logic itself: every source must be
    independently fetched/validated before it can count as evidence, and
    verdicts must come from actual retrieved content, never a bare LLM
    assertion."""

    def setUp(self):
        self._orig_tavily = getattr(web_evidence, "TAVILY_API_KEY", "")
        self._orig_gemini = getattr(web_evidence, "GEMINI_API_KEY", "")
        self._orig_anthropic = getattr(web_evidence, "ANTHROPIC_API_KEY", "")
        web_evidence.ANTHROPIC_API_KEY = "fake-test-key"
        web_evidence.TAVILY_API_KEY = ""
        web_evidence.GEMINI_API_KEY = ""

    def tearDown(self):
        web_evidence.TAVILY_API_KEY = self._orig_tavily
        web_evidence.GEMINI_API_KEY = self._orig_gemini
        web_evidence.ANTHROPIC_API_KEY = self._orig_anthropic

    def _mock_search_response(self, candidates):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"content": [{"type": "text", "text": json.dumps(candidates)}]}
        return resp

    def _mock_judgment_response(self, verdicts):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"content": [{"type": "text", "text": json.dumps(verdicts)}]}
        return resp

    # --- valid source: full happy path, search -> fetch -> judge ---
    def test_valid_source_supports_claim(self):
        search_resp = self._mock_search_response([{"title": "Official page", "url": "https://example.com/about"}])
        judge_resp = self._mock_judgment_response([{"verdict": "SUPPORTS", "reason": "The retrieved text confirms it."}])

        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url",
                   return_value=("Real retrieved article text about the claim.", None,
                                 {"domain": "example.com", "title": "Official page", "is_https": True})):
            result = web_evidence.verify_claim_with_evidence("Company X launched Product Y.")

        self.assertEqual(result["status"], "SUPPORTED")
        self.assertTrue(result["sources"][0]["retrieved"])
        self.assertEqual(result["sources"][0]["stage"], "SOURCE_CONTENT_SUPPORTS_CLAIM")

    # --- invalid / malformed URL: rejected before any fetch attempt ---
    def test_malformed_url_rejected_before_fetch(self):
        search_resp = self._mock_search_response([{"title": "Fake", "url": "not-a-real-url-at-all"}])
        with patch("detector.verification.web_evidence.requests.post", return_value=search_resp):
            result = web_evidence.verify_claim_with_evidence("Some claim.")
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertFalse(result["sources"][0]["retrieved"])
        self.assertIn("Malformed", result["sources"][0]["validation_error"])

    # --- fabricated URL attempt: well-formed but does not actually exist/resolve ---
    def test_fabricated_url_that_fails_to_resolve_is_dropped(self):
        search_resp = self._mock_search_response(
            [{"title": "Nonexistent", "url": "https://this-domain-truthlens-fabricated-test.invalid/story"}]
        )
        with patch("detector.verification.web_evidence.requests.post", return_value=search_resp), \
             patch("detector.extract_text.extract_from_url",
                   return_value=(None, "Could not connect to this website. It may be down or blocking requests.", {})):
            result = web_evidence.verify_claim_with_evidence("A claim citing a fabricated source.")
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertFalse(result["sources"][0]["retrieved"])

    # --- unreachable source (timeout/connection error during fetch) ---
    def test_unreachable_source_does_not_crash_and_is_excluded(self):
        search_resp = self._mock_search_response([{"title": "Down site", "url": "https://down-site.example/page"}])
        with patch("detector.verification.web_evidence.requests.post", return_value=search_resp), \
             patch("detector.extract_text.extract_from_url",
                   return_value=(None, "The request to this URL timed out. Please try again or paste the text directly.", {})):
            result = web_evidence.verify_claim_with_evidence("A claim.")
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertEqual(len(result["sources"]), 1)
        self.assertFalse(result["sources"][0]["retrieved"])

    # --- malformed source (fetch "succeeds" but content is junk/too short - extract_from_url itself rejects it) ---
    def test_malformed_source_content_rejected_by_extractor(self):
        search_resp = self._mock_search_response([{"title": "Junk page", "url": "https://junk.example/x"}])
        with patch("detector.verification.web_evidence.requests.post", return_value=search_resp), \
             patch("detector.extract_text.extract_from_url",
                   return_value=(None, "No readable article content could be extracted from this page.", {})):
            result = web_evidence.verify_claim_with_evidence("A claim.")
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertFalse(result["sources"][0]["retrieved"])

    # --- source content unrelated to the claim: retrieved fine, but judgment says UNRELATED ---
    def test_retrieved_but_unrelated_content_does_not_count_as_evidence(self):
        search_resp = self._mock_search_response([{"title": "Unrelated page", "url": "https://example.com/other-topic"}])
        judge_resp = self._mock_judgment_response([{"verdict": "UNRELATED", "reason": "Text is about a different topic."}])
        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url",
                   return_value=("Some completely unrelated article text.", None,
                                 {"domain": "example.com", "title": "Unrelated page", "is_https": True})):
            result = web_evidence.verify_claim_with_evidence("A specific claim about something else.")
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertEqual(result["sources"][0]["stage"], "SOURCE_NOT_SUFFICIENT")

    # --- supporting source ---
    def test_supporting_source_produces_supported_status(self):
        search_resp = self._mock_search_response([{"title": "Confirming page", "url": "https://reuters.com/story"}])
        judge_resp = self._mock_judgment_response([{"verdict": "SUPPORTS", "reason": "Directly confirms the event."}])
        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url",
                   return_value=("Detailed confirming article text.", None,
                                 {"domain": "reuters.com", "title": "Confirming page", "is_https": True})):
            result = web_evidence.verify_claim_with_evidence("An event occurred.")
        self.assertEqual(result["status"], "SUPPORTED")

    # --- contradicting source ---
    def test_contradicting_source_produces_contradicted_status(self):
        search_resp = self._mock_search_response([{"title": "Denial", "url": "https://official.gov/statement"}])
        judge_resp = self._mock_judgment_response([{"verdict": "CONTRADICTS", "reason": "Official statement denies this."}])
        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url",
                   return_value=("Official denial text.", None,
                                 {"domain": "official.gov", "title": "Denial", "is_https": True})):
            result = web_evidence.verify_claim_with_evidence("A false claim.")
        self.assertEqual(result["status"], "CONTRADICTED")
        self.assertTrue(result["sources"][0]["is_primary_source"])

    # --- multiple independent supporting sources ---
    def test_multiple_independent_sources_all_supporting(self):
        search_resp = self._mock_search_response([
            {"title": "A", "url": "https://outlet-a.example/story"},
            {"title": "B", "url": "https://outlet-b.example/story"},
        ])
        judge_resp = self._mock_judgment_response([
            {"verdict": "SUPPORTS", "reason": "Confirms it."},
            {"verdict": "SUPPORTS", "reason": "Also confirms it."},
        ])

        def fake_extract(url):
            if "outlet-a" in url:
                return ("Reporters at the scene confirmed the event took place this morning with multiple witnesses.",
                        None, {"domain": "outlet-a.example", "title": "outlet-a.example", "is_https": True})
            return ("An independent investigation separately verified the same event occurred, citing local officials.",
                    None, {"domain": "outlet-b.example", "title": "outlet-b.example", "is_https": True})

        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url", side_effect=fake_extract):
            result = web_evidence.verify_claim_with_evidence("An event occurred.")
        self.assertEqual(result["status"], "SUPPORTED")
        independent_sources = [s for s in result["sources"] if s.get("retrieved") and not s.get("syndicated_of")]
        self.assertEqual(len(independent_sources), 2)

    # --- duplicate/syndicated sources: same wire text via two domains counts as ONE independent source ---
    def test_syndicated_copies_counted_as_one_independent_source(self):
        search_resp = self._mock_search_response([
            {"title": "Wire copy A", "url": "https://outlet-a.example/wire-story"},
            {"title": "Wire copy B", "url": "https://outlet-b.example/wire-story"},
        ])
        # Only ONE judgment call expected, since the second source is
        # deduplicated as syndicated before reaching the judgment stage.
        judge_resp = self._mock_judgment_response([{"verdict": "SUPPORTS", "reason": "Confirms it."}])

        identical_text = "Identical wire service text distributed to many outlets word for word here today."

        def fake_extract(url):
            domain = "outlet-a.example" if "outlet-a" in url else "outlet-b.example"
            return (identical_text, None, {"domain": domain, "title": domain, "is_https": True})

        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url", side_effect=fake_extract):
            result = web_evidence.verify_claim_with_evidence("A wire-reported event occurred.")

        syndicated = [s for s in result["sources"] if s.get("syndicated_of")]
        self.assertEqual(len(syndicated), 1)
        self.assertEqual(result["status"], "SUPPORTED")

    # --- API unavailable (no key configured) ---
    def test_api_unavailable_no_fabricated_evidence(self):
        web_evidence.ANTHROPIC_API_KEY = ""
        result = web_evidence.verify_claim_with_evidence("A claim.")
        self.assertFalse(result["available"])
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertEqual(result["sources"], [])

    # --- malformed LLM response at the search stage ---
    def test_malformed_search_response_does_not_crash(self):
        bad_resp = MagicMock()
        bad_resp.raise_for_status.return_value = None
        bad_resp.json.return_value = {"content": [{"type": "text", "text": "not json at all {{{"}]}
        with patch("detector.verification.web_evidence.requests.post", return_value=bad_resp):
            result = web_evidence.verify_claim_with_evidence("A claim.")
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertEqual(result["sources"], [])

    # --- malformed LLM response at the judgment stage: safe UNRELATED default, never assumes support ---
    def test_malformed_judgment_response_defaults_to_unrelated_not_supported(self):
        search_resp = self._mock_search_response([{"title": "Some page", "url": "https://example.com/x"}])
        bad_judge_resp = MagicMock()
        bad_judge_resp.raise_for_status.return_value = None
        bad_judge_resp.json.return_value = {"content": [{"type": "text", "text": "garbled non-json response"}]}
        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, bad_judge_resp]), \
             patch("detector.extract_text.extract_from_url",
                   return_value=("Some retrieved text.", None,
                                 {"domain": "example.com", "title": "Some page", "is_https": True})):
            result = web_evidence.verify_claim_with_evidence("A claim.")
        # Malformed judgment must NEVER default to SUPPORTED - only INSUFFICIENT/UNVERIFIED.
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))

    # --- conflicting evidence: one source supports, an independent one contradicts -> INSUFFICIENT / MIXED ---
    def test_conflicting_evidence_produces_mixed_status(self):
        search_resp = self._mock_search_response([
            {"title": "Supports", "url": "https://outlet-a.example/x"},
            {"title": "Contradicts", "url": "https://outlet-b.example/y"},
        ])
        judge_resp = self._mock_judgment_response([
            {"verdict": "SUPPORTS", "reason": "Confirms it."},
            {"verdict": "CONTRADICTS", "reason": "Disputes it."},
        ])

        def fake_extract(url):
            if "outlet-a" in url:
                return ("Local officials confirmed the policy change was formally adopted on Tuesday.",
                        None, {"domain": "outlet-a.example", "title": "outlet-a.example", "is_https": True})
            return ("A spokesperson for the department denied any such policy change had been made.",
                    None, {"domain": "outlet-b.example", "title": "outlet-b.example", "is_https": True})

        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url", side_effect=fake_extract):
            result = web_evidence.verify_claim_with_evidence("A disputed claim.")
        self.assertIn(result["status"], ("INSUFFICIENT", "MIXED"))

    # --- insufficient evidence: search finds nothing usable ---
    def test_no_search_results_is_unverified_not_a_guess(self):
        empty_search_resp = self._mock_search_response([])
        with patch("detector.verification.web_evidence.requests.post", return_value=empty_search_resp):
            result = web_evidence.verify_claim_with_evidence("An obscure, hard-to-verify claim.")
        self.assertIn(result["status"], ("INSUFFICIENT", "UNVERIFIED"))
        self.assertEqual(result["sources"], [])

    # --- never fabricate a source title ---
    def test_source_title_comes_from_actual_page_not_search_claim(self):
        search_resp = self._mock_search_response([{"title": "Misleading search title", "url": "https://example.com/x"}])
        judge_resp = self._mock_judgment_response([{"verdict": "SUPPORTS", "reason": "Confirms it."}])
        with patch("detector.verification.web_evidence.requests.post", side_effect=[search_resp, judge_resp]), \
             patch("detector.extract_text.extract_from_url",
                   return_value=("Retrieved text.", None,
                                 {"domain": "example.com", "title": "Actual Page Title From HTML", "is_https": True})):
            result = web_evidence.verify_claim_with_evidence("A claim.")
        self.assertEqual(result["sources"][0]["title"], "Actual Page Title From HTML")

    def test_gather_evidence_empty_claims_list(self):
        result = web_evidence.gather_evidence([])
        self.assertEqual(result["claim_evidence"], [])


class AggregationTests(TestCase):
    """Layers 9-10. These are the tests that prove the core design
    principles hold, using controlled synthetic inputs (not tuned to make
    any particular real article pass)."""

    def _base_kwargs(self, **overrides):
        base = dict(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 90.0,
                                "confidence_display": "90.0%", "confidence_status": "HIGH", "error": None},
            clickbait_result={"score": 5, "classification": "LOW", "signals": []},
            ai_text_result={"label": "Uncertain", "signals": {}},
            source_result={"available": False, "signals": []},
            fact_check_result={"available": False, "claims": []},
            evidence_result={"claim_evidence": []},
        )
        base.update(overrides)
        return base

    def test_unusable_content_forces_uncertain(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            quality_result={"usable": False, "reasons": ["too short"], "warnings": []}
        ))
        self.assertEqual(final["final_assessment"], "UNCERTAIN")

    def test_ambiguous_low_confidence_ml_leads_to_uncertain(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            prediction_result={"prediction": "UNCERTAIN", "raw_model_label": "FAKE", "confidence": 54.0,
                                "confidence_display": "54.0%", "confidence_status": "LOW", "error": None},
        ))
        self.assertEqual(final["final_assessment"], "UNCERTAIN")

    def test_ml_prediction_contributes_when_no_evidence_available(self):
        final = aggregation.aggregate_signals(**self._base_kwargs())
        self.assertEqual(final["final_assessment"], "LIKELY REAL")
        self.assertIn("ml_classification", final["signals_considered"])

    def test_no_evidence_caps_confidence_at_low(self):
        final = aggregation.aggregate_signals(**self._base_kwargs())
        self.assertEqual(final["verification_confidence"], "LOW")

    def test_supported_evidence_allows_higher_confidence(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            evidence_result={"claim_evidence": [
                {"claim": "x", "status": "SUPPORTED", "available": True, "sources": [], "summary": "yes"}
            ]}
        ))
        self.assertIn(final["verification_confidence"], ("MEDIUM", "HIGH"))
        self.assertIn("independent_web_evidence", final["signals_considered"])

    def test_contradicted_evidence_overrides_high_confidence_real_prediction(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            evidence_result={"claim_evidence": [
                {"claim": "x", "status": "CONTRADICTED", "available": True, "sources": [], "summary": "no"}
            ]}
        ))
        self.assertEqual(final["final_assessment"], "LIKELY FAKE")

    def test_clickbait_never_enters_the_score(self):
        # Same inputs except clickbait score - result must be identical,
        # proving clickbait is excluded from aggregation entirely (it is
        # still shown to the user separately, just not scored here).
        low_ctr = aggregation.aggregate_signals(**self._base_kwargs(
            clickbait_result={"score": 5, "classification": "LOW", "signals": []}))
        high_ctr = aggregation.aggregate_signals(**self._base_kwargs(
            clickbait_result={"score": 95, "classification": "HIGH", "signals": []}))
        self.assertEqual(low_ctr["final_assessment"], high_ctr["final_assessment"])
        self.assertEqual(low_ctr["verification_confidence"], high_ctr["verification_confidence"])

    def test_unknown_domain_never_forces_fake(self):
        # source_result with no reference_list_match (unknown domain) must
        # not, by itself, push the assessment toward LIKELY FAKE.
        final_unknown = aggregation.aggregate_signals(**self._base_kwargs(
            source_result={"available": True, "domain": "some-random-blog.example",
                            "reference_list_match": None, "signals": []}
        ))
        self.assertEqual(final_unknown["final_assessment"], "LIKELY REAL")

    def test_factcheck_false_rating_contributes_toward_fake(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            fact_check_result={"available": True, "claims": [{"claim": "x", "status": "False", "source": "FactChecker"}]}
        ))
        self.assertIn("fact_check_database", final["signals_considered"])

    def test_reasoning_is_always_present_for_the_user(self):
        final = aggregation.aggregate_signals(**self._base_kwargs())
        self.assertIsInstance(final["reasoning"], list)
        self.assertGreater(len(final["reasoning"]), 0)


class EndToEndViewTests(TestCase):
    """Full pipeline through Django's test Client - the same code path a
    real browser request takes."""

    def setUp(self):
        self.client = Client()
        self.client.post("/register/", {
            "username": "e2e_tester", "email": "e2e@example.com",
            "password1": "SuperSecret123!", "password2": "SuperSecret123!",
        })
        self.client.post("/login/", {"username": "e2e_tester", "password": "SuperSecret123!"})

    def test_unauth_redirects_to_login(self):
        c = Client()
        r = c.get("/dashboard/")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login/", r.url)

    def test_full_text_analysis_flow_ml_predicts(self):
        text = (
            "SHOCKING: leaked documents reveal a hidden group of world leaders has been "
            "secretly controlling the weather for decades!!! Doctors HATE this one simple "
            "trick that exposes the truth they don't want you to know. Share this before "
            "it gets DELETED by the mainstream media!!!"
        )
        r = self.client.post("/detector/", {"input_type": "text", "news_text": text}, follow=True)
        self.assertEqual(r.status_code, 200)
        analysis = Analysis.objects.filter(user__username="e2e_tester").first()
        self.assertIsNotNone(analysis)
        self.assertIsNotNone(analysis.prediction)  # ML actually produced a prediction
        self.assertIn(analysis.final_assessment, ("LIKELY REAL", "LIKELY FAKE", "UNCERTAIN", "MIXED / MISLEADING"))

        r = self.client.get(f"/analysis/{analysis.id}/")
        self.assertEqual(r.status_code, 200)

        r = self.client.get(f"/analysis/{analysis.id}/report/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Type"], "application/pdf")

    def test_short_content_forces_uncertain_not_a_guess(self):
        # 79 chars: passes extract_text.py's own 30-char minimum (so an
        # Analysis record IS created), but is well below the verification
        # pipeline's 150-char Layer 2 quality threshold - this is the
        # correct boundary to test "reaches the pipeline but gets flagged
        # as insufficient", as opposed to text so short it's rejected as a
        # form error before any Analysis is even created.
        short_text = "The mayor spoke today about new plans for the city budget review process."
        self.assertTrue(30 < len(short_text) < 150)  # sanity-check the fixture itself
        r = self.client.post("/detector/", {"input_type": "text", "news_text": short_text}, follow=True)
        self.assertEqual(r.status_code, 200)
        analysis = Analysis.objects.filter(user__username="e2e_tester").order_by("-id").first()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.final_assessment, "UNCERTAIN")
        self.assertIsNone(analysis.prediction)

    def test_extremely_short_text_rejected_before_pipeline_no_analysis_created(self):
        # Below extract_text.py's own 30-char minimum - should be caught as
        # a form validation error, never even reaching the verification
        # pipeline (no Analysis record created at all).
        before_count = Analysis.objects.filter(user__username="e2e_tester").count()
        r = self.client.post("/detector/", {"input_type": "text", "news_text": "Too short."})
        self.assertEqual(r.status_code, 200)
        after_count = Analysis.objects.filter(user__username="e2e_tester").count()
        self.assertEqual(before_count, after_count)

    def test_ambiguous_opinion_article_leads_to_low_confidence_or_uncertain(self):
        text = (
            "It is time we had an honest conversation about how our leaders have failed to "
            "address the housing crisis. For years, promises have been made and broken, while "
            "ordinary families struggle to find affordable places to live. I believe the "
            "solution starts with holding elected officials accountable at the ballot box."
        )
        r = self.client.post("/detector/", {"input_type": "text", "news_text": text}, follow=True)
        self.assertEqual(r.status_code, 200)
        analysis = Analysis.objects.filter(user__username="e2e_tester").order_by("-id").first()
        self.assertEqual(analysis.verification_confidence, "LOW")

    def test_cross_user_isolation(self):
        text = "The city council voted 6-3 on Tuesday to approve a new budget for road maintenance this year."
        self.client.post("/detector/", {"input_type": "text", "news_text": text}, follow=True)
        analysis = Analysis.objects.filter(user__username="e2e_tester").first()

        other_client = Client()
        other_client.post("/register/", {
            "username": "e2e_other", "email": "e2e_other@example.com",
            "password1": "AnotherPass123!", "password2": "AnotherPass123!",
        })
        other_client.post("/login/", {"username": "e2e_other", "password": "AnotherPass123!"})
        r = other_client.get(f"/analysis/{analysis.id}/")
        self.assertEqual(r.status_code, 404)

    def test_invalid_url_handled_without_crash(self):
        r = self.client.post("/detector/", {"input_type": "url", "news_url": "not-a-url"})
        self.assertEqual(r.status_code, 200)

    def test_ssrf_url_handled_without_crash(self):
        r = self.client.post("/detector/", {"input_type": "url",
                                             "news_url": "http://169.254.169.254/latest/meta-data/"})
        self.assertEqual(r.status_code, 200)

    def test_history_shows_analysis(self):
        text = "The city council voted 6-3 on Tuesday to approve a new road maintenance budget for the year."
        self.client.post("/detector/", {"input_type": "text", "news_text": text}, follow=True)
        r = self.client.get("/history/")
        self.assertEqual(r.status_code, 200)

    def test_chatbot_responds_without_api_key(self):
        text = "The city council voted 6-3 on Tuesday to approve a new road maintenance budget for the year."
        self.client.post("/detector/", {"input_type": "text", "news_text": text}, follow=True)
        analysis = Analysis.objects.filter(user__username="e2e_tester").order_by("-id").first()
        r = self.client.post(
            "/chatbot/ask/",
            data=json.dumps({"question": "What does the confidence score mean?", "analysis_id": analysis.id}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("answer", r.json())

    def test_feedback_and_save_article(self):
        text = "The city council voted 6-3 on Tuesday to approve a new road maintenance budget for the year."
        self.client.post("/detector/", {"input_type": "text", "news_text": text}, follow=True)
        analysis = Analysis.objects.filter(user__username="e2e_tester").order_by("-id").first()

        r = self.client.post(f"/analysis/{analysis.id}/feedback/", {"helpful": "yes", "comment": "Useful"}, follow=True)
        self.assertEqual(r.status_code, 200)

        r = self.client.post(f"/analysis/{analysis.id}/save/", follow=True)
        self.assertEqual(r.status_code, 200)

        r = self.client.get("/reports/")
        self.assertEqual(r.status_code, 200)


class TemporalContextDetectionTests(TestCase):
    def test_detects_historical_news_from_text(self):
        text = "Back in 2016, the presidential campaign reached a critical turning point during the autumn debates."
        result = temporal.detect_temporal_context(text)
        self.assertEqual(result["context"], "historical")
        self.assertEqual(result["detected_year"], 2016)

    def test_detects_historical_news_from_metadata(self):
        text = "Officials today concluded a review of financial agreements signed during the previous administration."
        result = temporal.detect_temporal_context(text, metadata={"published_date": "2016-10-12"})
        self.assertEqual(result["context"], "historical")
        self.assertEqual(result["detected_year"], 2016)

    def test_detects_current_news_from_text(self):
        text = "European regulators announced today in 2024 that new compliance measures would take effect immediately."
        result = temporal.detect_temporal_context(text)
        self.assertEqual(result["context"], "current")
        self.assertEqual(result["detected_year"], 2024)

    def test_detects_current_news_from_metadata(self):
        text = "Investigators delivered an update regarding ongoing infrastructure repair schedules."
        result = temporal.detect_temporal_context(text, metadata={"published_date": "2025-05-14"})
        self.assertEqual(result["context"], "current")
        self.assertEqual(result["detected_year"], 2025)

    def test_detects_unspecified_era_when_no_anchor(self):
        text = "A major scientific conference discussed planetary geology and atmospheric composition."
        result = temporal.detect_temporal_context(text)
        self.assertEqual(result["context"], "unspecified")
        self.assertIsNone(result["detected_year"])

    def test_explicit_user_preference_overrides_detection(self):
        # Text mentions 2016, but user explicitly specified "current"
        text = "Back in 2016, initial discussions began for the high-speed rail network."
        result = temporal.detect_temporal_context(text, explicit_choice="current")
        self.assertEqual(result["context"], "current")
        self.assertIn("explicitly specified", result["reasons"][0])

    def test_explicit_user_preference_historical(self):
        text = "Recent press releases stated that project milestones had been met."
        result = temporal.detect_temporal_context(text, explicit_choice="historical")
        self.assertEqual(result["context"], "historical")


class HistoricalVsCurrentPipelineTests(TestCase):
    def _base_kwargs(self, **overrides):
        base = dict(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 90.0,
                                "confidence_display": "90.0%", "confidence_status": "HIGH", "error": None},
            clickbait_result={"score": 5, "classification": "LOW", "signals": []},
            ai_text_result={"label": "Uncertain", "signals": {}},
            source_result={"available": False, "signals": []},
            fact_check_result={"available": False, "claims": []},
            evidence_result={"claim_evidence": []},
            temporal_info={"context": "unspecified", "label": "Unspecified Era", "detected_year": None},
        )
        base.update(overrides)
        return base

    def test_historical_news_relies_on_ml_when_evidence_absent(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            temporal_info={"context": "historical", "label": "Historical News", "detected_year": 2016}
        ))
        self.assertEqual(final["final_assessment"], "LIKELY REAL")
        self.assertEqual(final["temporal_context"], "historical")
        self.assertEqual(final["verification_confidence"], "LOW")
        self.assertIn("ml_classification", final["signals_considered"])

    def test_current_news_without_evidence_returns_uncertain(self):
        # Current news must NOT rely only on the historical ML model!
        final = aggregation.aggregate_signals(**self._base_kwargs(
            temporal_info={"context": "current", "label": "Current News", "detected_year": 2024},
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 94.0,
                                "confidence_display": "94.0%", "confidence_status": "HIGH", "error": None},
        ))
        self.assertEqual(final["final_assessment"], "UNCERTAIN")
        self.assertEqual(final["temporal_context"], "current")
        self.assertEqual(final["verification_confidence"], "LOW")
        self.assertTrue(any("cannot be reliably verified using only a historical ML classifier" in r for r in final["reasoning"]))

    def test_current_news_with_supporting_evidence_returns_likely_real(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            temporal_info={"context": "current", "label": "Current News", "detected_year": 2024},
            evidence_result={"claim_evidence": [
                {"claim": "Event happened", "status": "SUPPORTED", "available": True, "sources": [], "summary": "Confirmed"}
            ]},
        ))
        self.assertEqual(final["final_assessment"], "LIKELY REAL")
        self.assertIn(final["verification_confidence"], ("MEDIUM", "HIGH"))

    def test_current_news_with_contradicting_evidence_returns_likely_fake(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            temporal_info={"context": "current", "label": "Current News", "detected_year": 2024},
            evidence_result={"claim_evidence": [
                {"claim": "Event happened", "status": "CONTRADICTED", "available": True, "sources": [], "summary": "Debunked"}
            ]},
        ))
        self.assertEqual(final["final_assessment"], "LIKELY FAKE")
        self.assertEqual(final["verification_confidence"], "MEDIUM")

    def test_bidirectional_override_ml_fake_to_real(self):
        # Model thinks style is FAKE, but independent evidence confirms it is REAL
        final = aggregation.aggregate_signals(**self._base_kwargs(
            temporal_info={"context": "historical", "label": "Historical News", "detected_year": 2016},
            prediction_result={"prediction": "FAKE", "raw_model_label": "FAKE", "confidence": 91.0,
                                "confidence_display": "91.0%", "confidence_status": "HIGH", "error": None},
            evidence_result={"claim_evidence": [
                {"claim": "Headline claim", "status": "SUPPORTED", "available": True, "sources": [], "summary": "Verified by two wires"}
            ]},
        ))
        self.assertEqual(final["final_assessment"], "LIKELY REAL")
        self.assertTrue(any("overrides the historical ML model" in r for r in final["reasoning"]))

    def test_bidirectional_override_ml_real_to_fake(self):
        # Model thinks style is REAL, but independent evidence confirms it is FAKE
        final = aggregation.aggregate_signals(**self._base_kwargs(
            temporal_info={"context": "historical", "label": "Historical News", "detected_year": 2016},
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 93.0,
                                "confidence_display": "93.0%", "confidence_status": "HIGH", "error": None},
            evidence_result={"claim_evidence": [
                {"claim": "Headline claim", "status": "CONTRADICTED", "available": True, "sources": [], "summary": "Proven false"}
            ]},
        ))
        self.assertEqual(final["final_assessment"], "LIKELY FAKE")
        self.assertTrue(any("CONTRADICTS at least one central claim" in r for r in final["reasoning"]))

    def test_distinct_confidence_metrics_in_output(self):
        final = aggregation.aggregate_signals(**self._base_kwargs(
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 88.5,
                                "confidence_display": "88.5%", "confidence_status": "HIGH", "error": None},
        ))
        self.assertIn("ml_confidence", final)
        self.assertIn("verification_confidence", final)
        self.assertEqual(final["ml_confidence"], 88.5)
        self.assertEqual(final["verification_confidence"], "LOW")

    def test_four_pillars_present_in_output(self):
        final = aggregation.aggregate_signals(**self._base_kwargs())
        # 1. Final assessment
        self.assertIn("final_assessment", final)
        # 2. Verification confidence
        self.assertIn("verification_confidence", final)
        # 3. ML prediction
        self.assertIn("ml_prediction", final)
        # 4. Evidence found
        self.assertIn("evidence_found", final)
        self.assertIsInstance(final["evidence_found"], dict)


class SystemEvaluationCommandTests(TestCase):
    def test_evaluation_command_runs_successfully(self):
        out = io.StringIO()
        call_command("evaluate_system", limit=2, stdout=out)
        output_str = out.getvalue()
        self.assertIn("TruthLens AI - Pipeline Evaluation", output_str)
        self.assertIn("PERFORMANCE METRICS", output_str)
        self.assertIn("CONFUSION MATRIX", output_str)
        self.assertIn("BREAKDOWN BY TEMPORAL ERA", output_str)

    def test_evaluation_command_handles_missing_file_gracefully(self):
        err = io.StringIO()
        call_command("evaluate_system", dataset="non_existent_dataset.json", stderr=err)
        err_str = err.getvalue()
        self.assertIn("not found", err_str)


class EndToEndTemporalViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.client.post("/register/", {
            "username": "temporal_tester", "email": "temporal@example.com",
            "password1": "Password123!", "password2": "Password123!",
        })
        self.client.post("/login/", {"username": "temporal_tester", "password": "Password123!"})

    def test_form_temporal_preference_saved_as_current(self):
        text = ("The city council voted 6-3 on Tuesday to approve a new budget for road maintenance in the coming fiscal year. "
                "Officials confirmed that funding for bridges and resurfacing would be disbursed in the subsequent financial quarter.")
        r = self.client.post("/detector/", {
            "input_type": "text",
            "news_text": text,
            "temporal_preference": "current",
        }, follow=True)
        self.assertEqual(r.status_code, 200)
        analysis = Analysis.objects.filter(user__username="temporal_tester").order_by("-id").first()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.temporal_context, "current")

    def test_form_temporal_preference_saved_as_historical(self):
        text = ("The city council voted 6-3 on Tuesday to approve a new budget for road maintenance in the coming fiscal year. "
                "Officials confirmed that funding for bridges and resurfacing would be disbursed in the subsequent financial quarter.")
        r = self.client.post("/detector/", {
            "input_type": "text",
            "news_text": text,
            "temporal_preference": "historical",
        }, follow=True)
        self.assertEqual(r.status_code, 200)
        analysis = Analysis.objects.filter(user__username="temporal_tester").order_by("-id").first()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.temporal_context, "historical")

    def test_result_page_renders_four_pillars_and_era_badge(self):
        text = ("The city council voted 6-3 on Tuesday to approve a new budget for road maintenance in the coming fiscal year. "
                "Officials confirmed that funding for bridges and resurfacing would be disbursed in the subsequent financial quarter.")
        r = self.client.post("/detector/", {
            "input_type": "text",
            "news_text": text,
            "temporal_preference": "historical",
        }, follow=True)
        self.assertEqual(r.status_code, 200)
        content = r.content.decode("utf-8")
        self.assertIn("Historical", content)
        self.assertIn("Verification Confidence", content)
        self.assertIn("ML Model Prediction", content)
        self.assertIn("Evidence Found", content)


class MultiApiIntegrationTests(TestCase):
    """Verifies that all external APIs (Gemini, Fact Check, Tavily, Sarvam, Anthropic)
    are securely integrated, degrade gracefully when missing or failing, and never
    leak secrets or crash the system."""

    def test_gemini_api_call_and_degradation(self):
        from detector import ai_services
        # When key is unset or empty
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            res = ai_services.call_gemini("Test prompt")
            self.assertFalse(res["success"])
            self.assertIn("not configured", res["error"])

        # When mock response succeeds
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": "Mocked Gemini summary."}]}
            }]
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-test-key"}):
            with patch("detector.ai_services.requests.post", return_value=mock_resp):
                res = ai_services.call_gemini("Test prompt")
                self.assertTrue(res["success"])
                self.assertEqual(res["text"], "Mocked Gemini summary.")

    def test_tavily_search_and_degradation(self):
        from detector import ai_services
        # When key is unset
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            results = ai_services.search_tavily("NASA discovery")
            self.assertEqual(results, [])

        # When search returns results
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"title": "NASA Report", "url": "https://nasa.gov/news/1"}
            ]
        }
        with patch.dict(os.environ, {"TAVILY_API_KEY": "dummy-test-key"}):
            with patch("detector.ai_services.requests.post", return_value=mock_resp):
                results = ai_services.search_tavily("NASA discovery")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["url"], "https://nasa.gov/news/1")

    def test_sarvam_indic_translation_and_degradation(self):
        from detector import translation, ai_services
        # Detection
        self.assertFalse(translation.detect_indic_script("This is purely in English language."))
        self.assertTrue(translation.detect_indic_script("यह हिंदी में लिखा गया एक समाचार है।"))

        # Degradation without key
        with patch.dict(os.environ, {"SARVAM_API_KEY": "", "SARVAM_AI_API_KEY": ""}):
            res = translation.translate_if_indic("यह हिंदी समाचार है।")
            self.assertFalse(res["translated"])
            self.assertIn("SARVAM_API_KEY", res["message"])

        # Mocked translation success
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"translated_text": "This is Hindi news."}
        with patch.dict(os.environ, {"SARVAM_API_KEY": "dummy-test-key"}):
            with patch("detector.ai_services.requests.post", return_value=mock_resp):
                res = translation.translate_if_indic("यह हिंदी समाचार है।")
                self.assertTrue(res["translated"])
                self.assertEqual(res["processed_text"], "This is Hindi news.")

    def test_check_all_api_status_does_not_reveal_secrets(self):
        from detector import ai_services
        status = ai_services.check_all_api_status()
        self.assertIn("Google Gemini API", status)
        self.assertIn("Tavily Search API", status)
        self.assertIn("Google Fact Check Tools API", status)
        self.assertIn("Sarvam AI API", status)
        self.assertIn("Anthropic Claude API", status)

        # Inspect all status fields to verify NO secrets are ever exposed
        for api_name, info in status.items():
            self.assertIn("purpose", info)
            self.assertIn("connected", info)
            self.assertIn("configured", info)
            self.assertIn("notes", info)
            # Ensure no key values leaked into notes or purpose
            notes_lower = str(info["notes"]).lower()
            self.assertNotIn("sk-", notes_lower)
            self.assertNotIn("aiza", notes_lower)
            self.assertNotIn("tvly-", notes_lower)


class EvidenceVerificationFixTests(TestCase):
    """
    Validation Suite for Critical Evidence Verification Fix:
    Covers:
      - Test A: Known genuine current news
      - Test B: Known fake news
      - Test C: Ambiguous news
      - Test D: News where ML and external evidence disagree (including 4 strong support vs 3 weak contra -> LIKELY REAL)
      - Test E: News where sources genuinely conflict -> UNCERTAIN
      - Rule 3: Direct contradiction required ('could not be independently verified' is INSUFFICIENT, not CONTRADICTS)
      - Rule 4: Fact-checker claim verified transparency
      - Rule 6: Primary source override
    """

    def test_a_known_genuine_current_news(self):
        """Test A: Known genuine current news with strong corroboration -> LIKELY REAL."""
        claims_ev = [{
            "claim": "Government health agency announced new health safety guidelines.",
            "status": "SUPPORTED",
            "available": True,
            "sources": [
                {
                    "url": "https://who.int/news/item/guidelines",
                    "domain": "who.int",
                    "publisher": "World Health Organization",
                    "retrieved": True,
                    "is_primary_source": True,
                    "source_reliability": "VERY_HIGH",
                    "source_reliability_weight": 2.0,
                    "content_verdict": "SUPPORTS",
                    "verdict": "SUPPORTS",
                    "directness": "DIRECT",
                    "strength_score": 2.0,
                    "strength_label": "STRONG",
                    "reason": "Official institutional announcement confirms guidelines.",
                },
                {
                    "url": "https://reuters.com/world/guidelines",
                    "domain": "reuters.com",
                    "publisher": "Reuters",
                    "retrieved": True,
                    "is_primary_source": False,
                    "source_reliability": "HIGH",
                    "source_reliability_weight": 1.2,
                    "content_verdict": "SUPPORTS",
                    "verdict": "SUPPORTS",
                    "directness": "DIRECT",
                    "strength_score": 1.2,
                    "strength_label": "STRONG",
                    "reason": "Independent wire report corroborates health safety release.",
                }
            ],
            "summary": "Corroborated by primary and major wire sources."
        }]

        res = aggregation.aggregate_signals(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 75.0, "confidence_display": "75.0%", "error": None},
            clickbait_result={"score": 10, "classification": "LOW", "signals": []},
            ai_text_result={"label": "Human", "signals": {}},
            source_result={"available": True, "domain": "reuters.com", "reference_list_match": "established_news_organization", "signals": []},
            fact_check_result={"available": False, "claims": []},
            evidence_result={"claim_evidence": claims_ev},
            temporal_info={"context": "current", "label": "October 2026", "reasons": []}
        )

        self.assertEqual(res["final_assessment"], "LIKELY REAL")
        self.assertEqual(res["verification_confidence"], "HIGH")
        self.assertTrue(any("Official primary source" in r for r in res["reasoning"]))

    def test_b_known_fake_news(self):
        """Test B: Known fake news directly debunked by reliable fact-checking source -> LIKELY FAKE."""
        fact_check = {
            "available": True,
            "claims": [{
                "claim": "Viral scheme offers free government cash to all citizens.",
                "status": "False",
                "source": "PolitiFact",
                "url": "https://politifact.com/factchecks/scheme-false",
                "alignment": "CONTRADICTS"
            }]
        }
        claims_ev = [{
            "claim": "Viral scheme offers free government cash to all citizens.",
            "status": "CONTRADICTED",
            "available": True,
            "sources": [
                {
                    "url": "https://politifact.com/factchecks/scheme-false",
                    "domain": "politifact.com",
                    "publisher": "PolitiFact",
                    "retrieved": True,
                    "is_primary_source": False,
                    "source_reliability": "HIGH",
                    "source_reliability_weight": 1.2,
                    "content_verdict": "CONTRADICTS",
                    "verdict": "CONTRADICTED",
                    "directness": "DIRECT",
                    "strength_score": 1.2,
                    "strength_label": "STRONG",
                    "reason": "Debunked as an online phishing scheme.",
                }
            ],
            "summary": "Debunked by verified fact-checkers."
        }]

        res = aggregation.aggregate_signals(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 65.0, "confidence_display": "65.0%", "error": None},
            clickbait_result={"score": 40, "classification": "MEDIUM", "signals": []},
            ai_text_result={"label": "Human", "signals": {}},
            source_result={"available": False, "signals": []},
            fact_check_result=fact_check,
            evidence_result={"claim_evidence": claims_ev},
            temporal_info={"context": "current", "label": "October 2026", "reasons": []}
        )

        self.assertEqual(res["final_assessment"], "LIKELY FAKE")
        self.assertEqual(res["verification_confidence"], "HIGH")
        self.assertTrue(any("Authoritative fact-checking databases record claims in this article as false" in r for r in res["reasoning"]))

    def test_c_ambiguous_news(self):
        """Test C: Ambiguous news with insufficient evidence -> UNCERTAIN, LOW confidence."""
        claims_ev = [{
            "claim": "A local developer announced an unverified concept design.",
            "status": "UNVERIFIED",
            "available": True,
            "sources": [],
            "summary": "No matching reporting found."
        }]

        res = aggregation.aggregate_signals(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "UNCERTAIN", "raw_model_label": "FAKE", "confidence": 52.0, "confidence_display": "52.0%", "error": None},
            clickbait_result={"score": 15, "classification": "LOW", "signals": []},
            ai_text_result={"label": "Human", "signals": {}},
            source_result={"available": False, "signals": []},
            fact_check_result={"available": False, "claims": []},
            evidence_result={"claim_evidence": claims_ev},
            temporal_info={"context": "current", "label": "October 2026", "reasons": []}
        )

        self.assertEqual(res["final_assessment"], "UNCERTAIN")
        self.assertEqual(res["verification_confidence"], "LOW")

    def test_d_ml_and_external_evidence_disagree(self):
        """
        Test D & Rule 9: ML style says FAKE (98% confidence), but 4 strong supporting sources
        corroborate event against 3 weak/indirect blogs -> LIKELY REAL.
        Must NEVER determine result using raw count (4 vs 3 -> FAKE).
        """
        supporting_sources = [
            {
                "url": f"https://outlet-{i}.com/report",
                "domain": f"outlet-{i}.com",
                "publisher": f"Major Outlet {i}",
                "retrieved": True,
                "is_primary_source": (i == 1),
                "source_reliability": "VERY_HIGH" if i == 1 else "HIGH",
                "source_reliability_weight": 2.0 if i == 1 else 1.2,
                "content_verdict": "SUPPORTS",
                "verdict": "SUPPORTS",
                "directness": "DIRECT",
                "strength_score": 2.0 if i == 1 else 1.2,
                "strength_label": "STRONG",
                "reason": "Directly corroborates that the event took place.",
            }
            for i in range(1, 5)  # 4 strong sources
        ]

        contradicting_weak_sources = [
            {
                "url": f"https://random-blog-{j}.com/post",
                "domain": f"random-blog-{j}.com",
                "publisher": f"Random Blog {j}",
                "retrieved": True,
                "is_primary_source": False,
                "source_reliability": "LOW",
                "source_reliability_weight": 0.3,
                "content_verdict": "CONTRADICTS",
                "verdict": "CONTRADICTS",
                "directness": "INDIRECT",
                "strength_score": 0.2,
                "strength_label": "WEAK",
                "reason": "Indirect commentary expressing skepticism.",
            }
            for j in range(1, 4)  # 3 weak sources
        ]

        claims_ev = [{
            "claim": "Municipal authority initiated 36-hour planned maintenance.",
            "status": "SUPPORTED",
            "available": True,
            "sources": supporting_sources + contradicting_weak_sources,
            "summary": "Corroborated by 4 reliable sources outweighing weak blogs."
        }]

        res = aggregation.aggregate_signals(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "LIKELY FAKE", "raw_model_label": "FAKE", "confidence": 98.0, "confidence_display": "98.0%", "error": None},
            clickbait_result={"score": 20, "classification": "LOW", "signals": []},
            ai_text_result={"label": "Human", "signals": {}},
            source_result={"available": False, "signals": []},
            fact_check_result={"available": False, "claims": []},
            evidence_result={"claim_evidence": claims_ev},
            temporal_info={"context": "current", "label": "Present Event", "reasons": []}
        )

        # Under the old raw count bug, any contra >= sup or contra > 0 would fail.
        # Now, strength-weighted evaluation must classify this as LIKELY REAL!
        self.assertEqual(res["final_assessment"], "LIKELY REAL")
        self.assertIn("Independent verified reporting SUPPORTS", res["reasoning"][1])
        self.assertTrue(any("weak or indirect contradictory source(s) were evaluated and superseded" in r for r in res["reasoning"]))

    def test_e_sources_genuinely_conflict(self):
        """
        Test E & Rule 7: Genuine conflict between strong sources (1 strong support + 1 strong contra)
        -> Must NOT choose a side; reports UNCERTAIN.
        """
        claims_ev = [{
            "claim": "Country X formally signed ceasefire accord with Country Y.",
            "status": "MIXED",
            "available": True,
            "sources": [
                {
                    "url": "https://apnews.com/article/ceasefire-signed",
                    "domain": "apnews.com",
                    "publisher": "Associated Press",
                    "retrieved": True,
                    "is_primary_source": False,
                    "source_reliability": "HIGH",
                    "source_reliability_weight": 1.2,
                    "content_verdict": "SUPPORTS",
                    "verdict": "SUPPORTS",
                    "directness": "DIRECT",
                    "strength_score": 1.2,
                    "strength_label": "STRONG",
                    "reason": "AP reports official signing occurred this morning.",
                },
                {
                    "url": "https://reuters.com/world/ceasefire-denied",
                    "domain": "reuters.com",
                    "publisher": "Reuters",
                    "retrieved": True,
                    "is_primary_source": False,
                    "source_reliability": "HIGH",
                    "source_reliability_weight": 1.2,
                    "content_verdict": "CONTRADICTS",
                    "verdict": "CONTRADICTS",
                    "directness": "DIRECT",
                    "strength_score": 1.2,
                    "strength_label": "STRONG",
                    "reason": "Reuters quotes spokesperson directly denying accord was signed.",
                }
            ],
            "summary": "Conflicting reporting between major wire services."
        }]

        res = aggregation.aggregate_signals(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "REAL", "raw_model_label": "REAL", "confidence": 60.0, "confidence_display": "60.0%", "error": None},
            clickbait_result={"score": 10, "classification": "LOW", "signals": []},
            ai_text_result={"label": "Human", "signals": {}},
            source_result={"available": False, "signals": []},
            fact_check_result={"available": False, "claims": []},
            evidence_result={"claim_evidence": claims_ev},
            temporal_info={"context": "current", "label": "Breaking News", "reasons": []}
        )

        self.assertEqual(res["final_assessment"], "UNCERTAIN")
        self.assertTrue(any("genuinely conflict" in r for r in res["reasoning"]))

    def test_rule_3_non_confirmation_is_not_contradiction(self):
        """Rule 3: Stating 'could not be independently verified' must be INSUFFICIENT, not CONTRADICTED."""
        sources = [{
            "url": "https://outlet.example/story",
            "text_excerpt": "Officials announced a new agreement. Reuters could not independently verify the casualties reported.",
        }]
        res = web_evidence._judge_sources_heuristic("The casualties reached fifty.", sources)
        self.assertEqual(len(res), 1)
        self.assertNotEqual(res[0]["verdict"], "CONTRADICTS")
        self.assertEqual(res[0]["verdict"], "INSUFFICIENT")

    def test_rule_6_primary_source_override(self):
        """Rule 6: Primary official source directly confirming overrides weak doubt."""
        claims_ev = [{
            "claim": "Space agency successfully landed lunar probe.",
            "status": "SUPPORTED",
            "available": True,
            "sources": [
                {
                    "url": "https://isro.gov.in/press/lunar-landing",
                    "domain": "isro.gov.in",
                    "publisher": "ISRO Official",
                    "retrieved": True,
                    "is_primary_source": True,
                    "source_reliability": "VERY_HIGH",
                    "source_reliability_weight": 2.0,
                    "content_verdict": "SUPPORTS",
                    "verdict": "SUPPORTS",
                    "directness": "DIRECT",
                    "strength_score": 2.0,
                    "strength_label": "STRONG",
                    "reason": "Direct telemetric confirmation of successful landing.",
                }
            ],
            "summary": "Confirmed by primary agency."
        }]

        res = aggregation.aggregate_signals(
            quality_result={"usable": True, "reasons": [], "warnings": []},
            prediction_result={"prediction": "UNCERTAIN", "raw_model_label": "FAKE", "confidence": 55.0, "confidence_display": "55.0%", "error": None},
            clickbait_result={"score": 5, "classification": "LOW", "signals": []},
            ai_text_result={"label": "Human", "signals": {}},
            source_result={"available": False, "signals": []},
            fact_check_result={"available": False, "claims": []},
            evidence_result={"claim_evidence": claims_ev},
            temporal_info={"context": "current", "label": "Today", "reasons": []}
        )

        self.assertEqual(res["final_assessment"], "LIKELY REAL")
        self.assertEqual(res["verification_confidence"], "HIGH")


class ClaimEvidenceCountConsistencyTests(TestCase):
    """
    Validates the fix for verification result count consistency:
    - Exactly ONE authoritative claims verification structure.
    - Canonical claim status normalization: SUPPORTED, CONTRADICTED, INSUFFICIENT.
    - Separation of claim counts (claims analyzed) and source counts (URLs evaluated).
    - Summary card counts dynamically match detailed claim cards.
    - Consistency with Problematic Scenario (2 support, 2 contradict, 1 insufficient).
    - Consistency with ISRO PSLV-C62 example.
    """

    def test_canonical_status_normalization(self):
        from .verification.claims import normalize_claim_status

        # Variations of SUPPORTED
        self.assertEqual(normalize_claim_status("supported"), "SUPPORTED")
        self.assertEqual(normalize_claim_status("support"), "SUPPORTED")
        self.assertEqual(normalize_claim_status("SUPPORTED"), "SUPPORTED")
        self.assertEqual(normalize_claim_status("true"), "SUPPORTED")
        self.assertEqual(normalize_claim_status("corroborated"), "SUPPORTED")

        # Variations of CONTRADICTED
        self.assertEqual(normalize_claim_status("contradicted"), "CONTRADICTED")
        self.assertEqual(normalize_claim_status("contradiction"), "CONTRADICTED")
        self.assertEqual(normalize_claim_status("CONTRADICTED"), "CONTRADICTED")
        self.assertEqual(normalize_claim_status("false"), "CONTRADICTED")
        self.assertEqual(normalize_claim_status("refuted"), "CONTRADICTED")

        # Variations of INSUFFICIENT
        self.assertEqual(normalize_claim_status("insufficient"), "INSUFFICIENT")
        self.assertEqual(normalize_claim_status("uncertain"), "INSUFFICIENT")
        self.assertEqual(normalize_claim_status("INSUFFICIENT"), "INSUFFICIENT")
        self.assertEqual(normalize_claim_status("unverified"), "INSUFFICIENT")
        self.assertEqual(normalize_claim_status("mixed"), "INSUFFICIENT")
        self.assertEqual(normalize_claim_status(None), "INSUFFICIENT")
        self.assertEqual(normalize_claim_status(""), "INSUFFICIENT")

    def test_problematic_scenario_counts_consistency(self):
        """
        Scenario 11:
        Claim 1 -> SUPPORTED
        Claim 2 -> SUPPORTED
        Claim 3 -> CONTRADICTED
        Claim 4 -> CONTRADICTED
        Claim 5 -> INSUFFICIENT

        Expected Summary:
        2 Supporting
        2 Contradicting
        1 Insufficient
        """
        from .verification.claims import build_authoritative_claims_structure

        claims_list = [
            "Claim 1 text",
            "Claim 2 text",
            "Claim 3 text",
            "Claim 4 text",
            "Claim 5 text",
        ]
        evidence_data = {
            "claim_evidence": [
                {"claim": "Claim 1 text", "status": "SUPPORTED", "sources": [{"url": "http://a.com", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 2 text", "status": "SUPPORTED", "sources": [{"url": "http://b.com", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 3 text", "status": "CONTRADICTED", "sources": []},
                {"claim": "Claim 4 text", "status": "CONTRADICTED", "sources": []},
                {"claim": "Claim 5 text", "status": "INSUFFICIENT", "sources": []},
            ]
        }

        res = build_authoritative_claims_structure(claims_list=claims_list, evidence_data=evidence_data)

        # Claim counts must match exactly
        self.assertEqual(res["claim_counts"]["supported"], 2)
        self.assertEqual(res["claim_counts"]["contradicted"], 2)
        self.assertEqual(res["claim_counts"]["insufficient"], 1)
        self.assertEqual(res["claim_counts"]["total"], 5)

        # Summary structure must match claim counts exactly
        self.assertEqual(res["evidence_summary"]["supporting"], 2)
        self.assertEqual(res["evidence_summary"]["contradicting"], 2)
        self.assertEqual(res["evidence_summary"]["insufficient"], 1)

        # Detailed claims must match exactly
        claims = res["claims"]
        self.assertEqual(len(claims), 5)
        self.assertEqual(claims[0]["status"], "SUPPORTED")
        self.assertEqual(claims[0]["status_icon"], "✓")
        self.assertEqual(claims[1]["status"], "SUPPORTED")
        self.assertEqual(claims[1]["status_icon"], "✓")
        self.assertEqual(claims[2]["status"], "CONTRADICTED")
        self.assertEqual(claims[2]["status_icon"], "✕")
        self.assertEqual(claims[3]["status"], "CONTRADICTED")
        self.assertEqual(claims[3]["status_icon"], "✕")
        self.assertEqual(claims[4]["status"], "INSUFFICIENT")
        self.assertEqual(claims[4]["status_icon"], "?")

    def test_separation_of_claim_count_and_source_count(self):
        """
        Scenario 4:
        Claim 1 supported by Source A, Source B, Source C.
        Must report:
        1 Supporting Claim
        3 Supporting Sources
        NOT 3 Supporting Claims!
        """
        from .verification.claims import build_authoritative_claims_structure

        claims_list = ["Claim 1 text"]
        evidence_data = {
            "claim_evidence": [
                {
                    "claim": "Claim 1 text",
                    "status": "SUPPORTED",
                    "sources": [
                        {"url": "http://a.com/1", "content_verdict": "SUPPORTS", "publisher": "Wire A"},
                        {"url": "http://b.com/2", "content_verdict": "SUPPORTS", "publisher": "Wire B"},
                        {"url": "http://c.com/3", "content_verdict": "SUPPORTS", "publisher": "Wire C"},
                    ],
                }
            ]
        }

        res = build_authoritative_claims_structure(claims_list=claims_list, evidence_data=evidence_data)

        self.assertEqual(res["claim_counts"]["supported"], 1)
        self.assertEqual(res["evidence_summary"]["supporting"], 1)
        self.assertEqual(res["source_counts"]["total"], 3)
        self.assertEqual(res["source_counts"]["supporting"], 3)

    def test_isro_pslv_example_consistency(self):
        """
        Scenario 12: Test with ISRO PSLV-C62 example.
        2 claims SUPPORTED, 2 claims CONTRADICTED.
        Summary must report:
        2 Supporting Claims
        2 Contradicting Claims
        0 Insufficient Claims
        """
        from .models import Analysis
        from .views import _build_result_context_helpers

        # Find or create ISRO analysis
        isro_a = Analysis.objects.filter(extracted_text__icontains="PSLV").first()
        if not isro_a:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            user = User.objects.first() or User.objects.create_user(username="isro_tester")
            isro_a = Analysis.objects.create(
                user=user,
                input_type="text",
                extracted_text="ISRO launched the PSLV-C62 mission.",
                claims=[
                    "ISRO launched the PSLV-C62 mission from the Satish Dhawan Space Centre in Sriharikota.",
                    "The PSLV-C62 mission carried multiple satellites into orbit.",
                    "ISRO officials confirmed the successful launch of the PSLV-C62 mission.",
                    "ISRO officials confirmed the successful deployment of the PSLV-C62 mission payloads.",
                ],
                evidence={
                    "claim_evidence": [
                        {"claim": "ISRO launched the PSLV-C62 mission from the Satish Dhawan Space Centre in Sriharikota.", "status": "SUPPORTED", "sources": []},
                        {"claim": "The PSLV-C62 mission carried multiple satellites into orbit.", "status": "SUPPORTED", "sources": []},
                        {"claim": "ISRO officials confirmed the successful launch of the PSLV-C62 mission.", "status": "CONTRADICTED", "sources": []},
                        {"claim": "ISRO officials confirmed the successful deployment of the PSLV-C62 mission payloads.", "status": "CONTRADICTED", "sources": []},
                    ]
                },
                final_assessment="LIKELY REAL",
            )

        helpers = _build_result_context_helpers(isro_a)
        ev_summary = helpers["evidence_summary"]
        claim_counts = helpers["claim_counts"]

        self.assertEqual(claim_counts["supported"], 2)
        self.assertEqual(claim_counts["contradicted"], 2)
        self.assertEqual(claim_counts["insufficient"], 0)

        # Summary must NOT show 0 Contradicting when 2 claims are contradicted
        self.assertEqual(ev_summary["supporting"], 2)
        self.assertEqual(ev_summary["contradicting"], 2)
        self.assertEqual(ev_summary["insufficient"], 0)

        # Final assessment must be MIXED / MISLEADING for 2 supported + 2 contradicted
        self.assertEqual(helpers["canonical_result"]["final_assessment"]["verdict"], "MIXED / MISLEADING")

    def test_regression_test_a_all_real(self):
        """Test A — All real: 4 supported, 0 contradicted, 0 insufficient -> LIKELY REAL"""
        from .verification.canonical import build_canonical_verification_result

        claims_list = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
        evidence_data = {
            "claim_evidence": [
                {"claim": "Claim 1", "status": "SUPPORTED", "sources": [{"url": "http://s1.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 2", "status": "SUPPORTED", "sources": [{"url": "http://s2.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 3", "status": "SUPPORTED", "sources": [{"url": "http://s3.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 4", "status": "SUPPORTED", "sources": [{"url": "http://s4.org", "content_verdict": "SUPPORTS"}]},
            ]
        }
        res = build_canonical_verification_result(claims_list=claims_list, evidence_data=evidence_data)

        self.assertEqual(res["claim_counts"]["supported"], 4)
        self.assertEqual(res["claim_counts"]["contradicted"], 0)
        self.assertEqual(res["claim_counts"]["insufficient"], 0)
        self.assertEqual(res["final_assessment"]["verdict"], "LIKELY REAL")

    def test_regression_test_b_all_fake(self):
        """Test B — All fake: 0 supported, 4 contradicted, 0 insufficient -> LIKELY FAKE"""
        from .verification.canonical import build_canonical_verification_result

        claims_list = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
        evidence_data = {
            "claim_evidence": [
                {"claim": "Claim 1", "status": "CONTRADICTED", "sources": [{"url": "http://s1.org", "content_verdict": "CONTRADICTS"}]},
                {"claim": "Claim 2", "status": "CONTRADICTED", "sources": [{"url": "http://s2.org", "content_verdict": "CONTRADICTS"}]},
                {"claim": "Claim 3", "status": "CONTRADICTED", "sources": [{"url": "http://s3.org", "content_verdict": "CONTRADICTS"}]},
                {"claim": "Claim 4", "status": "CONTRADICTED", "sources": [{"url": "http://s4.org", "content_verdict": "CONTRADICTS"}]},
            ]
        }
        res = build_canonical_verification_result(claims_list=claims_list, evidence_data=evidence_data)

        self.assertEqual(res["claim_counts"]["supported"], 0)
        self.assertEqual(res["claim_counts"]["contradicted"], 4)
        self.assertEqual(res["claim_counts"]["insufficient"], 0)
        self.assertEqual(res["final_assessment"]["verdict"], "LIKELY FAKE")

    def test_regression_test_c_mixed(self):
        """Test C — Mixed: 2 supported, 2 contradicted, 0 insufficient -> MIXED / MISLEADING"""
        from .verification.canonical import build_canonical_verification_result

        claims_list = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
        evidence_data = {
            "claim_evidence": [
                {"claim": "Claim 1", "status": "SUPPORTED", "sources": [{"url": "http://s1.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 2", "status": "SUPPORTED", "sources": [{"url": "http://s2.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 3", "status": "CONTRADICTED", "sources": [{"url": "http://s3.org", "content_verdict": "CONTRADICTS"}]},
                {"claim": "Claim 4", "status": "CONTRADICTED", "sources": [{"url": "http://s4.org", "content_verdict": "CONTRADICTS"}]},
            ]
        }
        res = build_canonical_verification_result(claims_list=claims_list, evidence_data=evidence_data)

        self.assertEqual(res["claim_counts"]["supported"], 2)
        self.assertEqual(res["claim_counts"]["contradicted"], 2)
        self.assertEqual(res["claim_counts"]["insufficient"], 0)
        self.assertEqual(res["final_assessment"]["verdict"], "MIXED / MISLEADING")

    def test_regression_test_d_uncertain(self):
        """Test D — Uncertain: 0 supported, 0 contradicted, 4 insufficient -> UNCERTAIN"""
        from .verification.canonical import build_canonical_verification_result

        claims_list = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
        evidence_data = {
            "claim_evidence": [
                {"claim": "Claim 1", "status": "INSUFFICIENT", "sources": []},
                {"claim": "Claim 2", "status": "INSUFFICIENT", "sources": []},
                {"claim": "Claim 3", "status": "INSUFFICIENT", "sources": []},
                {"claim": "Claim 4", "status": "INSUFFICIENT", "sources": []},
            ]
        }
        res = build_canonical_verification_result(claims_list=claims_list, evidence_data=evidence_data)

        self.assertEqual(res["claim_counts"]["supported"], 0)
        self.assertEqual(res["claim_counts"]["contradicted"], 0)
        self.assertEqual(res["claim_counts"]["insufficient"], 4)
        self.assertEqual(res["final_assessment"]["verdict"], "UNCERTAIN")

    def test_regression_test_e_ml_disagreement(self):
        """Test E — ML disagreement: ML = FAKE 99.5%, Evidence = strongly supported -> Final Assessment = LIKELY REAL, ML Signal = LEANS FAKE"""
        from .verification.canonical import build_canonical_verification_result

        claims_list = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
        evidence_data = {
            "claim_evidence": [
                {"claim": "Claim 1", "status": "SUPPORTED", "sources": [{"url": "http://s1.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 2", "status": "SUPPORTED", "sources": [{"url": "http://s2.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 3", "status": "SUPPORTED", "sources": [{"url": "http://s3.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 4", "status": "SUPPORTED", "sources": [{"url": "http://s4.org", "content_verdict": "SUPPORTS"}]},
            ]
        }
        prediction_result = {
            "prediction": "FAKE",
            "confidence": 99.5,
            "confidence_status": "HIGH",
            "raw_model_label": "FAKE",
        }
        res = build_canonical_verification_result(
            claims_list=claims_list,
            evidence_data=evidence_data,
            prediction_result=prediction_result,
        )

        # Final assessment must be grounded in facts, not ML style
        self.assertEqual(res["final_assessment"]["verdict"], "LIKELY REAL")
        # ML signal must remain visibly separate
        self.assertEqual(res["ml_signal"]["status"], "FAKE")
        self.assertEqual(res["ml_signal"]["strength_label"], "LEANS FAKE")
        self.assertEqual(res["ml_signal"]["score"], 99.5)
        self.assertIn("LEANS FAKE", res["ml_signal"]["label_display"])

    def test_section_31_test_1_clearly_supported(self):
        """Test 1 — Clearly supported: A claim with authoritative primary evidence -> SUPPORTED"""
        from .verification.web_evidence import _judge_sources_heuristic, calculate_evidence_strength
        claim = "ISRO launched EOS-05 satellite from Sriharikota."
        sources = [{
            "url": "https://www.isro.gov.in/press-release/eos-05",
            "domain": "isro.gov.in",
            "text_excerpt": "ISRO launched EOS-05 satellite from Sriharikota spaceport.",
            "is_primary_source": True,
            "source_reliability_weight": 2.0,
            "source_reliability": "VERY_HIGH",
        }]
        verdicts = _judge_sources_heuristic(claim, sources)
        self.assertEqual(verdicts[0]["verdict"], "SUPPORTS")
        self.assertEqual(verdicts[0]["directness"], "DIRECT")
        sources[0]["content_verdict"] = verdicts[0]["verdict"]
        sources[0]["directness"] = verdicts[0]["directness"]
        score, label = calculate_evidence_strength(sources[0], claim)
        self.assertGreaterEqual(score, 1.0)
        self.assertEqual(label, "STRONG")

    def test_section_31_test_2_clearly_contradicted(self):
        """Test 2 — Clearly contradicted: A claim directly contradicted by authoritative evidence -> CONTRADICTED"""
        from .verification.web_evidence import _judge_sources_heuristic
        claim = "100 people attended the conference in London."
        sources = [{
            "url": "https://www.reuters.com/world/uk/conference-attendance",
            "domain": "reuters.com",
            "text_excerpt": "Official records state only 10 people attended the conference.",
            "is_primary_source": False,
            "source_reliability_weight": 1.2,
            "source_reliability": "HIGH",
        }]
        verdicts = _judge_sources_heuristic(claim, sources)
        self.assertEqual(verdicts[0]["verdict"], "CONTRADICTS")
        self.assertEqual(verdicts[0]["directness"], "DIRECT")

    def test_section_31_test_3_insufficient(self):
        """Test 3 — Insufficient: A claim with no reliable evidence -> INSUFFICIENT"""
        from .verification.web_evidence import _judge_sources_heuristic
        claim = "A secret tunnel was discovered under the city."
        sources = [{
            "url": "https://www.randomblog.net/tunnels",
            "domain": "randomblog.net",
            "text_excerpt": "Reports of secret tunnels could not be independently verified.",
            "is_primary_source": False,
            "source_reliability_weight": 0.3,
            "source_reliability": "LOW",
        }]
        verdicts = _judge_sources_heuristic(claim, sources)
        self.assertEqual(verdicts[0]["verdict"], "INSUFFICIENT")

    def test_section_31_test_4_mixed_article(self):
        """Test 4 — Mixed article: 2 supported, 2 contradicted, 0 insufficient -> MIXED / MISLEADING"""
        from .verification.canonical import build_canonical_verification_result
        claims = ["Claim 1", "Claim 2", "Claim 3", "Claim 4"]
        ev = {
            "claim_evidence": [
                {"claim": "Claim 1", "status": "SUPPORTED", "sources": [{"url": "http://s1.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 2", "status": "SUPPORTED", "sources": [{"url": "http://s2.org", "content_verdict": "SUPPORTS"}]},
                {"claim": "Claim 3", "status": "CONTRADICTED", "sources": [{"url": "http://s3.org", "content_verdict": "CONTRADICTS"}]},
                {"claim": "Claim 4", "status": "CONTRADICTED", "sources": [{"url": "http://s4.org", "content_verdict": "CONTRADICTS"}]},
            ]
        }
        res = build_canonical_verification_result(claims_list=claims, evidence_data=ev)
        self.assertEqual(res["claim_counts"]["supported"], 2)
        self.assertEqual(res["claim_counts"]["contradicted"], 2)
        self.assertEqual(res["claim_counts"]["insufficient"], 0)
        self.assertEqual(res["final_assessment"]["label"], "MIXED / MISLEADING")

    def test_section_31_test_5_ml_disagreement(self):
        """Test 5 — ML disagreement: ML -> LEANS FAKE, Evidence -> strongly supports -> Final assessment follows evidence, ML remains a separate signal"""
        from .verification.canonical import build_canonical_verification_result
        claims = ["ISRO launched satellite"]
        ev = {
            "claim_evidence": [
                {"claim": "ISRO launched satellite", "status": "SUPPORTED", "sources": [{"url": "https://isro.gov.in", "content_verdict": "SUPPORTS", "is_primary_source": True}]}
            ]
        }
        pred = {"prediction": "FAKE", "confidence": 98.7, "confidence_status": "HIGH", "raw_model_label": "FAKE"}
        res = build_canonical_verification_result(claims_list=claims, evidence_data=ev, prediction_result=pred)
        self.assertEqual(res["final_assessment"]["label"], "LIKELY REAL")
        self.assertEqual(res["ml_signal"]["label"], "LEANS FAKE")
        self.assertEqual(res["ml_signal"]["confidence"], 0.99)

    def test_section_31_test_6_duplicate_sources(self):
        """Test 6 — Duplicate sources: Multiple copies of the same article -> count as one independent source"""
        from .verification.canonical import build_canonical_verification_result
        claims = ["Event occurred"]
        ev = {
            "claim_evidence": [
                {
                    "claim": "Event occurred",
                    "status": "SUPPORTED",
                    "sources": [
                        {"url": "https://www.reuters.com/news/article?utm_source=twitter", "content_verdict": "SUPPORTS"},
                        {"url": "https://reuters.com/news/article?utm_medium=email", "content_verdict": "SUPPORTS"},
                        {"url": "https://reuters.com/news/article/", "content_verdict": "SUPPORTS"},
                    ]
                }
            ]
        }
        res = build_canonical_verification_result(claims_list=claims, evidence_data=ev)
        self.assertEqual(res["source_counts"]["unique"], 1)
        self.assertEqual(res["source_counts"]["supporting"], 1)

    def test_section_31_test_7_semantic_match(self):
        """Test 7 — Semantic match: Different wording but same factual meaning -> SUPPORTED when authoritative evidence establishes the claim"""
        from .verification.web_evidence import _judge_sources_heuristic
        claim = "ISRO launched GSLV-F17/EOS-05 on September 4, 2026."
        sources = [{
            "url": "https://www.isro.gov.in/launch-update",
            "domain": "isro.gov.in",
            "text_excerpt": "GSLV-F17 carrying EOS-05 lifted off from Satish Dhawan Space Centre on September 4, 2026.",
            "is_primary_source": True,
            "source_reliability_weight": 2.0,
        }]
        verdicts = _judge_sources_heuristic(claim, sources)
        self.assertEqual(verdicts[0]["verdict"], "SUPPORTS")
        self.assertEqual(verdicts[0]["directness"], "DIRECT")

    def test_section_31_test_8_similar_but_different_event(self):
        """Test 8 — Similar but different event: Same keywords but different date/location/entity -> UNRELATED not SUPPORTS"""
        from .verification.web_evidence import _judge_sources_heuristic
        claim = "ISRO launched EOS-05 on September 4, 2026."
        sources = [{
            "url": "https://www.space-archive.com/2023-launch",
            "domain": "space-archive.com",
            "text_excerpt": "ISRO launched EOS-05 on September 4, 2023.",
            "is_primary_source": False,
            "source_reliability_weight": 0.5,
        }]
        verdicts = _judge_sources_heuristic(claim, sources)
        self.assertEqual(verdicts[0]["verdict"], "UNRELATED")

    def test_section_32_pslv_c62_case(self):
        """Test Section 32: PSLV-C62 Case verification end-to-end consistency"""
        from .verification.canonical import build_canonical_verification_result
        from .verification.claims import decompose_into_atomic_claims, normalize_claim_text
        article = (
            "ISRO successfully launched the PSLV-C62 mission carrying the EOS-09 satellite "
            "from the Satish Dhawan Space Centre in Sriharikota. The mission successfully injected "
            "the satellite into the intended sun-synchronous orbit."
        )
        claims = decompose_into_atomic_claims(article)
        self.assertGreaterEqual(len(claims), 2)
        norm_claim = normalize_claim_text(claims[0])
        self.assertIn("PSLV-C62", norm_claim)

        ev_data = {
            "claim_evidence": [
                {
                    "claim": claims[0],
                    "status": "SUPPORTED",
                    "confidence": 0.94,
                    "reason": "Official ISRO announcement confirms the PSLV-C62 launch.",
                    "sources": [
                        {
                            "url": "https://www.isro.gov.in/pslv-c62-mission",
                            "domain": "isro.gov.in",
                            "content_verdict": "SUPPORTS",
                            "is_primary_source": True,
                            "source_reliability": "VERY_HIGH",
                            "strength_score": 2.0,
                            "strength_label": "STRONG",
                            "directness": "DIRECT",
                            "excerpt": "PSLV-C62 successfully launched from Sriharikota.",
                        }
                    ]
                },
                {
                    "claim": claims[1],
                    "status": "SUPPORTED",
                    "confidence": 0.92,
                    "reason": "Tracking data confirmed satellite orbit injection.",
                    "sources": [
                        {
                            "url": "https://www.isro.gov.in/pslv-c62-mission",
                            "domain": "isro.gov.in",
                            "content_verdict": "SUPPORTS",
                            "is_primary_source": True,
                            "source_reliability": "VERY_HIGH",
                            "strength_score": 2.0,
                            "strength_label": "STRONG",
                            "directness": "DIRECT",
                            "excerpt": "Satellite injected into sun-synchronous orbit.",
                        }
                    ]
                }
            ]
        }
        res = build_canonical_verification_result(
            claims_list=claims[:2],
            evidence_data=ev_data,
            input_type="Pasted Text",
        )
        # Verify Section 30 assertions
        self.assertEqual(res["claim_counts"]["supported"] + res["claim_counts"]["contradicted"] + res["claim_counts"]["insufficient"], 2)
        self.assertEqual(res["claim_counts"]["supported"], 2)
        self.assertEqual(res["claim_counts"]["contradicted"], 0)
        # Deduplicated sources check (same URL used for both claims must count as 1 unique source)
        self.assertEqual(res["source_counts"]["unique"], 1)
        self.assertEqual(res["source_counts"]["supporting"], 1)
        self.assertEqual(res["final_assessment"]["label"], "LIKELY REAL")
        self.assertEqual(res["timeline"][3]["counts"]["supported"], 2)
        self.assertEqual(res["timeline"][3]["counts"]["contradicted"], 0)


class UniversalEvidenceGroundedVerificationTests(TestCase):
    """
    Universal, domain-independent test suite satisfying:
    - Section 1: Core pipeline grounded in exact proposition entailment
    - Section 7: Topical relevance is NOT evidence
    - Section 9: Contradiction detection (action antonyms, conflict terms)
    - Section 10: Numerical verification (percentages, currencies, metrics)
    - Section 11: Date verification (day/month/year mismatches)
    - Section 15: Search for disconfirmation (dual query generation)
    - Section 21 & 22: Insufficient is NOT fake, absence of evidence != fake
    - Section 27 & 28: Single canonical verification object and 100% UI consistency
    - Section 33: Artemis II Moon landing bug fix without NASA-specific hardcoding
    """

    def test_artemis_ii_general_action_conflict_is_contradicted(self):
        """
        Section 33 Regression Test:
        Claim asserts astronauts 'landed on the Moon', retrieved source asserts
        mission is a 'lunar flyby without landing'.
        Old system: 40% keyword overlap marked this SUPPORTED.
        New system: Action conflict (landed <-> flyby / without landing) evaluates to CONTRADICTS.
        Must work with ZERO NASA-specific hardcoded rules.
        """
        from detector.verification.entailment import evaluate_claim_evidence_entailment
        claim = "NASA astronauts landed on the Moon during the Artemis II mission."
        source_text = "NASA's Artemis II mission will carry four astronauts on a lunar flyby test flight without landing on the Moon."

        ent_res = evaluate_claim_evidence_entailment(claim, source_text)
        self.assertEqual(ent_res["verdict"], "CONTRADICTS")
        self.assertEqual(ent_res["relationship"], "CONTRADICTS")
        self.assertIn(ent_res["conflict_type"], ("ACTION_CONFLICT", "NEGATION_CONFLICT"))

        # Verify claim verification status through universal entailment engine
        with patch.object(web_evidence, "_search_candidate_sources", return_value=[{"title": "Artemis II Flyby", "url": "https://space.example/artemis2"}]), \
             patch.object(web_evidence, "ANTHROPIC_API_KEY", ""), \
             patch.object(web_evidence, "GEMINI_API_KEY", ""), \
             patch.object(web_evidence, "TAVILY_API_KEY", "mock-key"), \
             patch("detector.extract_text.extract_from_url",
                   return_value=(source_text, None, {"domain": "space.example", "title": "Artemis II Flyby", "is_https": True})):
            claim_rec = web_evidence.verify_claim_with_evidence(claim)

        self.assertEqual(claim_rec["status"], "CONTRADICTED")
        self.assertEqual(claim_rec["sources"][0]["content_verdict"], "CONTRADICTS")

        # Canonical assessment check: single central contradicted claim leads to LIKELY FAKE
        canonical = build_canonical_verification_result(
            claims_list=[claim],
            evidence_data={"claim_evidence": [claim_rec]},
            prediction_result={"prediction": "REAL", "confidence": 98.0},
        )
        self.assertEqual(canonical["final_assessment"]["label"], "LIKELY FAKE")
        self.assertEqual(canonical["claim_counts"]["contradicted"], 1)
        self.assertEqual(canonical["claim_counts"]["supported"], 0)

    def test_topical_relevance_vs_exact_action_neutral(self):
        """
        Section 7: Topical relevance is NOT evidence.
        Claim: 'Company X launched Product Y on January 10.'
        Source: 'Company X develops Product Y.' -> NEUTRAL / INSUFFICIENT, NEVER SUPPORTS.
        """
        from detector.verification.entailment import evaluate_claim_evidence_entailment
        claim = "Company X launched Product Y on January 10."

        # Case 1: Company develops product (topical background) -> INSUFFICIENT
        bg_text = "Company X develops Product Y for enterprise clients across multiple platforms."
        res1 = evaluate_claim_evidence_entailment(claim, bg_text)
        self.assertEqual(res1["verdict"], "INSUFFICIENT")
        self.assertEqual(res1["relationship"], "NEUTRAL")

        # Case 2: Under discussion / hypothetical -> INSUFFICIENT
        hyp_text = "Discussions were held regarding whether Company X would introduce Product Y to market."
        res2 = evaluate_claim_evidence_entailment(claim, hyp_text)
        self.assertEqual(res2["verdict"], "INSUFFICIENT")
        self.assertEqual(res2["relationship"], "NEUTRAL")

        # Case 3: Contradiction (postponed launch) -> CONTRADICTS
        postpone_text = "Company X postponed the Product Y launch to February."
        res3 = evaluate_claim_evidence_entailment(claim, postpone_text)
        self.assertEqual(res3["verdict"], "CONTRADICTS")

        # Case 4: Corroboration -> SUPPORTS
        support_text = "Company X officially launched Product Y on January 10 during the annual keynote."
        res4 = evaluate_claim_evidence_entailment(claim, support_text)
        self.assertEqual(res4["verdict"], "SUPPORTS")

    def test_modality_opinion_and_prediction_not_fake(self):
        """
        Section 5: Subjective opinions and future predictions must not be labelled REAL or FAKE.
        They must be categorized as INSUFFICIENT (OPINION / PREDICTION) without guessing.
        """
        from detector.verification.claims import classify_claim_modality

        # Opinion
        opinion_claim = "I think this is the best phone ever produced."
        mod1 = classify_claim_modality(opinion_claim)
        self.assertEqual(mod1["modality"], "OPINION")
        self.assertFalse(mod1["is_checkable"])

        op_rec = web_evidence.verify_claim_with_evidence(opinion_claim)
        self.assertEqual(op_rec["status"], "INSUFFICIENT")

        # Prediction
        prediction_claim = "Company X will release a new electric vehicle next month."
        mod2 = classify_claim_modality(prediction_claim)
        self.assertEqual(mod2["modality"], "PREDICTION")
        self.assertFalse(mod2["is_checkable"])

        pred_rec = web_evidence.verify_claim_with_evidence(prediction_claim)
        self.assertEqual(pred_rec["status"], "INSUFFICIENT")

    def test_numeric_metric_discrepancy_contradicts(self):
        """
        Section 10: Numbers must be verified exactly.
        40% != 4%, $5 billion != $5 million, 100 deaths != 10 deaths.
        """
        from detector.verification.entailment import evaluate_claim_evidence_entailment

        # Percentage discrepancy
        pct_claim = "Company revenue increased by 40%."
        pct_evidence = "Company revenue increased by 4% in the latest quarter."
        res1 = evaluate_claim_evidence_entailment(pct_claim, pct_evidence)
        self.assertEqual(res1["verdict"], "CONTRADICTS")
        self.assertEqual(res1["conflict_type"], "NUMERIC_CONFLICT")

        # Currency discrepancy
        curr_claim = "The company secured $5 billion in funding."
        curr_evidence = "The company secured $5 million in early seed funding."
        res2 = evaluate_claim_evidence_entailment(curr_claim, curr_evidence)
        self.assertEqual(res2["verdict"], "CONTRADICTS")
        self.assertEqual(res2["conflict_type"], "NUMERIC_CONFLICT")

        # Quantity count discrepancy
        qty_claim = "The incident resulted in 100 deaths."
        qty_evidence = "Authorities reported 10 deaths following the incident."
        res3 = evaluate_claim_evidence_entailment(qty_claim, qty_evidence)
        self.assertEqual(res3["verdict"], "CONTRADICTS")
        self.assertEqual(res3["conflict_type"], "NUMERIC_CONFLICT")

    def test_date_discrepancy_contradicts(self):
        """
        Section 11: Date verification.
        October 4 != October 5 -> DATE_CONFLICT CONTRADICTS.
        """
        from detector.verification.entailment import evaluate_claim_evidence_entailment
        date_claim = "The official summit took place on October 4."
        date_evidence = "The official summit took place on October 5 in Geneva."
        res = evaluate_claim_evidence_entailment(date_claim, date_evidence)
        self.assertEqual(res["verdict"], "CONTRADICTS")
        self.assertEqual(res["conflict_type"], "DATE_CONFLICT")

    def test_location_discrepancy_contradicts(self):
        """
        Section 9 & 19: Geographic location consistency.
        Delhi != Mumbai -> LOCATION_CONFLICT CONTRADICTS.
        """
        from detector.verification.entailment import evaluate_claim_evidence_entailment
        loc_claim = "The bilateral accord was signed in Delhi."
        loc_evidence = "The bilateral accord was signed in Mumbai following week-long talks."
        res = evaluate_claim_evidence_entailment(loc_claim, loc_evidence)
        self.assertEqual(res["verdict"], "CONTRADICTS")
        self.assertEqual(res["conflict_type"], "LOCATION_CONFLICT")

    def test_disconfirmation_search_queries_generated(self):
        """
        Section 15: Dual query generation (support queries + contradiction/disconfirmation queries).
        """
        claim = "Candidate Smith won the presidential election."
        queries = web_evidence.generate_search_queries(claim)
        self.assertGreaterEqual(len(queries), 2)
        # Verify disconfirmation query exists with refutation or conflict terms
        queries_text = " ".join(queries).lower()
        self.assertTrue(
            any(t in queries_text for t in ("lost", "defeated", "conceded", "denied", "fact check", "debunked")),
            f"Expected disconfirmation term in queries, got: {queries}"
        )

    def test_universal_domains_cross_section(self):
        """
        Section 2: Domain-independent verification across Politics, Business, Sports, Health, Entertainment.
        """
        from detector.verification.entailment import evaluate_claim_evidence_entailment

        # Politics: resigned vs denied resignation
        pol_claim = "Minister X resigned from the cabinet."
        pol_ev = "Minister X denied resignation reports and remains in office."
        self.assertEqual(evaluate_claim_evidence_entailment(pol_claim, pol_ev)["verdict"], "CONTRADICTS")

        # Business: profit vs net loss
        biz_claim = "Global Motors posted a net profit in Q4."
        biz_ev = "Global Motors posted a net loss of $400 million in Q4."
        self.assertEqual(evaluate_claim_evidence_entailment(biz_claim, biz_ev)["verdict"], "CONTRADICTS")

        # Sports: won vs lost/defeated
        spt_claim = "City United won the national championship."
        spt_ev = "City United was defeated in the final match by rivals."
        self.assertEqual(evaluate_claim_evidence_entailment(spt_claim, spt_ev)["verdict"], "CONTRADICTS")

        # Entertainment: died vs alive/survived
        ent_claim = "Famous Actor X died in a car crash."
        ent_ev = "Famous Actor X survived the accident and is alive and in stable condition."
        self.assertEqual(evaluate_claim_evidence_entailment(ent_claim, ent_ev)["verdict"], "CONTRADICTS")

    def test_canonical_invariants_and_zero_fabrication(self):
        """
        Section 27 & 28: Canonical object invariants:
        - 100% of summary counts match claim statuses
        - 100% of UI sections consume the canonical verification data
        - Zero fabricated sources
        """
        claims = [
            "Company A launched Product X on Monday.",
            "Company A sold 100,000 units.",
        ]
        ev_data = {
            "claim_evidence": [
                {
                    "claim": claims[0],
                    "status": "SUPPORTED",
                    "sources": [
                        {
                            "url": "https://news.example/launch",
                            "domain": "news.example",
                            "content_verdict": "SUPPORTS",
                            "is_primary_source": True,
                            "strength_score": 2.0,
                            "strength_label": "STRONG",
                            "directness": "DIRECT",
                            "text_excerpt": "Company A launched Product X on Monday.",
                            "evidence_text": "Company A launched Product X on Monday.",
                        }
                    ]
                },
                {
                    "claim": claims[1],
                    "status": "CONTRADICTED",
                    "sources": [
                        {
                            "url": "https://stats.example/sales",
                            "domain": "stats.example",
                            "content_verdict": "CONTRADICTS",
                            "is_primary_source": True,
                            "strength_score": 2.0,
                            "strength_label": "STRONG",
                            "directness": "DIRECT",
                            "text_excerpt": "Company A sold 10,000 units, far below claims.",
                            "evidence_text": "Company A sold 10,000 units, far below claims.",
                        }
                    ]
                }
            ]
        }

        canonical = build_canonical_verification_result(
            claims_list=claims,
            evidence_data=ev_data,
            prediction_result={"prediction": "REAL", "confidence": 95.0},
        )

        # 1. Counts sum must match total claims exactly
        total = canonical["claim_counts"]["total"]
        sup = canonical["claim_counts"]["supported"]
        contra = canonical["claim_counts"]["contradicted"]
        insuf = canonical["claim_counts"]["insufficient"]
        self.assertEqual(sup + contra + insuf, total)
        self.assertEqual(sup, 1)
        self.assertEqual(contra, 1)
        self.assertEqual(insuf, 0)

        # 2. Both supported and contradicted claims exist -> MIXED / MISLEADING
        self.assertEqual(canonical["final_assessment"]["label"], "MIXED / MISLEADING")

        # 3. Timeline counts must match claim counts exactly
        timeline_step4 = canonical["timeline"][3]
        self.assertEqual(timeline_step4["counts"]["supported"], sup)
        self.assertEqual(timeline_step4["counts"]["contradicted"], contra)
        self.assertEqual(timeline_step4["counts"]["insufficient"], insuf)

        # 4. Processed claims contain structured attributes
        c0 = canonical["claims"][0]
        self.assertIn("modality", c0)
        self.assertIn("subject", c0)
        self.assertIn("action", c0)
        self.assertIn("location", c0)
        self.assertIn("evidence", c0)
        self.assertEqual(c0["evidence"][0]["relationship"], "SUPPORTS")
        self.assertEqual(c0["evidence"][0]["evidence_text"], "Company A launched Product X on Monday.")




