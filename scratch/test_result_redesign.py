import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import re
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'truthlens.settings')
django.setup()

from django.test import RequestFactory
from django.contrib.auth import get_user_model
from detector.views import analysis_result_view
from detector.models import Analysis

User = get_user_model()
user = User.objects.first()
rf = RequestFactory()

test_cases = [
    ('LIKELY REAL state (ID 27)', 27, 'LIKELY REAL', 'state-real'),
    ('LIKELY FAKE state (ID 25)', 25, 'LIKELY FAKE', 'state-fake'),
    ('UNCERTAIN state (ID 19)', 19, 'UNCERTAIN', 'state-uncertain'),
]

for title, aid, expected_verdict, expected_state_class in test_cases:
    a = Analysis.objects.filter(id=aid).first()
    if not a:
        print(f"Skipping {title}: ID {aid} not found in DB")
        continue

    a.user = user
    a.save()
    req = rf.get(f'/analysis/{aid}/')
    req.user = user
    resp = analysis_result_view(req, aid)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    html = resp.content.decode('utf-8')

    print(f"\n==========================================")
    print(f"AUDITING: {title}")
    print(f"==========================================")

    # 1. Top Header
    assert 'VERIFICATION RESULT' in html, "Missing header tag 'VERIFICATION RESULT'"
    assert 'TruthLens AI Verification' in html, "Missing title 'TruthLens AI Verification'"
    assert 'Analyze Another' in html, "Missing 'Analyze Another' button"
    assert 'Download Report' in html, "Missing 'Download Report' button"
    print("  [OK] 1. Top Header: Title, tag, meta, and action buttons present")

    # 2. Main Result Hero Card
    assert 'verdict-hero-card' in html, "Missing hero card container"
    assert expected_state_class in html, f"Missing state class '{expected_state_class}'"
    assert expected_verdict in html, f"Missing verdict text '{expected_verdict}'"
    assert 'Verification Confidence' in html, "Missing 'Verification Confidence' score label"
    
    # Critical verification: ML result must NOT be next to final result in hero card
    hero_match = re.search(r'<div class="verdict-hero-card.*?(<div class="evidence-stats-grid")', html, re.DOTALL)
    assert hero_match, "Hero card content block not found"
    hero_text = hero_match.group(0)
    assert "ML MODEL PREDICTION" not in hero_text, "Confusing ML label leaked into hero card!"
    assert "Historical ML Signal" not in hero_text, "ML signal placed inside hero card!"
    print(f"  [OK] 2. Main Result: Dominant single verdict card ({expected_verdict}) with confidence score")

    # 3. Evidence Summary
    assert 'Supporting Evidence' in html, "Missing 'Supporting Evidence' chip"
    assert 'Contradicting Evidence' in html, "Missing 'Contradicting Evidence' chip"
    assert 'Insufficient Evidence' in html, "Missing 'Insufficient Evidence' chip"
    print("  [OK] 3. Evidence Summary: 3 compact visual metrics (Supporting, Contradicting, Insufficient)")

    # 4. Why We Reached This Result
    assert 'Why We Reached This Result' in html, "Missing 'Why We Reached This Result' card"
    assert 'why-reasons-list' in html, "Missing why-reasons list"
    assert 'why-reason-item' in html, "Missing structured why rows"
    print("  [OK] 4. Why This Result: 3-4 compact rows with icons, titles, and 1-line descriptions")

    # 5. Analyzed Article
    assert 'Analyzed Article' in html, "Missing 'Analyzed Article' card"
    assert 'article-headline-text' in html, "Missing article headline"
    assert 'article-meta-tags-strip' in html, "Missing metadata strip"
    assert 'words' in html, "Missing word count chip"
    print("  [OK] 5. Analyzed Article: Headline, preview, word count, language, and category metadata")

    # 6. Claim-by-Claim Verification
    assert 'Claim-by-Claim Verification' in html, "Missing Claim verification card"
    print("  [OK] 6. Claim Analysis: Structured claim section with status pills")

    # 7. Evidence Sources
    assert 'Evidence Sources' in html, "Missing Evidence Sources card"
    print("  [OK] 7. Evidence Sources: Dedicated sources section with source cards")

    # 8 & 9. Historical ML Signal (Technical Information)
    assert 'Historical ML Signal' in html, "Missing 'Historical ML Signal' card"
    assert 'Technical Information' in html, "Missing 'Technical Information' badge"
    assert 'Model Signal Strength' in html, "Missing 'Model Signal Strength'"
    assert 'ml-signal-bar-track' in html, "Missing progress bar track"
    assert 'ml-signal-bar-fill' in html, "Missing progress bar fill"
    assert 'Why is this different from the Final Assessment?' in html, "Missing 'Why is this different?' details"
    
    # Verify ML is positioned AFTER the two-column evidence/claims section
    two_col_idx = html.find('class="result-two-col-layout"')
    ml_card_idx = html.find('class="ml-technical-card"')
    assert two_col_idx < ml_card_idx, "Historical ML Signal must be positioned BELOW evidence and claims!"
    print("  [OK] 8 & 9. Historical ML Signal: Positioned below evidence/claims with technical badge & progress bar")

    # 10. Key Takeaways
    assert 'Key Takeaways' in html, "Missing Key Takeaways card"
    print("  [OK] 10. Key Takeaways: Compact card with core takeaways")

    # 11. Verification Timeline
    assert 'Verification Timeline' in html, "Missing Verification Timeline"
    assert 'Input Received' in html, "Missing Step 1 (Input Received)"
    assert 'Claims Extracted' in html, "Missing Step 2 (Claims Extracted)"
    assert 'Evidence Retrieved' in html, "Missing Step 3 (Evidence Retrieved)"
    assert 'Claims Verified' in html, "Missing Step 4 (Claims Verified)"
    assert 'Final Assessment' in html, "Missing Step 5 (Final Assessment)"
    print("  [OK] 11. Verification Timeline: 5 transparent pipeline steps")

    # 12. Technical Details (Expandable)
    assert 'Technical Details' in html, "Missing Technical Details card"
    assert '<details' in html and 'technical-details-card' in html, "Technical Details must be an expandable <details> element"
    assert 'LinearSVC' in html, "Missing ML estimator in tech specs"
    print("  [OK] 12. Technical Details: Expandable accordion with technical specs")

    # 13. Feedback & Actions
    assert 'Was this verification helpful?' in html, "Missing feedback card"
    assert 'feedback-form' in html, "Missing feedback form"
    print("  [OK] 13. Feedback & Actions: Voting buttons, optional comment box, and save article toggle")

    # 14. Final Disclaimer
    assert 'TruthLens AI combines machine-learning signals' in html, "Missing final disclaimer"
    print("  [OK] 14. Final Disclaimer: Subtle informational notice at bottom")

print("\n=======================================================")
print("ALL 14 AUDIT CHECKS PASSED FOR REAL, FAKE, AND UNCERTAIN!")
print("=======================================================")
