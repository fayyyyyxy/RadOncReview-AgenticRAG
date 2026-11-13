"""Local browser QA fixture. Run uvicorn tests.ui_fixture_app:app; no AWS calls.

Synthetic content exercises citation inspection, warnings, and context changes.
This module is never imported by the production app or included in its image.
"""
from types import SimpleNamespace

import api
from core.models import AgentResponse, Citation


def query(question, labels, **kwargs):
    for stage in ('input', 'retrieve', 'synthesize', 'citations', 'evidence'):
        kwargs['progress_callback'](stage)
    passage = ('Synthetic supporting passage for interface testing only. ' * 25
               + '\nEND OF FULL PASSAGE — beyond the old 200-character excerpt.')
    return AgentResponse(
        query=question,
        answer='**Synthetic UI test — not a clinical answer.**\n\nInspect the full retrieved passage [1]. '
               'This grouped citation also opens it [1, 2].\n\n`Code [1] stays unchanged.`',
        citations=[Citation(1, 'Synthetic document one', passage[:300], .92, 1,
                            source_url='https://example.com', document_title='Synthetic source one'),
                   Citation(2, 'Synthetic document two', 'Second synthetic passage.', .85, 2)],
        context_passages=[{'citation_id': 1, 'source_path': 'Synthetic document one', 'text': passage},
                          {'citation_id': 2, 'source_path': 'Synthetic document two', 'text': 'Second synthetic passage.'}],
        metadata={'evidence_check': {'status': 'completed', 'flagged': True,
                  'judgment': 'Partial Abstention Preferred', 'reason': 'Synthetic warning used to test the interface.'}},
        latency_ms=10)


api._agent = SimpleNamespace(query=query)
app = api.app
