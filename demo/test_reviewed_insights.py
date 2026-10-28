"""Focused checks for the two-role handoff. All provider responses are mocked."""
import json
import pytest
from demo.reviewed_insights import ReviewedInsightService
from demo.test_azure_insights import evidence, environment, completion


def accepted(e, **changes):
    return dict(verdict='pass',player_id=e['player_id'],anchor_frame=e['anchor_frame'],
                target_time_s=e['target_time_s'],player_correct=True,numbers_correct=True,
                time_correct=True,forecast_labeled=True,**changes)


def mock_service(env,e,review=None):
    calls=[]
    def transport(url,headers,payload,timeout):
        calls.append(payload)
        if payload['response_format']['json_schema']['name']=='commentary_review':
            if isinstance(review,Exception): raise review
            return completion(json.dumps(review or accepted(e)))
        return completion()
    return ReviewedInsightService(env=env,transport=transport),calls


def test_two_independent_calls_and_cached_review(evidence,environment):
    service,calls=mock_service(environment,evidence)
    first=service.generate(evidence)
    assert len(calls)==2
    assert 'editor' in calls[0]['messages'][0]['content']
    assert 'Reviewer Agent' in calls[1]['messages'][0]['content']
    assert first['agent_trace']['reviewer']['status']=='passed'
    assert first['reviewed_script']
    second=service.generate(evidence)
    assert len(calls)==2
    assert second['agent_trace']['analyst']['cached']
    assert second['agent_trace']['reviewer']['cached']


@pytest.mark.parametrize('change',[{'player_id':'Away_9'},{'anchor_frame':301},{'target_time_s':14.0}, {'numbers_correct':False},{'forecast_labeled':False}])
def test_wrong_identity_time_or_failed_check_never_approved(evidence,environment,change):
    review=accepted(evidence);review.update(change)
    service,calls=mock_service(environment,evidence,review)
    result=service.generate(evidence)
    assert result['agent_trace']['reviewer']['status']!='passed'
    assert result['reviewed_script'] is None
    assert len(calls)==2


def test_shared_quota_blocks_reviewer(evidence,environment):
    service,calls=mock_service({**environment,'INSIGHT_MAX_CALLS_TOTAL':'1'},evidence)
    result=service.generate(evidence)
    assert len(calls)==1
    assert result['agent_trace']['reviewer']['reason']=='budget_exhausted'


def test_timeout_keeps_evidence_without_claiming_review(evidence,environment):
    service,calls=mock_service(environment,evidence,TimeoutError())
    result=service.generate(evidence)
    assert result['grounded'] and result['agent_trace']['reviewer']['status']=='unavailable'
    assert result['reviewed_script'] is None


def test_unconfigured_does_not_invent_agent_success(evidence):
    result=ReviewedInsightService(env={}).generate(evidence)
    assert result['mode']=='local'
    assert result['agent_trace']['analyst']['status']=='local_fallback'
    assert result['agent_trace']['reviewer']['status']=='not_run'
