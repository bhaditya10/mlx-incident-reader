"""Checker independent of report generator. Rejects bool-as-int and extra keys."""
import json, re
FIELDS = ('service', 'severity', 'status', 'duration_minutes')

def check(text, expected):
    try:
        got = json.loads(text)
    except (ValueError, TypeError):
        return {'json': False, 'schema': False, 'fields': {k: False for k in FIELDS}, 'exact': False}
    schema = isinstance(got, dict) and set(got) == set(FIELDS)
    if schema:
        schema = all((got[k] is None or type(got[k]) is str for k in FIELDS[:3])) and (got['duration_minutes'] is None or (type(got['duration_minutes']) is int and got['duration_minutes'] >= 0))
    if schema:
        schema = got['service'] in [None, 'checkout', 'billing', 'search', 'inventory', 'auth', 'notifications', 'gateway', 'catalog'] and got['severity'] in [None, 'SEV1', 'SEV2', 'SEV3', 'SEV4'] and (got['status'] in [None, 'investigating', 'identified', 'recovering', 'resolved'])
    fields = {k: bool(schema and type(got[k]) is type(expected[k]) and (got[k] == expected[k])) for k in FIELDS}
    return {'json': True, 'schema': bool(schema), 'fields': fields, 'exact': all(fields.values())}

def rules(text):

    def one(pattern):
        matches = set(re.findall(pattern, text, re.I))
        return next(iter(matches)) if len(matches) == 1 else None
    service = one('\\b(checkout|billing|search|inventory|auth|notifications|gateway|catalog)\\b')
    severity = one('\\b(SEV[1-4])\\b')
    status = one('\\b(investigating|identified|recovering|resolved)\\b')
    duration = one('\\b(\\d+)\\s+minutes?\\b')
    return dict(service=service.lower() if service else None, severity=severity.upper() if severity else None, status=status.lower() if status else None, duration_minutes=int(duration) if duration is not None else None)

def selftest():
    e = dict(service='checkout', severity='SEV2', status='recovering', duration_minutes=0)
    assert check(json.dumps(e), e)['exact']
    assert not check(json.dumps({**e, 'duration_minutes': None}), e)['exact']
    assert not check(json.dumps({**e, 'duration_minutes': False}), e)['schema']
    assert not check(json.dumps({**e, 'extra': 1}), e)['schema']
    assert not check('not json', e)['json']
    assert rules('Checkout is recovering. SEV2. 0 minutes.') == e
