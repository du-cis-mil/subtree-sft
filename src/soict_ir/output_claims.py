"""Fixed output inventory without historical manuscript numbers."""
from pathlib import Path
from .protocol import object_sha256, write_json

INVENTORY_SHA256 = "b9bc1ff5b73739710cab5d871a3d7763194091de1f35f2cf1866dea68748d223"

def validate_manifest(manifest):
    # Full canonical identity binds IDs, types, locations, paths and all labels.
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 2 or object_sha256(manifest) != INVENTORY_SHA256:
        raise ValueError('Invalid output inventory: all 159 reviewed fields and 2 figures/4 tables are required')

def evaluate_claims(manifest, context, *, historical=False):
    from .claims import resolve
    validate_manifest(manifest)
    claims=[]
    for spec in manifest['claims']:
        c=dict(spec)
        if c.get('status')=='method_description':
            c.update(recomputed_value=None,historical_value_verified=False)
        else:
            value=resolve(context,c['computed_path'])
            c.update(recomputed_value=value,status='unverified' if value is None else 'computed_value',
                     historical_value_verified=False,manuscript_comparison=None,
                     reason='Input unavailable; no substitution.' if value is None else 'Computed from supplied inputs; historical comparison not provided.')
        claims.append(c)
    return claims

def save_claims(directory, manifest, claims):
    root=Path(directory)
    counts={s:sum(c['status']==s for c in claims) for s in sorted({c['status'] for c in claims})}
    write_json(root/'manuscript_claims.json',dict(schema_version=2,scope=manifest['scope'],status_counts=counts,claims=claims))
    lines=['# Computed output coverage','','Historical reference values are not supplied. Method descriptions are not measured results.','',
           '| ID | Computed | Status | Path |','|---|---|---|---|']
    for c in claims:
        value=c['recomputed_value']
        lines.append('| '+ ' | '.join(str(v).replace('|','\\|') for v in [c['id'],'NA' if value is None else value,c['status'],c['computed_path']])+' |')
    (root/'manuscript_coverage.md').write_text('\n'.join(lines)+'\n')
    return counts
