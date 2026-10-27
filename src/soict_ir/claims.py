"""Compare computed values with a fixed manuscript inventory; never fill missing values."""
import json
from pathlib import Path
import math
from .protocol import object_sha256, write_json

# Reviewed inventory identity, independent of a caller-supplied manifest. The
# digest binds every ID, value, location and computed field, not just its count.
# JSON whitespace/key order may change without changing this identity.
MANUSCRIPT_SHA256 = "d82c4b7f67b83bb651b2ad3d7d2a7e61becf6646dfa619d2ceb1265fdbc32c4b"
INVENTORY_SHA256 = "4aec03f298abe08e18ecdb1453fee94e3cc0e49f772789b802b67e35294354c2"
FIGURES = {"fig:method_overview", "fig:depthwise_top1"}
TABLES = {"tab:implementation", "tab:tree_granularity", "tab:sample_level", "tab:prompt_ablation"}


def validate_manifest(manifest):
    """Accept only a complete pinned inventory, even in partial mode."""
    if manifest.get("schema_version") == 2:
        from .output_claims import validate_manifest as validate_output
        return validate_output(manifest)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Invalid claim inventory schema")
    if manifest.get("manuscript_sha256") != MANUSCRIPT_SHA256:
        raise ValueError("Unsupported manuscript identity in claim inventory")
    for field, expected in (("figures", FIGURES), ("tables", TABLES)):
        values = manifest.get(field)
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or \
           len(values) != len(expected) or set(values) != expected:
            raise ValueError("Claim inventory requires all manuscript " + field)
    claims = manifest.get("claims")
    if not isinstance(claims, list) or len(claims) != 159:
        raise ValueError("Claim inventory requires all 159 fixed claims")
    ids = set()
    for spec in claims:
        required = {"id", "manuscript_value", "computed_path", "locations"}
        if not isinstance(spec, dict) or not required.issubset(spec):
            raise ValueError("Claim inventory is missing required fields")
        claim_id = spec["id"]
        if not isinstance(claim_id, str) or not claim_id or claim_id in ids:
            raise ValueError("Invalid or duplicate claim ID in inventory")
        ids.add(claim_id)
        locations = spec["locations"]
        if not isinstance(locations, list) or not locations:
            raise ValueError("Claim locations must be a nonempty list: " + claim_id)
        for location in locations:
            if not isinstance(location, dict) or any(not isinstance(location.get(k), str) or not location[k]
                    for k in ("section", "lines")) or ("label" in location and
                    (not isinstance(location["label"], str) or location["label"] not in FIGURES | TABLES)):
                raise ValueError("Invalid claim location: " + claim_id)
        if spec.get("status") == "method_description":
            if spec["computed_path"] is not None or not isinstance(spec["manuscript_value"], str):
                raise ValueError("Invalid method-description claim: " + claim_id)
        else:
            if not isinstance(spec["computed_path"], str) or not spec["computed_path"] or "decimals" not in spec:
                raise ValueError("Missing/invalid claim computation field: " + claim_id)
            decimals = spec["decimals"]
            if decimals is not None and (type(decimals) is not int or not 0 <= decimals <= 15):
                raise ValueError("Invalid claim precision: " + claim_id)
            value = spec["manuscript_value"]
            if type(value) not in (str, bool, int, float) or (type(value) is float and not math.isfinite(value)):
                raise ValueError("Invalid manuscript value in claim: " + claim_id)
    if object_sha256(manifest) != INVENTORY_SHA256:
        raise ValueError("Claim inventory differs from the fixed reviewed IDs, fields or manuscript references")


def resolve(context, path):
    value = context
    try:
        for key in path.split("/"):
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value
    except (KeyError, IndexError, TypeError):
        return None


def evaluate_claims(manifest, context, *, historical):
    validate_manifest(manifest)
    if manifest["schema_version"] == 2:
        from .output_claims import evaluate_claims as evaluate_output
        return evaluate_output(manifest, context, historical=historical)
    claims = []
    for spec in manifest["claims"]:
        claim = dict(spec)
        if spec.get("status") == "method_description":
            claim.update(recomputed_value=None, historical_value_verified=False)
            claims.append(claim)
            continue
        value = resolve(context, spec["computed_path"])
        expected = spec["manuscript_value"]
        legacy = resolve(context, spec["legacy_path"]) if "legacy_path" in spec else None
        if value is None:
            match, status = False, "unverified"
            reason = "Required saved outcome or audit/statistics input is unavailable; no substitution."
        else:
            decimals = spec.get("decimals")
            if spec.get("comparison") == "lt":
                match = value < expected
            elif decimals is not None and isinstance(value, (int, float)):
                match = round(value, decimals) == expected
            else:
                match = value == expected
            status = "verified_match" if match else "corrected_value"
            reason = spec.get("correction") or ("Matches manuscript at its stated precision." if match else
                "Recomputed observation differs from manuscript; requires author review.")
            if spec.get("correction"):
                status = "corrected_value"
            if not historical:
                status = "new_run_value"
                reason += " This suite is not proof of a historical run's value."
        claim.update(recomputed_value=value, status=status, reason=reason,
                     manuscript_comparison=match,
                     historical_value_verified=historical and status == "verified_match")
        if "legacy_path" in spec:
            claim["legacy_recomputed_value"] = legacy
            claim["legacy_matches_manuscript"] = legacy is not None and round(legacy, spec["decimals"]) == expected
        claims.append(claim)
    return claims


def save_claims(directory, manifest, claims):
    if manifest["schema_version"] == 2:
        from .output_claims import save_claims as save_output
        return save_output(directory, manifest, claims)
    directory = Path(directory)
    counts = {status: sum(c["status"] == status for c in claims) for status in sorted({c["status"] for c in claims})}
    write_json(directory / "manuscript_claims.json", {"schema_version": 1,
        "manuscript_sha256": manifest["manuscript_sha256"], "scope": manifest["scope"],
        "status_counts": counts, "claims": claims})
    lines = ["# Manuscript claim coverage", "", manifest["scope"], "",
             "`verified_match` is a saved-evidence comparison at manuscript precision. `corrected_value` requires author review. Missing values remain NA. Method descriptions are not measured results.", "",
             "| Claim | Manuscript | Recomputed | Status | Output field | Locations |", "|---|---|---|---|---|---|"]
    for c in claims:
        where = "; ".join(l.get("label", l["section"]) + ":" + l["lines"] for l in c["locations"])
        def cell(v):
            return str(v if v is not None else "NA").replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join(cell(v) for v in [c["id"], c["manuscript_value"], c["recomputed_value"], c["status"], c.get("computed_path"), where]) + " |")
    (directory / "manuscript_coverage.md").write_text("\n".join(lines)+"\n")
    return counts
