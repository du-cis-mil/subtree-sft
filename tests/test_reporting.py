"""Regression tests for manuscript omissions, schema rejection and honest completeness."""
import copy
import csv
import importlib.util
import json
import math
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from test_suite import fixture, report_audits, ROOT
from soict_ir.artifacts import validate_inputs, validate_figure_assets, make_figures, MAIN
from soict_ir.protocol import CONDITIONS, expected_sources, object_sha256, file_sha256
from soict_ir.reporting import generate_report, validate_artifact_files, emit_table
from soict_ir.data_audit import audit


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.tree, self.inputs = fixture(self.base)
        self.paths = report_audits(self.tree, self.inputs, self.base)
        self.data = json.loads(self.paths['corpus_audit'].read_text())
        self.training = json.loads(self.paths['training_audit'].read_text())
        self.granularity = self.data['granularity']

    def validate(self, tree=None, training=None, data=None):
        validate_inputs(tree if tree is not None else self.granularity,
            training if training is not None else self.training,
            data if data is not None else self.data, self.data['tree_identity'], 2)

    def test_synthetic_corpus_definitions_and_path_mapping(self):
        # Repeated title within one query is one target, while the same title in
        # another query remains another within-source target.
        row = json.loads((self.inputs/'toy_0.json').read_text())
        row['inspiration'] *= 2
        test = self.base/'duplicate-test.jsonl'
        test.write_text(json.dumps(row)+'\n'+json.dumps(json.loads((self.inputs/'toy_1.json').read_text()))+'\n')
        result = audit(self.base/'train.jsonl', test, self.tree)
        c = result['counts']
        self.assertEqual((c['test_inspiration_seen'],c['within_source_deduplicated_titles'],c['distinct_matched_leaf_papers']), (3,2,1))
        self.assertEqual((c['test_inspiration_added'], c['unmatched_annotations'],c['duplicate_matched_annotations']), (0,0,1))
        self.assertEqual(list(result['parent_counts'].values()), [2]*5)

    def test_empty_missing_depth_and_invalid_counts_are_rejected(self):
        self.validate()
        for mutation in ['empty','missing_depth','missing_field','negative','wrong_identity']:
            bad = copy.deepcopy(self.granularity)
            if mutation == 'empty': bad['depths'] = {}
            if mutation == 'missing_depth': del bad['depths']['4']
            if mutation == 'missing_field': del bad['depths']['0']['mean_children']
            if mutation == 'negative': bad['num_papers'] = -1
            if mutation == 'wrong_identity': bad['tree_identity']['papers'] = 'f'*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.validate(tree=bad)
        training = copy.deepcopy(self.training); training['shared_sources'] = 500
        with self.assertRaises(ValueError): self.validate(training=training)
        data = copy.deepcopy(self.data); data['counts']['matched_targets_within_source'] = 999
        with self.assertRaises(ValueError): self.validate(data=data)

    def test_no_missing_file_is_listed_as_generated(self):
        f = self.base/'exists.csv'; f.write_text('x\n1\n')
        artifacts, missing = validate_artifact_files(self.base, {'body:statistics':['exists.csv','absent.json']})
        self.assertEqual(artifacts, {'body:statistics':['exists.csv']})
        self.assertEqual(missing['body:statistics']['files'], ['absent.json'])

    def complete_fixture(self):
        # This test isolates report generation. The seven-condition execution
        # path itself is exercised separately with a network-blocked extractor.
        ids = ['synthetic_'+str(i) for i in range(1658)]
        population = expected_sources(ids)
        values = [{'source_id':sid,'correct':[True]*5,'decisions':[
            {'node_id':'n'+str(d),'level':d,'candidate_count':2,'is_correct':True,
             'format_error':False,'best_correct_rank':1} for d in range(5)]} for sid in ids]
        rows = {n:copy.deepcopy(values) for n in CONDITIONS}
        data = copy.deepcopy(self.data); c=data['counts']
        for key in ['test_rows','test_inspiration_seen','within_source_deduplicated_titles','matched_targets_within_source','sources_with_mapped_target']:
            c[key]=len(ids)
        data['parent_counts']={str(d):len(ids) for d in range(5)}
        data['source_sets']['original_ids_sha256']=population['source_ids_sha256']
        # Bind the count/hash to an actual synthetic test file, without copying
        # any historical source IDs or scores into the fixture.
        test = self.base/'full-test.jsonl'
        test.write_text(''.join(json.dumps({'source_id':sid,'inspiration':[{'found_title':'Synthetic 0'}]})+'\n' for sid in ids))
        data['inputs']['source_test_sha256']=file_sha256(test)
        p=self.base/'synthetic-corpus.json'; p.write_text(json.dumps(data))
        provenance={'kind':'suite results','plan':{'tree_identity':data['tree_identity'],'tree_profile':'rebuilt','mode':'paper','profile':'historical'},'weight_identity_verified':False}
        kwargs=dict(experiments=ROOT/'configs/experiments.json', suite_dir=self.base/'synthetic-suite',
                    tree_summary=self.paths['tree_summary'], training_audit=self.paths['training_audit'], corpus_audit=p,
                    iterations=10)
        return rows, population, provenance, kwargs

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Optional analysis dependency unavailable')
    def test_complete_synthetic_suite_reports_success_without_historical_claim(self):
        rows, population, provenance, kwargs = self.complete_fixture()
        with patch('soict_ir.reporting.suite_rows', return_value=(rows,population,{},provenance)):
            coverage=generate_report(self.base/'synthetic-report',**kwargs)
        self.assertTrue(coverage['paper_complete'])
        self.assertTrue(coverage['conditions_complete'])
        self.assertTrue(coverage['report_artifacts_complete'])
        self.assertFalse(coverage['historical_numeric_match'])
        self.assertEqual(coverage['missing'],{})
        claims=json.loads((self.base/'synthetic-report/manuscript_claims.json').read_text())['claims']
        self.assertTrue(all(not c['historical_value_verified'] for c in claims))

    def test_valid_but_incomplete_target_mapping_cannot_complete_paper(self):
        rows, population, provenance, kwargs = self.complete_fixture()
        data = json.loads(kwargs['corpus_audit'].read_text())
        data['counts']['sources_with_mapped_target'] = 1657
        data['counts']['matched_targets_within_source'] = 1657
        data['counts']['unmatched_annotations'] = 1
        data['counts']['distinct_test_normalized_titles'] = 2
        data['parent_counts'] = {str(d):1657 for d in range(5)}
        test = self.base/'full-test.jsonl'
        records = [json.loads(line) for line in test.read_text().splitlines()]
        records[0]['inspiration'][0]['found_title'] = 'Unmatched synthetic title'
        test.write_text(''.join(json.dumps(r)+'\n' for r in records))
        data['inputs']['source_test_sha256'] = file_sha256(test)
        kwargs['corpus_audit'].write_text(json.dumps(data))
        with patch('soict_ir.reporting.suite_rows', return_value=(rows,population,{},provenance)):
            with self.assertRaisesRegex(ValueError, 'mapping|target'):
                generate_report(self.base/'incomplete-strict', **kwargs)
            coverage = generate_report(self.base/'incomplete-partial', **kwargs,
                                       allow_partial=True, skip_statistics=True)
        self.assertFalse(coverage['paper_complete'])
        self.assertIn('target_mapping', coverage['missing'])

    def test_missing_depth_in_audit_is_a_partial_gap(self):
        rows, population, provenance, kwargs = self.complete_fixture()
        data = json.loads(kwargs['corpus_audit'].read_text())
        data['parent_counts']['4'] = 1657
        data['counts']['distinct_test_normalized_titles'] = 2
        data['counts']['distinct_matched_leaf_papers'] = 2
        test = self.base/'full-test.jsonl'
        records = [json.loads(line) for line in test.read_text().splitlines()]
        records[0]['inspiration'][0]['found_title'] = 'Synthetic 4'  # leaf at depth 4
        test.write_text(''.join(json.dumps(r)+'\n' for r in records))
        data['inputs']['source_test_sha256'] = file_sha256(test)
        kwargs['corpus_audit'].write_text(json.dumps(data))
        with patch('soict_ir.reporting.suite_rows', return_value=(rows,population,{},provenance)):
            with self.assertRaisesRegex(ValueError, 'depth|D4'):
                generate_report(self.base/'depth-strict', **kwargs)
            coverage = generate_report(self.base/'depth-partial', **kwargs,
                                       allow_partial=True, skip_statistics=True)
        self.assertFalse(coverage['paper_complete'])
        self.assertIn('depth_coverage:D4', coverage['missing'])

    def test_concept_asset_hash_changes_are_rejected(self):
        import shutil
        assets = validate_figure_assets(ROOT)
        self.assertEqual(len(assets['assets']),2)
        self.assertFalse(assets['redistribution_ready'])
        target = self.base/'tampered'
        shutil.copytree(ROOT/'docs/figures',target/'docs/figures')
        p = target/assets['assets'][0]['path']
        raw = p.read_bytes(); p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaisesRegex(ValueError,'signature/hash mismatch'):
            validate_figure_assets(target)

    def check_figure_boundaries(self, values):
        categories = ['D0','D1','D2','D3','D4','Overall']
        accuracy = {name:dict(zip(categories, values[i::3])) for i,name in enumerate(MAIN)}
        target = self.base/'figure-boundary'; target.mkdir()
        rows = make_figures(target, accuracy, validate_figure_assets(ROOT), ROOT)
        emit_table(target, 'depthwise', rows)
        with (target/'depthwise.csv').open() as f:
            csv_rows = list(csv.DictReader(f))
        svg = ET.parse(target/'depthwise_figure.svg')
        bars = [e for e in svg.iter() if 'data-series' in e.attrib]
        labels = [e for e in svg.iter() if 'data-value-label' in e.attrib]
        self.assertEqual(len(bars), sum(v is not None for v in values))
        self.assertEqual(len(labels), len(bars))
        expected = {(c,n):accuracy[n][c] for c in categories for n in MAIN}
        for bar in bars:
            a = bar.attrib; key = (a['data-category'],a['data-series'])
            x,y,w,h = (float(a[k]) for k in ('x','y','width','height'))
            self.assertTrue(all(math.isfinite(v) for v in (x,y,w,h)))
            self.assertGreaterEqual(h,0); self.assertGreaterEqual(y,105)
            self.assertLessEqual(x+w,1050); self.assertLessEqual(y+h,465)
            self.assertEqual(float(a['data-value']),expected[key])
            record = next(r for r in csv_rows if r['depth']==key[0])
            self.assertEqual(float(record[key[1]]),expected[key])
        for label in labels:
            a=label.attrib; key=(a['data-category'],a['data-value-label'])
            self.assertGreaterEqual(float(a['y'])-12,70)  # clear of legend (y=25..43)
            self.assertLessEqual(float(a['y']),550)
            self.assertEqual(label.text, f'{expected[key]:.2f}')
        tex = (target/'depthwise_figure.tex').read_text()
        ymax = float(re.search(r'ymax=([0-9.]+)',tex).group(1))
        self.assertGreater(ymax,100)
        self.assertIn('ytick={0,20,40,60,80,100}',tex)
        plots = re.findall(r'\\addplot coordinates \{([^}]+)\}',tex)
        self.assertEqual(len(plots),3)
        for name,plot in zip(MAIN,plots):
            coords = re.findall(r'\(([^,]+),([^)]+)\)',plot)
            self.assertEqual([c for c,v in coords],categories)
            for category,value in coords:
                wanted = accuracy[name][category]
                if wanted is None:
                    self.assertEqual(value,'nan')
                    self.assertEqual(next(r for r in csv_rows if r['depth']==category)[name],'NA')
                else:
                    self.assertEqual(float(value),wanted); self.assertLessEqual(float(value),ymax)
        if None in values:
            self.assertIn('NA',tex)
            self.assertTrue(any(e.text=='NA' for e in svg.iter()))

    def test_figure_all_100_percent_stays_inside_canvas(self):
        self.check_figure_boundaries([100.]*18)

    def test_figure_mixed_boundary_values_match_csv_and_tex(self):
        self.check_figure_boundaries(([0.,65.,80.,100.]*5)[:18])

    def test_figure_na_preserves_remaining_series_values(self):
        self.check_figure_boundaries(([None,0.,65.,80.,100.]*4)[:18])

    def test_figure_invalid_accuracy_is_rejected_without_clamping(self):
        for value in (-1,101,float('nan'),float('inf'),True):
            accuracy = {n:{c:50. for c in ['D0','D1','D2','D3','D4','Overall']} for n in MAIN}
            accuracy[MAIN[0]]['D0']=value
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_figures(self.base,accuracy,validate_figure_assets(ROOT),ROOT)

    def test_output_inventory_is_complete_and_contains_no_reference_values(self):
        from soict_ir.claims import validate_manifest
        manifest=json.loads((ROOT/'configs/output_schema.json').read_text())
        validate_manifest(manifest)
        self.assertEqual(len(manifest['claims']),159)
        self.assertEqual(len({c['id'] for c in manifest['claims']}),159)
        self.assertTrue(all('manuscript_value' not in c for c in manifest['claims']))
        for mutation in ('subset','duplicate','path','location','label','type'):
            bad=copy.deepcopy(manifest)
            if mutation=='subset':bad['claims']=bad['claims'][:10]
            if mutation=='duplicate':bad['claims'][1]['id']=bad['claims'][0]['id']
            if mutation=='path':bad['claims'][0]['computed_path']='accuracy/child_child/D0'
            if mutation=='location':bad['claims'][0]['locations'][0]['lines']='1'
            if mutation=='label':bad['tables'].pop()
            if mutation=='type':bad['claims'][0]['value_type']='str'
            path=self.base/(mutation+'.json');path.write_text(json.dumps(bad))
            for partial in (False,True):
                with self.subTest(mutation=mutation,partial=partial), self.assertRaisesRegex(ValueError,'inventory'):
                    generate_report(self.base/(mutation+str(partial)),suite_dir='not-read',
                        experiments=ROOT/'configs/experiments.json',claim_manifest=path,allow_partial=partial)
        path=self.base/'formatted.json';path.write_text(json.dumps(manifest,sort_keys=True))
        validate_manifest(json.loads(path.read_text()))

    def test_no_implicit_historical_results(self):
        for partial in (False,True):
            with self.assertRaisesRegex(ValueError,'Supply --suite-dir'):
                generate_report(self.base/str(partial),experiments=ROOT/'configs/experiments.json',allow_partial=partial)

    def test_synthetic_audit_boundaries_in_strict_and_partial(self):
        rows,population,provenance,kwargs=self.complete_fixture()
        initial=json.loads(kwargs['corpus_audit'].read_text())
        for field,value in [('sources_with_mapped_target',1659),('distinct_matched_leaf_papers',999999),
                            ('matched_targets_within_source',True),('train_source_added',999999)]:
            bad=copy.deepcopy(initial);bad['counts'][field]=value
            kwargs['corpus_audit'].write_text(json.dumps(bad))
            for partial in (False,True):
                with self.subTest(field=field,partial=partial), patch('soict_ir.reporting.suite_rows',return_value=(rows,population,{},provenance)), self.assertRaises(ValueError):
                    generate_report(self.base/(field+str(partial)),**kwargs,allow_partial=partial,skip_statistics=True)
        kwargs['corpus_audit'].write_text(json.dumps(initial))
        for model in ('paper_selection','child_subtree'):
            bad=copy.deepcopy(self.training);bad[model+'_examples']=10
            path=self.base/(model+'.json');path.write_text(json.dumps(bad))
            for partial in (False,True):
                changed=dict(kwargs,training_audit=path)
                with self.subTest(model=model,partial=partial), patch('soict_ir.reporting.suite_rows',return_value=(rows,population,{},provenance)), self.assertRaisesRegex(ValueError,'10000'):
                    generate_report(self.base/(model+str(partial)),**changed,allow_partial=partial,skip_statistics=True)
        for index,value in enumerate((None,True,10,10000.0)):
            config=json.loads((ROOT/'configs/experiments.json').read_text());config['training']['examples_per_model']=value
            path=self.base/('config'+str(index)+'.json');path.write_text(json.dumps(config))
            for partial in (False,True):
                changed=dict(kwargs,experiments=path,project_root=ROOT)
                with self.subTest(budget=value,partial=partial), patch('soict_ir.reporting.suite_rows',return_value=(rows,population,{},provenance)), self.assertRaisesRegex(ValueError,'10000'):
                    generate_report(self.base/('budget'+str(index)+str(partial)),**changed,allow_partial=partial,skip_statistics=True)

    def test_synthetic_partial_outputs_na_and_skipped_statistics(self):
        rows,population,provenance,kwargs=self.complete_fixture()
        del rows['repro_child']
        with patch('soict_ir.reporting.suite_rows',return_value=(rows,population,{'repro_child':'missing synthetic condition'},provenance)):
            with self.assertRaisesRegex(ValueError,'repro_child'):
                generate_report(self.base/'strict',**kwargs,skip_statistics=True)
            cov=generate_report(self.base/'partial',**kwargs,allow_partial=True,skip_statistics=True)
        self.assertFalse(cov['paper_complete'])
        self.assertFalse(cov['report_artifacts_complete'])
        self.assertFalse(cov['historical_numeric_match'])
        self.assertNotIn('main_paired_statistics.json',cov['artifacts']['body:statistics'])
        claims=json.loads((self.base/'partial/manuscript_claims.json').read_text())['claims']
        missing=[c for c in claims if c['id'].startswith('accuracy.repro_child.')]
        self.assertEqual(len(missing),6)
        self.assertTrue(all(c['recomputed_value'] is None for c in missing))
        self.assertTrue(all('manuscript_value' not in c for c in claims))
        bars=[e for e in ET.parse(self.base/'partial/depthwise_figure.svg').iter() if 'data-series' in e.attrib]
        self.assertEqual(len(bars),18)
        self.assertIn('not provided',(self.base/'partial/ci_comparison.csv').read_text())


if __name__ == '__main__': unittest.main()
