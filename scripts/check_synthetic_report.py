#!/usr/bin/env python3
"""Write a synthetic report for CPU validation; never an experimental result.

Reuses the regression fixture: report-layer suite input is mocked. The separate
seven-condition pipeline test exercises actual protocols with fake extraction.
"""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_reporting import ReportTests
from soict_ir.reporting import generate_report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--skip-statistics',action='store_true')
    args=parser.parse_args()
    test=ReportTests();test.setUp()
    try:
        rows,population,provenance,kwargs=test.complete_fixture()
        with patch('socket.create_connection',side_effect=AssertionError('Network forbidden')), \
             patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')), \
             patch('soict_ir.reporting.suite_rows',return_value=(rows,population,{},provenance)):
            coverage=generate_report(args.output_dir,**kwargs,skip_statistics=args.skip_statistics,
                                     allow_partial=args.skip_statistics)
        assert not coverage['historical_numeric_match']
        if not args.skip_statistics:
            assert coverage['paper_complete'] and coverage['report_artifacts_complete']
        print('SYNTHETIC CPU VALIDATION ONLY; report-layer suite input mocked; bootstrap 10, not evidence of a model run')
    finally:
        test.doCleanups()
