"""Offline checks for the relocated frozen data; no model or mall imports."""
import ast
import hashlib
import json
import unittest
from pathlib import Path

class DeliveryInputsTests(unittest.TestCase):
 def test_frozen_inputs_and_scripts(self):
  base=Path(__file__).parent
  hashes=json.loads((base/'data/input_hashes.json').read_text(encoding='utf8'))
  for name,digest in hashes.items():
   self.assertEqual(hashlib.sha256((base/'data'/name).read_bytes()).hexdigest(),digest)
  cases=json.loads((base/'data/semantic_cases.json').read_text(encoding='utf8'))
  self.assertEqual(len(cases),35)
  self.assertEqual(len({c[0] for c in cases}),7)
  for group in {c[0] for c in cases}:self.assertEqual(sum(c[0]==group for c in cases),5)
  for c in cases:self.assertEqual(len(c[2]),len(c[3]))
  self.assertEqual(len(json.loads((base/'data/pagination_holdout.json').read_text(encoding='utf8'))),3)
  for name in ('live_semantic_evaluation.py','diagnose_filter_extraction.py'):
   ast.parse((base/name).read_text(encoding='utf8'))
