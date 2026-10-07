"""主控接管的隔离、退化和校准评分回归测试，临时数据只写入 D 盘。"""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('master12',HERE/'run_master.py')
master=importlib.util.module_from_spec(spec); spec.loader.exec_module(master)
ROOT=HERE.parents[2]
PROJECT=ROOT.parent
PACKAGE=PROJECT/'主控接收数据'/'主控接管_20261007'/'SocialGood_h3_f1'
SPEC=ROOT/'team_work/main/round2/split_spec_v2.json'


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=PROJECT/'主控接收数据')
        self.package=Path(self.temp.name)/'package'
        shutil.copytree(PACKAGE,self.package)
    def tearDown(self): self.temp.cleanup()
    def reseal(self):
        m=json.loads((self.package/'MANIFEST.json').read_text(encoding='utf-8'))
        m['files']=master.inventory(self.package)
        master.write_json(self.package/'MANIFEST.json',m)
    def test_valid_package(self): self.assertEqual(master.verify_package(self.package,SPEC)['status'],'PASS')
    def test_test_truth_rejected(self):
        np.save(self.package/'test_features/targets.npy',np.zeros((89,3))); self.reseal()
        with self.assertRaisesRegex(ValueError,'forbidden'): master.verify_package(self.package,SPEC)
    def test_test_history_rejected(self):
        np.save(self.package/'test_features/numeric_history.npy',np.zeros((89,24))); self.reseal()
        with self.assertRaisesRegex(ValueError,'forbidden'): master.verify_package(self.package,SPEC)
    def test_frequency_rejected(self):
        np.save(self.package/'test_features/frequency.npy',np.zeros((89,10))); self.reseal()
        with self.assertRaisesRegex(ValueError,'forbidden'): master.verify_package(self.package,SPEC)
    def test_tamper_rejected(self):
        p=self.package/'selection_fit/targets.npy'; p.write_bytes(p.read_bytes()+b'X')
        with self.assertRaisesRegex(ValueError,'inventory'): master.verify_package(self.package,SPEC)
    def test_mapping_shift_rejected(self):
        p=self.package/'selection_fit/metadata.csv'; m=master.pd.read_csv(p); m.loc[0,'origin_index']+=1
        m.to_csv(p,index=False,lineterminator='\n'); self.reseal()
        with self.assertRaisesRegex(ValueError,'mapping'): master.verify_package(self.package,SPEC)
    def test_bundle_signature_rejected(self):
        p=self.package/'MANIFEST.json'; m=json.loads(p.read_text()); m['bundle_signature']='0'*64; master.write_json(p,m)
        with self.assertRaisesRegex(ValueError,'anchor'): master.verify_package(self.package,SPEC)
    def test_spec_hash_rejected(self):
        p=self.package/'MANIFEST.json'; m=json.loads(p.read_text()); m['split_spec_sha256']='0'*64; master.write_json(p,m)
        with self.assertRaisesRegex(ValueError,'anchor'): master.verify_package(self.package,SPEC)


class ModelTests(unittest.TestCase):
    def test_zero_text_and_constant_quality_equal_N(self):
        meta,f=master.load_selection(PACKAGE,'proxy')
        train,cal,dec=[f.slice(meta.segment.eq(s).to_numpy()) for s in master.SELECTION]
        base=master._fit('N',train,cal)
        for name in ('N+Q','N+S+Q','N+S+Q+SF'):
            model=master._fit(name,train,cal)
            self.assertEqual('+'.join(b.name for b in model.branches),'N')
            np.testing.assert_array_equal(model.predict(master.infer_view(dec)),base.predict(master.infer_view(dec)))
    def test_selection_loader_does_not_open_test_arrays(self):
        original=np.load
        def guarded(path,*args,**kwargs):
            self.assertIn('selection_fit',str(path))
            return original(path,*args,**kwargs)
        with patch.object(np,'load',guarded): master.load_selection(PACKAGE,'proxy')
    def test_inference_view_has_no_targets(self):
        _,f=master.load_selection(PACKAGE,'proxy')
        v=master.infer_view(f)
        self.assertIsNone(v.targets); self.assertIsNone(v.targets_standardized)
    def test_bootstrap_reproducible_and_zero(self):
        self.assertEqual(master.block_ci(np.arange(40),3),master.block_ci(np.arange(40),3))
        self.assertEqual(master.block_ci(np.zeros(40),3),[0.0,0.0])
    def test_calibration_scored_before_refit_including_cal(self):
        # This order is checked by real phase output separately; retain a structural guard.
        text=(HERE/'run_master.py').read_text(encoding='utf-8')
        self.assertIn('model.refit(train); calibration_pred[name]=model.predict(infer_view(cal))',text)
        self.assertIn('cal_mse={c:mse(calibration_pred[c]',text)


if __name__=='__main__':
    with threadpool_limits(limits=1): unittest.main(verbosity=2)
