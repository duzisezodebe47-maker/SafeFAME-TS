"""真实输入、时序反馈及本地接口回归测试；不重新调参。"""
import unittest, tempfile, threading, urllib.request, urllib.error
from r14_common import *
from prepare_data import clean_assessment, release_times, parse_narrative
from predictor14 import validate, forecast, example
from adaptive import DelayedRiskMixer, causal_replay
from serve14 import Handler, ThreadingHTTPServer

class Contracts(unittest.TestCase):
    def test_clean_outlook(self):
        original,text=clean_assessment('National drought summary<br>Observed 25 drought areas.<br>Looking Ahead<br>future secret')
        self.assertNotIn('future',text);self.assertNotIn('25',text);self.assertIn('Observed',original)
    def test_dst_schedule(self):
        self.assertEqual(release_times('2024-01-02')[0].hour,13)
        self.assertEqual(release_times('2024-07-02')[0].hour,12)
        self.assertEqual(release_times('2024-07-02')[1],'official_schedule_inferred')
    def test_explicit_date_conservative(self):
        t,g=release_times('2024-07-02','2024-07-04')
        self.assertEqual(str(t),'2024-07-05 04:00:00+00:00');self.assertEqual(g,'document_explicit')
    def test_xml_period_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=LOCAL) as d:
            p=Path(d)/'USDM_20240102.xml'
            p.write_text('<root><forecastPeriod>2024-01-09</forecastPeriod><assessment>'+('drought soil conditions '*40)+'</assessment></root>',encoding='utf8')
            dump(p.with_suffix('.xml.meta.json'),{'sha256':sha(p),'url':'https://example.invalid','fetched_utc':'2026-10-08'})
            self.assertFalse(parse_narrative(p)['usable_content'])
            p.write_text('<broken>',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'raw text changed'):parse_narrative(p)
    def test_invalid_lengths_and_fields(self):
        for payload in ({'horizon':4,'history':[1]*51},{'horizon':4,'history':[1]*52,'text':'unverified'},{'horizon':True,'history':[1]*52},{'horizon':5,'history':[1]*52}):
            with self.subTest(payload=str(payload)[:40]),self.assertRaises(ValueError):validate(payload)
    def test_invalid_numbers(self):
        for v in (True,float('nan'),float('inf'),-1,501,'1'):
            with self.subTest(v=v),self.assertRaises(ValueError):validate({'horizon':4,'history':[v]*52})
    def test_actual_strict_forecasts(self):
        for h in (4,12):
            result=forecast({'horizon':h,'history':example(h)['history']});s=read(LOCAL/f'prepared/h{h}/scaler.json')
            saved=np.load(LOCAL/f'evaluation/h{h}/fixed_predictions.npz')['SelectedNumeric'][-1]*s['std']+s['mean']
            np.testing.assert_allclose(result['forecast_dsci'],saved,atol=1e-4);self.assertEqual(result['text_weight'],0)
    def test_future_feedback_rejected(self):
        m=DelayedRiskMixer(4,cfg()['adaptive'])
        with self.assertRaisesRegex(ValueError,'future'):m.observe(10,np.zeros(4),np.ones(4),np.zeros(4),13)
    def test_future_perturbation_invariance(self):
        origins=np.arange(60);base=np.zeros((60,4));direct=np.ones((60,4));a=np.ones((60,4));b=a.copy();b[30:]=100
        p,t=causal_replay(origins,base,direct,np.ones(60,bool),4,cfg()['adaptive'],lambda i:a[i])
        q,u=causal_replay(origins,base,direct,np.ones(60,bool),4,cfg()['adaptive'],lambda i:b[i])
        np.testing.assert_array_equal(p[:34],q[:34]);self.assertEqual(t[:34],u[:34])
    def test_http_contract(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(url+'/api/summary') as r:self.assertIn('metrics',json.load(r))
            with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(url+'/../../protocol.json')
            self.assertEqual(e.exception.code,404)
            for body,origin in [({'horizon':4,'history':[1]},url),({'horizon':4,'history':[1]*52},'https://other.invalid')]:
                req=urllib.request.Request(url+'/api/predict',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Origin':origin})
                with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
                self.assertEqual(e.exception.code,400)
            req=urllib.request.Request(url+'/api/predict',data=json.dumps({'horizon':4,'history':[1]*52}).encode(),headers={'Content-Type':'application/json','Origin':url})
            with urllib.request.urlopen(req) as r:np.testing.assert_allclose(json.load(r)['forecast_dsci'],[1.0]*4,atol=1e-12)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':
    import torch,io
    torch.set_num_threads(2);stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts));log=stream.getvalue();print(log)
    (HERE/'results/final/tests.log').write_text(log,encoding='utf8');dump(HERE/'results/final/tests.json',{'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'test_source_sha256':sha(Path(__file__))})
    sys.exit(0 if result.wasSuccessful() else 1)
