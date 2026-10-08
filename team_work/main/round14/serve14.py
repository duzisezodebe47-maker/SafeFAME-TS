"""第14轮本地演示：新时间段结果、来源证据和严格数值推理。"""
import argparse,json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
from r14_common import *
from predictor14 import forecast,example
LOCK=threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def respond(self,status,payload,mime='application/json; charset=utf-8'):
        data=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8');self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        try:
            url=urlsplit(self.path)
            if url.path in ('/','/app.js','/style.css'):
                filename='index.html' if url.path=='/' else url.path[1:];return self.respond(200,(HERE/'web'/filename).read_bytes(),{'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[filename])
            if url.path=='/api/summary':return self.respond(200,read(HERE/'results/final/dashboard.json'))
            if url.path in ('/api/task','/api/example'):
                h=int(parse_qs(url.query).get('horizon',['4'])[0]);require(h in cfg()['horizons'],'步长无效')
                return self.respond(200,example(h) if url.path.endswith('example') else read(HERE/f'results/h{h}/demo.json'))
            self.respond(404,{'error':'路径不存在'})
        except ValueError as e:self.respond(400,{'error':str(e)})
        except Exception:self.respond(503,{'error':'请检查D盘实验文件和验收结果是否完整'})
    def do_POST(self):
        if self.path!='/api/predict':return self.respond(404,{'error':'路径不存在'})
        try:
            if self.headers.get_content_type()!='application/json':raise ValueError('仅接受JSON请求')
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':raise ValueError('请求来源不匹配')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=65536:raise ValueError('请求长度无效')
            value=json.loads(self.rfile.read(length))
            with LOCK:result=forecast(value)
            self.respond(200,result)
        except (ValueError,UnicodeDecodeError) as e:self.respond(400,{'error':str(e)})
        except Exception:self.respond(503,{'error':'本地模型校验或推理失败'})
if __name__=='__main__':
    import torch
    torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8874);a=ap.parse_args();server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);print(f'Local: http://127.0.0.1:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
