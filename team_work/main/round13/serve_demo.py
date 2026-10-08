"""只在本机监听的课程演示服务，无外部上传、无任意文件访问。"""
from __future__ import annotations
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
from common import HERE,read
from predictor import forecast,example,validate_task

LOCK=threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def send(self,status,content,kind='application/json; charset=utf-8'):
        data=content if isinstance(content,bytes) else json.dumps(content,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        try:
            u=urlsplit(self.path)
            if u.path in ('/','/app.js','/style.css'):
                name='index.html' if u.path=='/' else u.path[1:];kind={'index.html':'text/html','app.js':'text/javascript','style.css':'text/css'}[name];return self.send(200,(HERE/'web'/name).read_bytes(),kind+'; charset=utf-8')
            if u.path=='/api/summary':return self.send(200,read(HERE/'results/final/dashboard.json'))
            if u.path in ('/api/example','/api/replay'):
                task=parse_qs(u.query).get('task',[''])[0];validate_task(task)
                return self.send(200,example(task) if u.path.endswith('example') else read(HERE/'results/demo'/f'{task}.json'))
            self.send(404,{'error':'路径不存在'})
        except ValueError as e:self.send(400,{'error':str(e)})
        except Exception:self.send(503,{'error':'本地模型或验收文件不可用，请先运行验收入口并检查D盘数据目录'})
    def do_POST(self):
        if self.path!='/api/predict':return self.send(404,{'error':'路径不存在'})
        try:
            # JSON-only and same-origin requests prevent ordinary cross-site form submissions.
            if self.headers.get_content_type()!='application/json':raise ValueError('仅接受application/json')
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':raise ValueError('来源不匹配')
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=65536:raise ValueError('请求长度无效或超过64KB')
            payload=json.loads(self.rfile.read(n))
            with LOCK:result=forecast(payload)
            self.send(200,result)
        except (ValueError,UnicodeDecodeError) as e:self.send(400,{'error':str(e)})
        except Exception:self.send(503,{'error':'模型推理失败，请检查本地模型及依赖完整性'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8873);a=ap.parse_args();server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);print(f'Local: http://127.0.0.1:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
