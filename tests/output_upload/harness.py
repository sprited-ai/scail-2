import json, time, threading, subprocess, urllib.request
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
puts=[]
class Handler(BaseHTTPRequestHandler):
 def do_PUT(self):
  data=self.rfile.read(int(self.headers['Content-Length']))
  puts.append({'path':self.path,'size':len(data),'contains_payload':b'test-mp4-placeholder' in data or b'{"seed":42}' in data})
  self.send_response(201);self.send_header('Location','http://127.0.0.1:15088/files/'+self.path.rsplit('/',1)[-1]);self.end_headers()
 def log_message(self,*args):pass
threading.Thread(target=HTTPServer(('127.0.0.1',15088),Handler).serve_forever,daemon=True).start()
log=open('/check/server.log','w')
p=subprocess.Popen(['python','-m','cog.server.http','--host','127.0.0.1'],stdout=log,stderr=subprocess.STDOUT,cwd='/check')
try:
 for _ in range(90):
  try:
   with urllib.request.urlopen('http://127.0.0.1:5000/health-check',timeout=1) as r:
    h=json.load(r)
   if h.get('status')=='READY':break
  except Exception:pass
  if p.poll() is not None:raise RuntimeError('server exit '+str(p.returncode))
  time.sleep(1)
 body={'input':{},'output_file_prefix':'http://127.0.0.1:15088/uploads/'}
 req=urllib.request.Request('http://127.0.0.1:5000/predictions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:j=json.load(r)
 print(json.dumps({'status':j.get('status'),'output':j.get('output'),'puts':puts},indent=2))
 Path('/check/result.json').write_text(json.dumps(j,indent=2))
 assert j['status']=='succeeded'
 assert len(puts)==2,puts
 assert all(x['contains_payload'] for x in puts),puts
 assert j['output']['video'].startswith('http://127.0.0.1:15088/uploads/')
 assert j['output']['metadata'].startswith('http://127.0.0.1:15088/uploads/')
 assert j['output'].get('frames') is None
 print('PASS: nested named output paths uploaded, no ZIP')
finally:
 p.terminate()
 try:p.wait(timeout=5)
 except subprocess.TimeoutExpired:p.kill()
