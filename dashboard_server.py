import http.server
import json
import socketserver
from pathlib import Path
from urllib.parse import urlsplit

from src.strategy.orderflow import OrderflowSignalStore, parse_orderflow_payload

PORT = 8000
BASE_DIR = Path(__file__).resolve().parent


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    orderflow_store = OrderflowSignalStore()

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR if directory is None else directory), **kwargs)

    def do_GET(self):
        request_path = urlsplit(self.path).path
        if request_path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            state_file = BASE_DIR / 'bot_state.json'
            if state_file.exists():
                with state_file.open('r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(json.dumps({"status": "waiting_for_bot"}).encode())
        elif request_path == '/api/panic':
            with (BASE_DIR / 'panic.signal').open('w', encoding='utf-8') as f:
                f.write('PANIC')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "panic_sent"}')
        else:
            super().do_GET()

    def do_POST(self):
        request_path = urlsplit(self.path).path
        if request_path != '/api/orderflow':
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error": "not_found"}')
            return

        try:
            length = int(self.headers.get('Content-Length', '0') or '0')
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            signal = parse_orderflow_payload(payload)
            self.orderflow_store.record(signal)
        except Exception as exc:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())
            return

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "symbol": signal.symbol}).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.end_headers()

if __name__ == "__main__":
    print(f"Dashboard serving at http://localhost:{PORT}")
    with ReusableTCPServer(("", PORT), DashboardHandler) as httpd:
        httpd.serve_forever()
