import http.server
import socketserver
import json
import os

from src.strategy.orderflow import OrderflowSignalStore, parse_orderflow_payload

PORT = 8000

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    orderflow_store = OrderflowSignalStore()

    def do_GET(self):
        if self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            state_file = 'bot_states.json'
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(json.dumps({"status": "waiting_for_bot"}).encode())
        elif self.path == '/api/panic':
            with open('panic.signal', 'w') as f:
                f.write('PANIC')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "panic_sent"}')
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/api/orderflow':
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

if __name__ == "__main__":
    print(f"Dashboard serving at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        httpd.serve_forever()
