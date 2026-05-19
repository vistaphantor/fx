from io import BytesIO

from dashboard_server import DashboardHandler
from src.strategy.orderflow import OrderflowSignalStore


class _Handler(DashboardHandler):
    def __init__(self, body: bytes, store):
        self.rfile = BytesIO(body)
        self.wfile = BytesIO()
        self.headers = {"Content-Length": str(len(body))}
        self.path = "/api/orderflow"
        self.command = "POST"
        self.request_version = "HTTP/1.1"
        self.orderflow_store = store
        self.responses = []

    def send_response(self, code, message=None):
        self.responses.append(code)

    def send_header(self, keyword, value):
        pass

    def end_headers(self):
        pass

    def log_message(self, format, *args):
        pass


def test_dashboard_accepts_orderflow_webhook(tmp_path):
    store = OrderflowSignalStore(tmp_path / "orderflow_state.json")
    body = b'{"symbol":"GC","target_symbol":"XAUUSD","delta":-900,"cvd_slope":-0.5}'
    handler = _Handler(body, store)

    handler.do_POST()

    assert handler.responses[0] == 200
    assert store.latest_for("XAUUSD").source_symbol == "GC"
