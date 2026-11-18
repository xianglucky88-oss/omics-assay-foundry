import argparse
from .api import ROUTES, dispatch
from .http_server import make_server

def main():
    parser = argparse.ArgumentParser(description="Local scientific data processing service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9840)
    args = parser.parse_args()
    with make_server(args.host, args.port, dispatch, ROUTES) as server:
        print(f"Listening on http://{server.server_address[0]}:{server.server_address[1]}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
