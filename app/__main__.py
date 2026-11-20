"""python -m app [--host 127.0.0.1] [--port 8080]"""
import argparse
import logging


def main():
    ap = argparse.ArgumentParser(description="Drone Mission Replay demo server")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other devices in the LAN")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    print(f"Mission Replay: http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}")
    uvicorn.run("app.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
