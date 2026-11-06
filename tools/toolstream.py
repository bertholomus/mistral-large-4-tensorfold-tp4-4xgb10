#!/usr/bin/env python3
"""toolstream.py [--port 8561]: a tool call asked for streamed and not, at temperature 0: the streamed argument pieces must
join to JSON that parses and equals the plain reply's call. Standard library only."""
import argparse, http.client, json, sys
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8561)
ap.add_argument("--model", default="ml4")
a = ap.parse_args()
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city", "unit"]}}},
         {"type": "function", "function": {"name": "find_flights", "description": "Flights between two cities on a date.",
          "parameters": {"type": "object", "properties": {"from_city": {"type": "string"}, "to_city": {"type": "string"},
                                                           "date": {"type": "string"}}, "required": ["from_city", "to_city", "date"]}}}]
CASES = [("What is the weather in Tokyo in celsius? Use the tool.", "get_weather"),
         ("Find flights from Kyoto to Oslo on 2026-11-07. Use the tool.", "find_flights")]

def post(body):
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    return c.getresponse()

bad = 0
for text, want in CASES:
    body = {"model": a.model, "messages": [{"role": "user", "content": text}], "tools": TOOLS, "temperature": 0, "max_tokens": 1024,
            "reasoning_effort": "low"}
    plain = json.loads(post(body).read())
    call = plain["choices"][0]["message"]["tool_calls"][0]["function"]
    res = post({**body, "stream": True})
    names, args, pieces = "", "", 0
    for raw in res:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        for ch in json.loads(line[5:]).get("choices") or []:
            for t in (ch.get("delta") or {}).get("tool_calls") or []:
                f = t.get("function") or {}
                names += f.get("name") or ""
                if f.get("arguments"):
                    args += f["arguments"]
                    pieces += 1
    ok = names == want == call["name"] and json.loads(args) == json.loads(call["arguments"])
    bad += not ok
    print(f"[toolstream] {want}: {'PASS' if ok else 'FAIL'} name {names!r}, {pieces} pieces, args {args!r}, plain {call['arguments']!r}", flush=True)
print(f"[toolstream] {len(CASES) - bad} of {len(CASES)} pass", flush=True)
sys.exit(1 if bad else 0)
