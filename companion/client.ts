/** Loopback only. SSE chunks are not model tokens. */
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 30_000);
try {
  const response = await fetch("http://127.0.0.1:18036/v1/chat/completions", {
    method: "POST",
    signal: controller.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "incident-reader",
      temperature: 0,
      max_tokens: 120,
      stream: true,
      messages: [
        {
          role: "user",
          content:
            "Checkout is recovering. Severity SEV2. Duration 18 minutes.",
        },
      ],
    }),
  });
  if (!response.ok || !response.body)
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  let buffer = "",
    output = "";
  const decoder = new TextDecoder();
  for await (const bytes of response.body) {
    buffer += decoder.decode(bytes, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 1);
      if (!line.startsWith("data:") || line.slice(5).trim() === "[DONE]")
        continue;
      const event = JSON.parse(line.slice(5));
      if (event.error) throw new Error(JSON.stringify(event.error));
      const text = event.choices?.[0]?.delta?.content ?? "";
      output += text;
      process.stdout.write(text);
    }
  }
  const parsed = JSON.parse(output);
  const fields = ["service", "severity", "status", "duration_minutes"];
  if (
    !parsed ||
    Array.isArray(parsed) ||
    Object.keys(parsed).sort().join() !== fields.sort().join()
  )
    throw new Error("Invalid record shape");
  if (
    parsed.duration_minutes !== null &&
    (!Number.isInteger(parsed.duration_minutes) || parsed.duration_minutes < 0)
  )
    throw new Error("Invalid duration");
  for (const [key, allowed] of Object.entries({
    service: [
      "checkout",
      "billing",
      "search",
      "inventory",
      "auth",
      "notifications",
      "gateway",
      "catalog",
    ],
    severity: ["SEV1", "SEV2", "SEV3", "SEV4"],
    status: ["investigating", "identified", "recovering", "resolved"],
  })) {
    if (parsed[key] !== null && !allowed.includes(parsed[key]))
      throw new Error(`Invalid ${key}`);
  }
  console.log("\nSchema valid. This does not establish factual correctness.");
} finally {
  clearTimeout(timeout);
}
