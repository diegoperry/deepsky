export default {
  async fetch(request, env) {
    const expected = `Bearer ${env.DEEPSKY_STORAGE_TOKEN}`;
    if (!env.DEEPSKY_STORAGE_TOKEN || request.headers.get("Authorization") !== expected) {
      return new Response("Unauthorized", { status: 401 });
    }

    const url = new URL(request.url);
    const bucket = env.DEEPSKY_BUCKET;
    if (!bucket) {
      return new Response("Missing DEEPSKY_BUCKET binding", { status: 500 });
    }

    if (url.pathname === "/list" && request.method === "GET") {
      const prefix = url.searchParams.get("prefix") || "";
      const listed = await bucket.list({ prefix, limit: 1 });
      return Response.json({ ok: true, count: listed.objects.length });
    }

    if (url.pathname === "/delete-prefix" && request.method === "POST") {
      const body = await request.json();
      const prefix = body.prefix || "";
      let cursor;
      do {
        const listed = await bucket.list({ prefix, cursor, limit: 1000 });
        await Promise.all(listed.objects.map((item) => bucket.delete(item.key)));
        cursor = listed.truncated ? listed.cursor : undefined;
      } while (cursor);
      return Response.json({ ok: true });
    }

    if (!url.pathname.startsWith("/objects/")) {
      return new Response("Not found", { status: 404 });
    }

    const key = decodeURIComponent(url.pathname.slice("/objects/".length));
    if (!key) {
      return new Response("Missing key", { status: 400 });
    }

    if (request.method === "PUT") {
      await bucket.put(key, request.body, {
        httpMetadata: {
          contentType: request.headers.get("Content-Type") || "application/octet-stream",
        },
      });
      return Response.json({ ok: true });
    }

    if (request.method === "GET") {
      const object = await bucket.get(key);
      if (!object) {
        return new Response("Not found", { status: 404 });
      }
      return new Response(object.body, {
        headers: {
          "Content-Type": object.httpMetadata?.contentType || "application/octet-stream",
          "Cache-Control": "no-store",
        },
      });
    }

    if (request.method === "HEAD") {
      const object = await bucket.head(key);
      return new Response(null, { status: object ? 200 : 404 });
    }

    if (request.method === "DELETE") {
      await bucket.delete(key);
      return Response.json({ ok: true });
    }

    return new Response("Method not allowed", { status: 405 });
  },
};
