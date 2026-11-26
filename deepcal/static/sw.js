// Minimal service worker: makes DeepCal installable (PWA) and lets the app shell open offline.
// API calls are never cached, so meals, voice logs and auth always hit the server.
const CACHE = "deepcal-shell-v1";
const SHELL = ["/", "/static/index.html", "/manifest.json", "/static/icons/icon-192.png"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return;       // network only for the API
  e.respondWith(fetch(e.request).then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return r; })
                 .catch(() => caches.match(e.request).then(m => m || caches.match("/"))));   // network first, cache fallback
});
