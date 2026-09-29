// object-fit: contain mapping; reject letterboxing instead of clamping it.
export function mapPoint(clientX, clientY, rect, width = 414, height = 896) {
  const scale = Math.min(rect.width / width, rect.height / height);
  const w = width * scale, h = height * scale;
  const x = (clientX - rect.left - (rect.width - w) / 2) / w;
  const y = (clientY - rect.top - (rect.height - h) / 2) / h;
  return x >= 0 && x <= 1 && y >= 0 && y <= 1 ? {x, y} : null;
}

export class Capture {
  constructor() { this.cancel(); }
  cancel() { this.id = null; this.points = []; }
  start(id, point, now) {
    if (this.id !== null || !point) return false;
    this.id = id; this.started = now; this.points = [{...point, t: 0}];
    return true;
  }
  move(id, point, now) {
    if (id !== this.id) return false;
    const t = Math.round(now - this.started);
    if (!point || t > 5000 || this.points.length >= 510) { this.cancel(); return false; }
    if (t > this.points.at(-1).t) this.points.push({...point, t});
    return true;
  }
  finish(id, point, now) {
    if (id !== this.id || !point || now - this.started > 5000) { this.cancel(); return null; }
    const t = Math.max(30, Math.round(now - this.started));
    if (this.points.at(-1).t === t) this.points.pop();
    this.points.push({...point, t});
    const points = this.points;
    this.cancel();
    return points;
  }
}
