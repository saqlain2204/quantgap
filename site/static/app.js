// Sort the results table and draw the quality-vs-size chart as plain SVG. No libraries.
(function () {
  "use strict";

  // ---- sortable table
  var table = document.getElementById("results");
  if (table) {
    var ths = table.querySelectorAll("thead th");
    ths.forEach(function (th, col) {
      function sort() {
        var dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        ths.forEach(function (o) { o.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", dir);
        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[col].getAttribute("data-v"), y = b.cells[col].getAttribute("data-v");
          var nx = parseFloat(x), ny = parseFloat(y);
          var r = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : String(x).localeCompare(String(y));
          return dir === "ascending" ? r : -r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      }
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } });
    });
  }

  // ---- chart
  var holder = document.getElementById("chart");
  var dataEl = document.getElementById("chart-data");
  if (!holder || !dataEl) return;
  var pts = JSON.parse(dataEl.textContent);
  if (!pts.length) { holder.parentNode.style.display = "none"; return; }

  var NS = "http://www.w3.org/2000/svg";
  var W = 880, H = 420, M = { l: 70, r: 24, t: 16, b: 46 };
  var css = getComputedStyle(document.documentElement);
  var INK = css.getPropertyValue("--ink"), MUTED = css.getPropertyValue("--muted"),
      GRID = css.getPropertyValue("--grid"), MARK = css.getPropertyValue("--mark"),
      BG = css.getPropertyValue("--panel"), FLAG = css.getPropertyValue("--flag");

  var xs = pts.map(function (p) { return p.size / 1e9; });
  var ys = pts.map(function (p) { return p.kld; });
  var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
  var pad = (xmax - xmin) * 0.12 || 0.5;
  xmin = Math.max(0, xmin - pad); xmax += pad;
  var ymax = Math.max.apply(null, pts.map(function (p) { return p.hi; })) * 1.08 || 1;
  var ymin = 0;
  function X(v) { return M.l + (v - xmin) / (xmax - xmin) * (W - M.l - M.r); }
  function Y(v) { return H - M.b - (v - ymin) / (ymax - ymin) * (H - M.t - M.b); }

  var svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(n);
    return n;
  }
  function nice(v) { return v >= 1 ? v.toFixed(2) : v.toPrecision(2); }

  // grid + axes (recessive)
  for (var i = 0; i <= 4; i++) {
    var gv = ymin + (ymax - ymin) * i / 4;
    el("line", { x1: M.l, x2: W - M.r, y1: Y(gv), y2: Y(gv), stroke: GRID, "stroke-width": 1 });
    var t = el("text", { x: M.l - 8, y: Y(gv) + 4, "text-anchor": "end", fill: MUTED, "font-size": 11 });
    t.textContent = nice(gv);
  }
  var step = Math.max(1, Math.round((xmax - xmin) / 6));
  for (var gx = Math.ceil(xmin / step) * step; gx <= xmax; gx += step) {
    var tx = el("text", { x: X(gx), y: H - M.b + 18, "text-anchor": "middle", fill: MUTED, "font-size": 11 });
    tx.textContent = gx + " GB";
  }
  var yl = el("text", { x: 14, y: (H - M.b + M.t) / 2, fill: MUTED, "font-size": 11, "text-anchor": "middle",
                        transform: "rotate(-90 14 " + (H - M.b + M.t) / 2 + ")" });
  yl.textContent = "KLD mean (lower is better)";

  // best-for-its-size frontier
  var sorted = pts.slice().sort(function (a, b) { return a.size - b.size || a.kld - b.kld; });
  var best = Infinity, front = [];
  sorted.forEach(function (p) { if (p.kld < best) { front.push(p); best = p.kld; } });
  el("polyline", { points: front.map(function (p) { return X(p.size / 1e9) + "," + Y(p.kld); }).join(" "),
                   fill: "none", stroke: MARK, "stroke-width": 2, opacity: 0.5 });

  var tip = document.createElement("div"); tip.id = "tip"; document.body.appendChild(tip);

  // Labels: add the publisher when a label repeats, and move a label out of the way
  // when it would sit on another label or on a point.
  var dup = {};
  pts.forEach(function (p) { dup[p.label] = (dup[p.label] || 0) + 1; });
  function labelText(p) {
    return p.label + (dup[p.label] > 1 && p.publisher ? " (" + p.publisher + ")" : "") + (p.approx ? " *" : "");
  }
  var boxes = pts.map(function (p) {
    var x = X(p.size / 1e9), y = Y(p.kld);
    return [x - 9, y - 9, x + 9, y + 9];
  });
  function hits(b) {
    return boxes.some(function (o) { return b[0] < o[2] && o[0] < b[2] && b[1] < o[3] && o[1] < b[3]; });
  }
  function placeLabel(text, cx, cy) {
    var w = text.length * 6.3 + 4;
    var tries = [[9, -8, "start"], [9, 16, "start"], [-9, -8, "end"], [-9, 16, "end"], [9, -24, "start"], [9, 32, "start"], [-9, -24, "end"], [-9, 32, "end"]];
    for (var i = 0; i < tries.length; i++) {
      var dx = tries[i][0], dy = tries[i][1], anchor = tries[i][2];
      var x0 = anchor === "start" ? cx + dx : cx + dx - w;
      var b = [x0, cy + dy - 11, x0 + w, cy + dy + 3];
      if (b[0] >= M.l - 4 && b[2] <= W - 4 && !hits(b)) { boxes.push(b); return { x: cx + dx, y: cy + dy, anchor: anchor }; }
    }
    var d = tries[0]; return { x: cx + d[0], y: cy + d[1], anchor: d[2] };
  }

  pts.forEach(function (p) {
    var cx = X(p.size / 1e9), cy = Y(p.kld);
    el("line", { x1: cx, x2: cx, y1: Y(p.lo), y2: Y(p.hi), stroke: MARK, "stroke-width": 1, opacity: 0.45 });
    var c = el("circle", { cx: cx, cy: cy, r: 5, fill: p.approx ? FLAG : MARK, stroke: BG, "stroke-width": 2, tabindex: 0 });
    var hit = el("circle", { cx: cx, cy: cy, r: 14, fill: "transparent" });
    var text = labelText(p), pos = placeLabel(text, cx, cy);
    var lbl = el("text", { x: pos.x, y: pos.y, fill: INK, "font-size": 11, "text-anchor": pos.anchor });
    lbl.textContent = text;
    function show(ev) {
      tip.style.display = "block";
      tip.textContent = labelText(p).replace(" *", "") + ": KLD " + p.kld.toFixed(4) + " (" + p.lo.toFixed(4) + " to " + p.hi.toFixed(4) + "), " + (p.size / 1e9).toFixed(2) + " GB";
      var r = (ev.target.getBoundingClientRect ? ev.target.getBoundingClientRect() : { left: ev.clientX, top: ev.clientY, width: 0 });
      tip.style.left = Math.min(window.innerWidth - 300, r.left + r.width / 2 + 10) + "px";
      tip.style.top = (r.top - 36) + "px";
    }
    function hide() { tip.style.display = "none"; }
    [hit, c].forEach(function (n) {
      n.addEventListener("mouseenter", show); n.addEventListener("mouseleave", hide);
      n.addEventListener("focus", show); n.addEventListener("blur", hide);
    });
  });
  holder.appendChild(svg);
})();
