// A small, safe Markdown renderer for model output.
//
// Why not innerHTML + a library? Answers can repeat text from CSV cells, which is untrusted
// input (see data/injected_sales.csv). Here nothing is ever parsed as HTML: parse() turns the
// text into plain objects, and toDom() builds elements with createElement + textContent.
// Links must be http(s); images go through a caller-supplied resolver (our chart files only).
//
// Supported: paragraphs, # headings, - / 1. lists, | tables |, ``` code blocks ```, ---,
// **bold**, *italic*, `code`, [links](https://...), ![images](file.png).

const INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\s][^*\n]*\*)|(!\[[^\]\n]*\]\([^)\s]+\))|(\[[^\]\n]+\]\([^)\s]+\))/g;

export function parseInline(text) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > last) out.push({ t: "text", v: text.slice(last, m.index) });
    const s = m[0];
    if (m[1]) out.push({ t: "code", v: s.slice(1, -1) });
    else if (m[2]) out.push({ t: "strong", v: s.slice(2, -2) });
    else if (m[3]) out.push({ t: "em", v: s.slice(1, -1) });
    else if (m[4]) { const [, alt, src] = s.match(/^!\[(.*)\]\((.*)\)$/); out.push({ t: "img", alt, src }); }
    else { const [, label, href] = s.match(/^\[(.*)\]\((.*)\)$/); out.push({ t: "link", v: label, href }); }
    last = m.index + s.length;
  }
  if (last < text.length) out.push({ t: "text", v: text.slice(last) });
  return out;
}

const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => parseInline(c.trim()));
const isTableSep = (line) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
const LIST = /^\s*(?:([-*+])|(\d+)[.)])\s+(.*)$/;

export function parse(text) {
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let para = [];
  const flush = () => {
    if (para.length) blocks.push({ type: "p", lines: para.map(parseInline) });
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {                               // fenced code
      flush();
      const code = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) code.push(lines[i]);
      blocks.push({ type: "code", text: code.join("\n") });
    } else if (/^\s*$/.test(line)) {
      flush();
    } else if (/^#{1,6}\s/.test(line)) {
      flush();
      const level = line.match(/^#+/)[0].length;
      blocks.push({ type: "h", level, inline: parseInline(line.replace(/^#+\s*/, "")) });
    } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      flush();
      blocks.push({ type: "hr" });
    } else if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      flush();
      const head = cells(line);
      const rows = [];
      for (i += 2; i < lines.length && lines[i].includes("|") && lines[i].trim(); i++) rows.push(cells(lines[i]));
      i--;
      blocks.push({ type: "table", head, rows });
    } else if (LIST.test(line)) {
      flush();
      const ordered = !!line.match(LIST)[2];
      const items = [];
      for (; i < lines.length && LIST.test(lines[i]); i++) items.push(parseInline(lines[i].match(LIST)[3]));
      i--;
      blocks.push({ type: ordered ? "ol" : "ul", items });
    } else {
      para.push(line);
    }
  }
  flush();
  return blocks;
}

function inlineDom(tokens, opts) {
  const frag = document.createDocumentFragment();
  for (const tok of tokens) {
    if (tok.t === "text") frag.append(document.createTextNode(tok.v));
    else if (tok.t === "strong" || tok.t === "em" || tok.t === "code") {
      const e = document.createElement(tok.t);
      e.textContent = tok.v;
      frag.append(e);
    } else if (tok.t === "link") {
      if (/^https?:\/\//i.test(tok.href)) {
        const a = document.createElement("a");
        a.href = tok.href;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = tok.v;
        frag.append(a);
      } else frag.append(document.createTextNode(tok.v));
    } else if (tok.t === "img") {
      const src = opts.image ? opts.image(tok.src) : null;
      if (src) {
        const img = document.createElement("img");
        img.src = src;
        img.alt = tok.alt;
        img.loading = "lazy";
        frag.append(img);
      } else frag.append(document.createTextNode(tok.alt ? `[${tok.alt}]` : ""));
    }
  }
  return frag;
}

export function toDom(blocks, opts = {}) {
  const frag = document.createDocumentFragment();
  const el = (tag, child) => { const e = document.createElement(tag); if (child) e.append(child); return e; };
  for (const b of blocks) {
    if (b.type === "p") {
      const p = el("p");
      b.lines.forEach((ln, k) => { if (k) p.append(el("br")); p.append(inlineDom(ln, opts)); });
      frag.append(p);
    } else if (b.type === "h") frag.append(el(`h${Math.min(b.level + 1, 6)}`, inlineDom(b.inline, opts)));
    else if (b.type === "hr") frag.append(el("hr"));
    else if (b.type === "code") { const pre = el("pre"); pre.append(el("code")); pre.firstChild.textContent = b.text; frag.append(pre); }
    else if (b.type === "ul" || b.type === "ol") {
      const list = el(b.type);
      for (const item of b.items) list.append(el("li", inlineDom(item, opts)));
      frag.append(list);
    } else if (b.type === "table") {
      const wrap = el("div"); wrap.className = "table-wrap";
      const table = el("table");
      const tr = el("tr");
      b.head.forEach((c) => tr.append(el("th", inlineDom(c, opts))));
      table.append(el("thead", tr));
      const body = el("tbody");
      for (const row of b.rows) { const r = el("tr"); row.forEach((c) => r.append(el("td", inlineDom(c, opts)))); body.append(r); }
      table.append(body);
      wrap.append(table);
      frag.append(wrap);
    }
  }
  return frag;
}

export function renderMarkdown(text, opts) {
  return toDom(parse(text), opts);
}
