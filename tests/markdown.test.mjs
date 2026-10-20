// Tests for the page's Markdown parser. Run: node --test tests/
import assert from "node:assert/strict";
import test from "node:test";

import { parse, parseInline } from "../src/web/static/markdown.js";

test("inline formatting", () => {
  assert.deepEqual(parseInline("Total is **5540** in `sales`"), [
    { t: "text", v: "Total is " }, { t: "strong", v: "5540" }, { t: "text", v: " in " }, { t: "code", v: "sales" },
  ]);
  assert.deepEqual(parseInline("![chart](x.png) and [docs](https://example.com)"), [
    { t: "img", alt: "chart", src: "x.png" }, { t: "text", v: " and " },
    { t: "link", v: "docs", href: "https://example.com" },
  ]);
});

test("html stays text", () => {
  // The parser has no HTML token at all: tags come back as plain text for textContent.
  assert.deepEqual(parseInline("<script>alert(1)</script>"), [{ t: "text", v: "<script>alert(1)</script>" }]);
});

test("blocks: headings, lists, code, rules", () => {
  const blocks = parse("# Title\n\nIntro line\nsecond line\n\n- a\n- **b**\n\n1. one\n2. two\n\n```\nx = 1\n```\n---");
  assert.deepEqual(blocks.map((b) => b.type), ["h", "p", "ul", "ol", "code", "hr"]);
  assert.equal(blocks[1].lines.length, 2);
  assert.equal(blocks[2].items[1][0].t, "strong");
  assert.equal(blocks[4].text, "x = 1");
});

test("tables", () => {
  const [table] = parse("| Region | Sales |\n|---|---:|\n| North | 3940 |\n| South | 1600 |");
  assert.equal(table.type, "table");
  assert.equal(table.head.length, 2);
  assert.deepEqual(table.rows.map((r) => r[0][0].v), ["North", "South"]);
});

test("a pipe in prose is not a table", () => {
  assert.deepEqual(parse("a | b").map((b) => b.type), ["p"]);
});
