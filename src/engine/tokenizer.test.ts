import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { BPETokenizer } from './tokenizer';

const tokPath = fileURLToPath(new URL('../../public/models/tokenizer.json', import.meta.url));
const casesPath = fileURLToPath(new URL('./fixtures/tokenizer_cases.json', import.meta.url));
const chatPath = fileURLToPath(new URL('./fixtures/chat_cases.json', import.meta.url));

const tok = new BPETokenizer(JSON.parse(fs.readFileSync(tokPath, 'utf-8')));
const cases = JSON.parse(fs.readFileSync(casesPath, 'utf-8')) as { text: string; ids: number[] }[];
const chat = JSON.parse(fs.readFileSync(chatPath, 'utf-8')) as (
  | { messages: { role: string; content: string }[]; ids: number[]; weights: number[] }
  | { prompt_ids: number[] }
)[];

describe('BPETokenizer', () => {
  it('matches the Python reference encoder exactly', () => {
    for (const c of cases) {
      expect(tok.encode(c.text)).toEqual(c.ids);
    }
  });
  it('roundtrips text (incl. unicode, whitespace, code)', () => {
    for (const c of cases) {
      expect(tok.decode(tok.encode(c.text))).toBe(c.text);
    }
  });
  it('matches Python chat encoding + weights', () => {
    for (const c of chat) {
      if ('messages' in c) {
        const got = tok.encodeChat(c.messages);
        expect(got.ids).toEqual(c.ids);
        expect(got.weights).toEqual(c.weights);
      } else {
        expect(tok.chatPrompt('Hello!')).toEqual(c.prompt_ids);
      }
    }
  });
  it('handles unknown-script text without crashing', () => {
    const ids = tok.encode('日本語テスト mixed 中文ünicode✓');
    expect(tok.decode(ids)).toBe('日本語テスト mixed 中文ünicode✓');
  });
});
