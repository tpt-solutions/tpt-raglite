//! WordPiece tokenizer for BERT-style uncased models such as all-MiniLM-L6-v2.
//!
//! The vocabulary is the `vocab.txt` file shipped with the model, one token per line.
//! Token ids are line numbers, so the file must match the model exactly.

use std::collections::HashMap;
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use crate::error::{RagError, Result};

/// Words longer than this become a single `[UNK]`, as in the reference implementation.
const MAX_WORD_CHARS: usize = 100;

/// BERT-style WordPiece tokenizer, loaded from a `vocab.txt` file.
pub struct WordPieceTokenizer {
    vocab: HashMap<String, i64>,
    unk_id: i64,
    cls_id: i64,
    sep_id: i64,
    pad_id: i64,
}

impl WordPieceTokenizer {
    /// Loads the vocabulary from a `vocab.txt` file with one token per line.
    pub fn from_vocab_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_vocab(text.lines())
    }

    /// Builds the tokenizer from vocabulary lines, where each line's position is its id.
    pub fn from_vocab<'a>(lines: impl Iterator<Item = &'a str>) -> Result<Self> {
        let vocab: HashMap<String, i64> = lines
            .enumerate()
            .map(|(id, line)| (line.trim_end_matches('\r').to_string(), id as i64))
            .collect();

        let special = |token: &str| {
            vocab
                .get(token)
                .copied()
                .ok_or_else(|| RagError::Model(format!("vocabulary has no {token} token")))
        };
        let unk_id = special("[UNK]")?;
        let cls_id = special("[CLS]")?;
        let sep_id = special("[SEP]")?;
        // BERT vocabularies put [PAD] at id 0. Fall back to that if the file omits it.
        let pad_id = vocab.get("[PAD]").copied().unwrap_or(0);

        Ok(Self {
            vocab,
            unk_id,
            cls_id,
            sep_id,
            pad_id,
        })
    }

    /// Id of the `[PAD]` token.
    pub fn pad_id(&self) -> i64 {
        self.pad_id
    }

    /// Returns `[CLS] ... [SEP]` token ids, at most `max_len` long. Input beyond the limit
    /// is dropped, so the closing `[SEP]` is always present.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<i64> {
        let budget = max_len.saturating_sub(2);
        let mut ids = self.word_ids(text);
        ids.truncate(budget);

        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(self.cls_id);
        out.extend(ids);
        out.push(self.sep_id);
        out
    }

    /// Encodes a sentence pair as `[CLS] a [SEP] b [SEP]`, with token type 0 for the first
    /// segment and 1 for the second. The longer segment is trimmed one token at a time
    /// until the sequence fits in `max_len`.
    pub fn encode_pair(&self, first: &str, second: &str, max_len: usize) -> (Vec<i64>, Vec<i64>) {
        let mut a = self.word_ids(first);
        let mut b = self.word_ids(second);
        let budget = max_len.saturating_sub(3);
        while a.len() + b.len() > budget {
            if a.len() >= b.len() {
                a.pop();
            } else {
                b.pop();
            }
        }

        let mut ids = Vec::with_capacity(a.len() + b.len() + 3);
        ids.push(self.cls_id);
        ids.extend(&a);
        ids.push(self.sep_id);
        ids.extend(&b);
        ids.push(self.sep_id);

        let mut type_ids = vec![0i64; a.len() + 2];
        type_ids.extend(std::iter::repeat_n(1i64, b.len() + 1));
        (ids, type_ids)
    }

    /// WordPiece ids for every word in `text`, without special tokens.
    fn word_ids(&self, text: &str) -> Vec<i64> {
        basic_tokenize(text)
            .iter()
            .flat_map(|word| self.wordpiece(word))
            .collect()
    }

    /// Greedy longest-match-first split of one word into vocabulary pieces. Continuation
    /// pieces carry the `##` prefix. If any part of the word has no piece, the whole word
    /// becomes `[UNK]`.
    fn wordpiece(&self, word: &str) -> Vec<i64> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > MAX_WORD_CHARS {
            return vec![self.unk_id];
        }

        let mut pieces = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let mut piece: String = chars[start..end].iter().collect();
                if start > 0 {
                    piece.insert_str(0, "##");
                }
                if let Some(&id) = self.vocab.get(&piece) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    pieces.push(id);
                    start = end;
                }
                None => return vec![self.unk_id],
            }
        }
        pieces
    }
}

/// Lowercases, removes accents, and splits on whitespace and punctuation. Punctuation
/// marks become words of their own.
///
/// Approximations of the reference implementation: CJK characters are not split into
/// single-character words, and non-ASCII symbols count as punctuation.
pub fn basic_tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in text.nfd() {
        if is_combining_mark(ch) {
            continue;
        }
        for lower in ch.to_lowercase() {
            if lower.is_whitespace() || lower.is_control() {
                flush(&mut current, &mut words);
            } else if is_punctuation(lower) {
                flush(&mut current, &mut words);
                words.push(lower.to_string());
            } else {
                current.push(lower);
            }
        }
    }
    flush(&mut current, &mut words);
    words
}

fn flush(current: &mut String, words: &mut Vec<String>) {
    if !current.is_empty() {
        words.push(std::mem::take(current));
    }
}

/// Combining diacritical marks (U+0300 to U+036F), removed after NFD decomposition.
fn is_combining_mark(ch: char) -> bool {
    ('\u{0300}'..='\u{036F}').contains(&ch)
}

fn is_punctuation(ch: char) -> bool {
    if ch.is_ascii() {
        ch.is_ascii_punctuation()
    } else {
        !ch.is_alphanumeric() && !ch.is_whitespace() && !ch.is_control()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small vocabulary that exercises every rule.
    fn test_tokenizer() -> WordPieceTokenizer {
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##s", "play", "##ing",
            "cafe", ",", "!", "un", "##aff", "##able",
        ];
        WordPieceTokenizer::from_vocab(vocab.into_iter()).unwrap()
    }

    fn id_of(tok: &WordPieceTokenizer, token: &str) -> i64 {
        tok.vocab[token]
    }

    #[test]
    fn test_basic_tokenize_lowercases_and_splits_punctuation() {
        assert_eq!(
            basic_tokenize("Hello, World!"),
            vec!["hello", ",", "world", "!"]
        );
    }

    #[test]
    fn test_basic_tokenize_strips_accents() {
        assert_eq!(basic_tokenize("Café"), vec!["cafe"]);
    }

    #[test]
    fn test_wordpiece_uses_longest_match() {
        let tok = test_tokenizer();
        // "playing" is "play" followed by the continuation piece "##ing".
        assert_eq!(
            tok.wordpiece("playing"),
            vec![id_of(&tok, "play"), id_of(&tok, "##ing")]
        );
        // "unaffable" is "un" + "##aff" + "##able".
        assert_eq!(
            tok.wordpiece("unaffable"),
            vec![
                id_of(&tok, "un"),
                id_of(&tok, "##aff"),
                id_of(&tok, "##able")
            ]
        );
    }

    #[test]
    fn test_unknown_word_becomes_single_unk() {
        let tok = test_tokenizer();
        assert_eq!(tok.wordpiece("zzz"), vec![tok.unk_id]);
        assert_eq!(tok.wordpiece(&"a".repeat(101)), vec![tok.unk_id]);
    }

    #[test]
    fn test_encode_wraps_with_cls_and_sep() {
        let tok = test_tokenizer();
        let ids = tok.encode("Hello world", 128);
        assert_eq!(
            ids,
            vec![
                tok.cls_id,
                id_of(&tok, "hello"),
                id_of(&tok, "world"),
                tok.sep_id
            ]
        );
    }

    #[test]
    fn test_encode_respects_max_len() {
        let tok = test_tokenizer();
        let ids = tok.encode("hello world hello world hello world", 4);
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], tok.cls_id);
        assert_eq!(ids[3], tok.sep_id);
    }

    /// Uses the real `vocab.txt` from all-MiniLM-L6-v2 at `models/vocab.txt` when it is
    /// present (see `scripts/fetch_model`). Skips otherwise.
    #[test]
    fn test_matches_reference_ids() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/vocab.txt");
        if !path.exists() {
            eprintln!("skipping: {} not found", path.display());
            return;
        }
        let tok = WordPieceTokenizer::from_vocab_file(&path).unwrap();
        // Ids from bert-base-uncased, which all-MiniLM-L6-v2 uses.
        assert_eq!(tok.encode("Hello world", 128), vec![101, 7592, 2088, 102]);
    }
}
