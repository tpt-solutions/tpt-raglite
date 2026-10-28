use serde::{Deserialize, Serialize};

use crate::error::{RagError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    pub window_size: usize,
    pub overlap: usize,
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            overlap: 50,
            min_chunk_size: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
}

pub struct Chunker {
    config: ChunkConfig,
}

impl Chunker {
    pub fn new(config: ChunkConfig) -> Result<Self> {
        if config.window_size == 0 {
            return Err(RagError::InvalidConfig(
                "window_size must be greater than 0".into(),
            ));
        }
        if config.overlap >= config.window_size {
            return Err(RagError::InvalidConfig(format!(
                "overlap ({}) must be smaller than window_size ({})",
                config.overlap, config.window_size
            )));
        }
        Ok(Self { config })
    }

    /// Splits `text` into windows of at most `window_size` words. Every chunk is a
    /// slice of the original text, so interior whitespace (including newlines) is kept.
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let spans = word_spans(text);
        let words = spans.len();
        if words == 0 {
            return Vec::new();
        }
        if words <= self.config.window_size {
            return vec![Chunk {
                text: text.to_string(),
                start_offset: 0,
                end_offset: text.len(),
            }];
        }

        let tokens: Vec<&str> = spans.iter().map(|&(s, e)| &text[s..e]).collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        loop {
            let end = (start + self.config.window_size).min(words);
            let take = find_sentence_boundary(&tokens[start..end], self.config.min_chunk_size);
            // Exclusive word index where this chunk stops.
            let last = start + take;
            let start_offset = spans[start].0;
            let end_offset = spans[last - 1].1;

            chunks.push(Chunk {
                text: text[start_offset..end_offset].to_string(),
                start_offset,
                end_offset,
            });

            if last >= words {
                break;
            }

            // Restart `overlap` words before the chunk end. Always move forward so a
            // chunk shorter than the overlap cannot stall the loop.
            let next = last.saturating_sub(self.config.overlap);
            start = if next > start { next } else { start + 1 };
        }

        chunks
    }
}

/// Byte ranges of each whitespace-separated word in `text`.
fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut word_start = None;
    for (i, ch) in text.char_indices() {
        match (ch.is_whitespace(), word_start) {
            (false, None) => word_start = Some(i),
            (true, Some(start)) => {
                spans.push((start, i));
                word_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = word_start {
        spans.push((start, text.len()));
    }
    spans
}

/// Returns how many leading words to take: up to the last sentence end at or after
/// `min_size`, or all of `words` when there is none.
fn find_sentence_boundary(words: &[&str], min_size: usize) -> usize {
    if words.len() <= min_size {
        return words.len();
    }

    for i in (min_size..words.len()).rev() {
        if ends_sentence(words[i]) {
            return i + 1;
        }
    }

    words.len()
}

const ABBREVIATIONS: &[&str] = &[
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "inc", "ltd", "co", "no", "fig",
    "vol", "dept", "gen", "gov", "capt", "lt", "sgt", "rev",
];

/// True when `token` ends a sentence. A trailing period does not count when the token
/// is a known abbreviation ("Dr."), a single initial ("J."), or a dotted acronym ("U.S.").
fn ends_sentence(token: &str) -> bool {
    let token = token.trim_end_matches(['"', '\'', ')', ']', '\u{201d}', '\u{2019}']);
    if token.ends_with('!') || token.ends_with('?') {
        return true;
    }
    match token.strip_suffix('.') {
        Some(core) => !core.is_empty() && !is_abbreviation(core),
        None => false,
    }
}

/// `core` is a token without its final period.
fn is_abbreviation(core: &str) -> bool {
    if core.contains('.') {
        return true;
    }
    let lower = core.to_lowercase();
    if ABBREVIATIONS.contains(&lower.as_str()) {
        return true;
    }
    core.chars().count() == 1 && core.chars().all(char::is_alphabetic)
}

pub struct SemanticChunker {
    min_chunk_size: usize,
}

impl SemanticChunker {
    pub fn new(min_chunk_size: usize) -> Self {
        Self { min_chunk_size }
    }

    /// Groups sentences into chunks, merging sentences until a chunk reaches
    /// `min_chunk_size` words. Chunk text is a slice of the original text.
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let sentences = split_sentences(text);
        let Some((first, rest)) = sentences.split_first() else {
            return Vec::new();
        };

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut current = *first;
        let mut current_words = word_count(&text[current.start_offset..current.end_offset]);

        for sentence in rest {
            let sentence_words = word_count(&text[sentence.start_offset..sentence.end_offset]);
            if current_words + sentence_words < self.min_chunk_size {
                current.end_offset = sentence.end_offset;
                current_words += sentence_words;
            } else {
                chunks.push(make_chunk(text, current));
                current = *sentence;
                current_words = sentence_words;
            }
        }

        chunks.push(make_chunk(text, current));
        chunks
    }
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn make_chunk(text: &str, span: Sentence) -> Chunk {
    Chunk {
        text: text[span.start_offset..span.end_offset].to_string(),
        start_offset: span.start_offset,
        end_offset: span.end_offset,
    }
}

/// Byte range of one sentence in the original text, with surrounding whitespace removed.
#[derive(Clone, Copy)]
struct Sentence {
    start_offset: usize,
    end_offset: usize,
}

fn split_sentences(text: &str) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut current_start = 0;

    for (i, ch) in text.char_indices() {
        if !matches!(ch, '.' | '!' | '?') {
            continue;
        }
        let next = i + ch.len_utf8();
        let followed_by_space = text[next..].starts_with(char::is_whitespace);
        if !followed_by_space {
            continue;
        }
        // The word ending at this punctuation mark, e.g. "Dr." or "sentence.".
        let word = text[..next]
            .rsplit(char::is_whitespace)
            .next()
            .unwrap_or_default();
        if !ends_sentence(word) {
            continue;
        }

        push_sentence(text, current_start, next, &mut sentences);
        // Skip whitespace after the boundary.
        let skipped = text[next..].len() - text[next..].trim_start().len();
        current_start = next + skipped;
    }

    push_sentence(text, current_start, text.len(), &mut sentences);
    sentences
}

fn push_sentence(text: &str, start: usize, end: usize, out: &mut Vec<Sentence>) {
    let segment = &text[start..end];
    let leading = segment.len() - segment.trim_start().len();
    let trimmed = segment.trim();
    if !trimmed.is_empty() {
        let start_offset = start + leading;
        out.push(Sentence {
            start_offset,
            end_offset: start_offset + trimmed.len(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunker(window_size: usize, overlap: usize, min_chunk_size: usize) -> Chunker {
        Chunker::new(ChunkConfig {
            window_size,
            overlap,
            min_chunk_size,
        })
        .unwrap()
    }

    #[test]
    fn test_basic_chunking() {
        let chunker = chunker(5, 2, 2);
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        assert!(chunks[0].text.contains("one"));
    }

    #[test]
    fn test_sentence_boundary() {
        let words = &[
            "Hello",
            "world.",
            "This",
            "is",
            "a",
            "test",
            "sentence.",
            "And",
            "more",
        ];
        // "world." is before min_size, so the last sentence end is "sentence.".
        assert_eq!(find_sentence_boundary(words, 2), 7);
    }

    #[test]
    fn test_abbreviations_do_not_end_sentences() {
        assert!(!ends_sentence("Dr."));
        assert!(!ends_sentence("U.S."));
        assert!(!ends_sentence("J."));
        assert!(ends_sentence("done."));
        assert!(ends_sentence("really?"));
        assert!(ends_sentence("\"quoted.\""));
    }

    #[test]
    fn test_short_text_no_chunking() {
        let chunker = Chunker::new(ChunkConfig::default()).unwrap();
        let text = "Short text here.";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_overlap() {
        let chunker = chunker(4, 2, 2);
        let text = "a b c d e f g h i j k l";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_invalid_config_rejected() {
        let overlap_too_big = ChunkConfig {
            window_size: 4,
            overlap: 4,
            min_chunk_size: 1,
        };
        assert!(Chunker::new(overlap_too_big).is_err());
        let zero_window = ChunkConfig {
            window_size: 0,
            overlap: 0,
            min_chunk_size: 1,
        };
        assert!(Chunker::new(zero_window).is_err());
    }

    #[test]
    fn test_large_document_keeps_all_words() {
        let words: Vec<String> = (0..5000)
            .map(|i| {
                if i % 10 == 9 {
                    format!("w{i}.")
                } else {
                    format!("w{i}")
                }
            })
            .collect();
        let text = words.join(" ");
        let chunks = Chunker::new(ChunkConfig::default()).unwrap().chunk(&text);

        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].start_offset, 0);
        assert_eq!(chunks.last().unwrap().end_offset, text.len());
        for pair in chunks.windows(2) {
            assert!(
                pair[1].start_offset <= pair[0].end_offset,
                "gap between chunks"
            );
            assert!(
                pair[1].start_offset > pair[0].start_offset,
                "chunks do not advance"
            );
        }
        for word in &words {
            assert!(
                chunks
                    .iter()
                    .any(|c| c.text.split_whitespace().any(|w| w == word)),
                "word {word} was dropped"
            );
        }
    }

    #[test]
    fn test_chunks_preserve_newlines() {
        let text = "line one\nline two\nline three\nline four\nline five\nline six";
        let chunks = chunker(4, 1, 1).chunk(text);
        assert!(chunks.iter().any(|c| c.text.contains('\n')));
        for c in &chunks {
            assert_eq!(c.text, &text[c.start_offset..c.end_offset]);
        }
    }

    #[test]
    fn test_semantic_chunker_basic() {
        let chunker = SemanticChunker::new(5);
        let text = "First sentence here. Second sentence here. Third sentence here.";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_semantic_chunker_merges_short() {
        let chunker = SemanticChunker::new(10);
        let text = "Hi. World. This is a much longer sentence that should be its own chunk.";
        let chunks = chunker.chunk(text);
        // "Hi." and "World." are short, should merge; the long one stays separate
        assert!(chunks.len() <= 2);
    }

    #[test]
    fn test_semantic_chunker_does_not_split_on_abbreviation() {
        let chunker = SemanticChunker::new(1);
        let text = "Dr. Smith arrived. He left.";
        let chunks = chunker.chunk(text);
        assert!(chunks[0].text.starts_with("Dr. Smith"));
    }

    #[test]
    fn test_semantic_chunker_empty() {
        let chunker = SemanticChunker::new(5);
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }
}
