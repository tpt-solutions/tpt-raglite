use serde::{Deserialize, Serialize};

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
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }

    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() <= self.config.window_size {
            return vec![Chunk {
                text: text.to_string(),
                start_offset: 0,
                end_offset: text.len(),
            }];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < words.len() {
            let end = (start + self.config.window_size).min(words.len());
            let chunk_words = &words[start..end];

            let (chunk_end, chunk_text) =
                find_sentence_boundary(chunk_words, self.config.min_chunk_size);

            let actual_end = start + chunk_end;
            let chunk_start_offset = word_offset(text, &words, start);
            let chunk_end_offset = word_offset(text, &words, actual_end);

            chunks.push(Chunk {
                text: chunk_text,
                start_offset: chunk_start_offset,
                end_offset: chunk_end_offset,
            });

            let advance = if chunk_end >= self.config.window_size {
                self.config.window_size - self.config.overlap
            } else {
                break;
            };
            start += advance.max(1);
        }

        chunks
    }
}

fn find_sentence_boundary(words: &[&str], min_size: usize) -> (usize, String) {
    if words.len() <= min_size {
        let text = words.join(" ");
        return (words.len(), text);
    }

    for i in (min_size..words.len()).rev() {
        if let Some(word) = words.get(i) {
            let trimmed = word.trim_end();
            if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
                let text = words[..=i].join(" ");
                return (i + 1, text);
            }
        }
    }

    (words.len(), words.join(" "))
}

fn word_offset(text: &str, _words: &[&str], word_index: usize) -> usize {
    if word_index == 0 {
        return 0;
    }
    let mut word_count = 0;
    let mut in_whitespace = false;

    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if !in_whitespace {
                in_whitespace = true;
                word_count += 1;
                if word_count >= word_index {
                    return i + 1;
                }
            }
        } else {
            in_whitespace = false;
        }
    }
    text.len()
}

pub struct SemanticChunker {
    min_chunk_size: usize,
}

impl SemanticChunker {
    pub fn new(min_chunk_size: usize) -> Self {
        Self { min_chunk_size }
    }

    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut current_text = String::new();
        let mut current_start = 0usize;

        for sentence in &sentences {
            if current_text.is_empty() {
                current_start = sentence.start_offset;
                current_text = sentence.text.clone();
            } else {
                let combined_len = current_text.split_whitespace().count()
                    + sentence.text.split_whitespace().count();
                if combined_len < self.min_chunk_size {
                    current_text.push(' ');
                    current_text.push_str(&sentence.text);
                } else {
                    chunks.push(Chunk {
                        text: std::mem::take(&mut current_text),
                        start_offset: current_start,
                        end_offset: sentence.start_offset,
                    });
                    current_start = sentence.start_offset;
                    current_text = sentence.text.clone();
                }
            }
        }

        if !current_text.is_empty() {
            let end_offset = text.len();
            chunks.push(Chunk {
                text: current_text,
                start_offset: current_start,
                end_offset,
            });
        }

        chunks
    }
}

struct Sentence {
    text: String,
    start_offset: usize,
}

fn split_sentences(text: &str) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut current_start = 0;

    for (i, ch) in text.char_indices() {
        if ch == '.' || ch == '!' || ch == '?' {
            let next = i + ch.len_utf8();
            if next < text.len() {
                let next_ch = text[next..].chars().next().unwrap_or(' ');
                if next_ch == ' ' || next_ch == '\n' || next_ch == '\r' {
                    let end = next;
                    let segment = text[current_start..end].trim();
                    if !segment.is_empty() {
                        sentences.push(Sentence {
                            text: segment.to_string(),
                            start_offset: current_start,
                        });
                    }
                    // Skip whitespace after sentence boundary
                    let mut advanced = end;
                    while advanced < text.len() {
                        let c = text[advanced..].chars().next().unwrap_or(' ');
                        if c == ' ' || c == '\n' || c == '\r' {
                            advanced += c.len_utf8();
                        } else {
                            break;
                        }
                    }
                    current_start = advanced;
                }
            }
        }
    }

    if current_start < text.len() {
        let remaining = text[current_start..].trim();
        if !remaining.is_empty() {
            sentences.push(Sentence {
                text: remaining.to_string(),
                start_offset: current_start,
            });
        }
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let config = ChunkConfig {
            window_size: 5,
            overlap: 2,
            min_chunk_size: 2,
        };
        let chunker = Chunker::new(config);
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        assert!(chunks[0].text.contains("one"));
    }

    #[test]
    fn test_sentence_boundary() {
        let words = &[
            "Hello", "world.", "This", "is", "a", "test", "sentence.", "And", "more",
        ];
        let (end, text) = find_sentence_boundary(words, 2);
        assert_eq!(end, 3);
        assert!(text.contains("world."));
    }

    #[test]
    fn test_short_text_no_chunking() {
        let config = ChunkConfig::default();
        let chunker = Chunker::new(config);
        let text = "Short text here.";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_overlap() {
        let config = ChunkConfig {
            window_size: 4,
            overlap: 2,
            min_chunk_size: 2,
        };
        let chunker = Chunker::new(config);
        let text = "a b c d e f g h i j k l";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() > 1);
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
    fn test_semantic_chunker_empty() {
        let chunker = SemanticChunker::new(5);
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }
}
