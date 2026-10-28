//! Labelled queries made from the database itself, for `raglite eval`.
//!
//! Each query is a few words taken from one chunk, and its expected source is that chunk's
//! source. Such queries check that retrieval can find a chunk from its own words. They are
//! easier than real questions, so use them to compare settings, not as a quality measure.

/// One labelled query: a query and the source it should find.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalCase {
    /// The query text.
    pub query: String,
    /// The source the query should return. `raglite eval` matches it as a substring.
    pub expected_source: String,
}

/// Most words in a generated query.
const QUERY_WORDS: usize = 6;

/// Shortest word that may appear in a generated query. Shorter words are mostly stop words.
const MIN_WORD_CHARS: usize = 4;

/// Makes a query from the words of a chunk. The words come from the chunk's longest sentence,
/// and up to six distinct ones are kept, spread evenly across the sentence, so the same chunk
/// always gives the same query. Returns `None` when the chunk has fewer than three such words.
pub fn lexical_query(text: &str) -> Option<String> {
    let sentence = text
        .split(['.', '!', '?', '\n'])
        .max_by_key(|s| s.split_whitespace().count())?;

    let mut words: Vec<&str> = Vec::new();
    for word in sentence.split_whitespace() {
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());
        let long_enough = word.chars().count() >= MIN_WORD_CHARS;
        let is_word = word.chars().all(char::is_alphabetic);
        let seen = words.iter().any(|w| w.eq_ignore_ascii_case(word));
        if long_enough && is_word && !seen {
            words.push(word);
        }
    }
    if words.len() < 3 {
        return None;
    }

    let picks = words.len().min(QUERY_WORDS);
    let query: Vec<&str> = (0..picks).map(|i| words[i * words.len() / picks]).collect();
    Some(query.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_words_come_from_the_chunk_and_are_distinct() {
        let text = "Jupiter is the largest planet in the solar system. It has dozen moons and a great red storm.";
        let query = lexical_query(text).unwrap();
        let words: Vec<&str> = query.split(' ').collect();
        assert!(words.len() <= QUERY_WORDS);
        for word in &words {
            assert!(text.contains(word), "{word} is not in the chunk");
        }
        let mut unique = words.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), words.len());
    }

    #[test]
    fn the_same_chunk_always_gives_the_same_query() {
        let text = "Tomato plants need full sun and regular watering through the summer months.";
        assert_eq!(lexical_query(text), lexical_query(text));
    }

    #[test]
    fn short_chunks_give_no_query() {
        assert_eq!(lexical_query("Hi there."), None);
        assert_eq!(lexical_query("12 34 56 78 90."), None);
    }
}
