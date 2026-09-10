use std::path::Path;

use crate::error::Result;

pub trait Parser: Send + Sync {
    fn parse(&self, path: &Path) -> Result<String>;
    fn supported_extensions(&self) -> &[&str];
}

pub struct PlainTextParser;

impl Parser for PlainTextParser {
    fn parse(&self, path: &Path) -> Result<String> {
        Ok(std::fs::read_to_string(path)?)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["txt"]
    }
}

pub struct MarkdownParser;

impl Parser for MarkdownParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        Ok(strip_markdown(&content))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown", "mdown"]
    }
}

pub struct HtmlParser;

impl Parser for HtmlParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let html = std::fs::read_to_string(path)?;
        Ok(extract_text_from_html(&html))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["html", "htm"]
    }
}

pub struct PdfParser;

impl Parser for PdfParser {
    fn parse(&self, path: &Path) -> Result<String> {
        pdf_extract::extract_text(path).map_err(|e| {
            crate::error::RagError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("PDF extraction failed: {e}"),
            ))
        })
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
}

pub fn get_parser(path: &Path) -> Option<Box<dyn Parser>> {
    let ext = path.extension()?.to_str()?;
    match ext.to_lowercase().as_str() {
        "txt" => Some(Box::new(PlainTextParser)),
        "md" | "markdown" | "mdown" => Some(Box::new(MarkdownParser)),
        "html" | "htm" => Some(Box::new(HtmlParser)),
        "pdf" => Some(Box::new(PdfParser)),
        _ => None,
    }
}

fn strip_markdown(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for line in text.lines() {
        let stripped = line
            .trim_start_matches('#')
            .trim_start_matches('*')
            .trim_start_matches('_')
            .trim_start_matches('>')
            .trim_start_matches('-')
            .trim_start_matches('+')
            .trim_start();
        if !stripped.is_empty() {
            result.push_str(stripped);
            result.push('\n');
        }
    }
    result.trim().to_string()
}

fn extract_text_from_html(html: &str) -> String {
    use html5ever::driver::ParseOpts;
    use html5ever::parse_document;
    use html5ever::tendril::TendrilSink;
    use markup5ever_rcdom::RcDom;

    let dom = parse_document(RcDom::default(), ParseOpts::default())
        .from_utf8()
        .read_from(&mut html.as_bytes())
        .unwrap_or_default();

    let mut text = String::new();
    extract_text_from_node(&dom.document, &mut text);
    text.trim().to_string()
}

fn extract_text_from_node(handle: &markup5ever_rcdom::Handle, text: &mut String) {
    use markup5ever_rcdom::NodeData;

    match &handle.data {
        NodeData::Text { contents } => {
            let borrowed = contents.borrow();
            let content = borrowed.trim();
            if !content.is_empty() {
                text.push_str(content);
                text.push(' ');
            }
        }
        NodeData::Element { name, .. } => {
            let tag = name.local.as_ref();
            let block_tags = ["p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"];
            if block_tags.contains(&tag) {
                text.push('\n');
            }
            let children = handle.children.borrow();
            for child in children.iter() {
                extract_text_from_node(child, text);
            }
        }
        NodeData::Document => {
            let children = handle.children.borrow();
            for child in children.iter() {
                extract_text_from_node(child, text);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, content: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("tpt_rag_parser_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content).unwrap();
        path
    }

    #[test]
    fn test_plain_text_parser() {
        let path = write_temp("test.txt", b"Hello, world!\nThis is a test.");
        let parser = PlainTextParser;
        let text = parser.parse(&path).unwrap();
        assert!(text.contains("Hello, world!"));
        assert!(text.contains("This is a test."));
    }

    #[test]
    fn test_markdown_parser() {
        let path = write_temp(
            "test.md",
            b"# Title\n\nSome **bold** text.\n\n## Section\n\n- item 1\n- item 2",
        );
        let parser = MarkdownParser;
        let text = parser.parse(&path).unwrap();
        assert!(text.contains("Title"));
        assert!(text.contains("Some"));
        assert!(text.contains("item 1"));
    }

    #[test]
    fn test_html_parser() {
        let path = write_temp(
            "test.html",
            b"<html><body><h1>Title</h1><p>Paragraph one.</p><p>Paragraph two.</p></body></html>",
        );
        let parser = HtmlParser;
        let text = parser.parse(&path).unwrap();
        assert!(text.contains("Title"));
        assert!(text.contains("Paragraph one."));
    }

    #[test]
    fn test_get_parser() {
        let p = get_parser(Path::new("foo.pdf"));
        assert!(p.is_some());
        let p = get_parser(Path::new("foo.xyz"));
        assert!(p.is_none());
    }
}
