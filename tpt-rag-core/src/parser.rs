use std::path::Path;

use pulldown_cmark::{Event, HeadingLevel, Parser as CmarkParser, Tag, TagEnd};

use crate::error::{RagError, Result};

/// Largest file a parser will read. Bigger files are rejected before they are loaded.
pub const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// A parsed document: its text, and the headings in it, for formats that have them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    /// The text. Chunk offsets and heading offsets refer to this text.
    pub text: String,
    /// Headings in document order.
    pub headings: Vec<Heading>,
    /// Offsets where a chunk must not cross, such as the edges of a Markdown code block.
    /// Headings are boundaries too, without being listed here.
    pub boundaries: Vec<usize>,
}

/// A heading in a parsed document.
#[derive(Debug, Clone, PartialEq)]
pub struct Heading {
    /// Byte offset in [`Document::text`] where the heading starts.
    pub offset: usize,
    /// The heading path from the top level down, such as `Setup > Backups`.
    pub path: String,
}

impl Document {
    /// The heading path in effect at a byte offset: the last heading that starts at or before
    /// it. `None` before the first heading, or when the document has no headings.
    pub fn section_at(&self, offset: usize) -> Option<String> {
        let count = self.headings.partition_point(|h| h.offset <= offset);
        count
            .checked_sub(1)
            .map(|index| self.headings[index].path.clone())
    }
}

/// Turns a file into text. Implement this to add support for a format.
pub trait Parser: Send + Sync {
    /// Returns the document's text. Fails if the file cannot be read or decoded.
    fn parse(&self, path: &Path) -> Result<String>;
    /// Returns the text and its headings. The default has no headings; a parser with a heading
    /// structure overrides this so chunks can record the section they belong to.
    fn parse_document(&self, path: &Path) -> Result<Document> {
        Ok(Document {
            text: self.parse(path)?,
            ..Document::default()
        })
    }
    /// File extensions this parser handles, in lowercase and without the dot.
    fn supported_extensions(&self) -> &[&str];
}

/// Reads plain text files.
pub struct PlainTextParser;

impl Parser for PlainTextParser {
    fn parse(&self, path: &Path) -> Result<String> {
        read_text(path)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["txt"]
    }
}

/// Reads Markdown and keeps its text without the markup.
pub struct MarkdownParser;

impl Parser for MarkdownParser {
    fn parse(&self, path: &Path) -> Result<String> {
        Ok(self.parse_document(path)?.text)
    }

    fn parse_document(&self, path: &Path) -> Result<Document> {
        Ok(markdown_document(&read_text(path)?))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown", "mdown"]
    }
}

/// Extracts visible text from HTML, leaving out scripts, styles and other non-visible content.
pub struct HtmlParser;

impl Parser for HtmlParser {
    fn parse(&self, path: &Path) -> Result<String> {
        extract_text_from_html(&read_text(path)?)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["html", "htm"]
    }
}

/// Extracts text from PDF files.
#[cfg(feature = "native")]
pub struct PdfParser;

#[cfg(feature = "native")]
impl Parser for PdfParser {
    fn parse(&self, path: &Path) -> Result<String> {
        Ok(self.parse_document(path)?.text)
    }

    fn parse_document(&self, path: &Path) -> Result<Document> {
        check_size(path, MAX_FILE_BYTES)?;

        // pdf-extract can panic on malformed files. Contain that panic so one bad
        // document returns an error instead of aborting the process or the FFI call.
        let pages = std::panic::catch_unwind(|| pdf_extract::extract_text_by_pages(path))
            .map_err(|_| {
                RagError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "PDF parser failed on a malformed file",
                ))
            })?
            .map_err(|e| {
                RagError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("PDF extraction failed: {e}"),
                ))
            })?;

        let document = document_from_pages(pages);
        // A scanned PDF has pages but no text layer. Report it instead of indexing nothing.
        if document.text.trim().is_empty() {
            return Err(RagError::EmptyDocument(format!(
                "{} (no text layer; scanned PDFs need OCR first)",
                path.display()
            )));
        }
        Ok(document)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
}

/// Word documents. Reads `word/document.xml` from the archive and keeps the body text in
/// document order: one line per paragraph, with tabs and line breaks preserved. Headers,
/// footers, comments and embedded objects are not read.
pub struct DocxParser;

impl Parser for DocxParser {
    fn parse(&self, path: &Path) -> Result<String> {
        use std::io::Read;

        check_size(path, MAX_FILE_BYTES)?;
        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| invalid_data(format!("not a valid DOCX file: {e}")))?;
        let mut xml = String::new();
        archive
            .by_name("word/document.xml")
            .map_err(|e| invalid_data(format!("DOCX has no document body: {e}")))?
            .read_to_string(&mut xml)?;
        Ok(docx_body_text(&xml))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["docx"]
    }
}

fn invalid_data(message: String) -> RagError {
    RagError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message,
    ))
}

/// Extracts the text from WordprocessingML. Text lives in `<w:t>` elements. A `<w:tab/>` is
/// a tab, `<w:br/>` is a line break, and the end of each `<w:p>` paragraph starts a new line.
fn docx_body_text(xml: &str) -> String {
    let mut out = String::new();
    let mut in_text = false;
    let mut rest = xml;
    while let Some(open) = rest.find('<') {
        if in_text {
            out.push_str(&unescape_xml(&rest[..open]));
        }
        let Some(close) = rest[open..].find('>') else {
            break;
        };
        let tag = &rest[open + 1..open + close];
        rest = &rest[open + close + 1..];

        let self_closing = tag.ends_with('/');
        let name = tag
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("");
        match name {
            "w:t" if !self_closing => in_text = true,
            "/w:t" => in_text = false,
            "w:tab" => out.push('\t'),
            "w:br" | "w:cr" => out.push('\n'),
            "/w:p" => out.push('\n'),
            _ => {}
        }
    }

    out.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn unescape_xml(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Renders each CSV row as `header: value; header: value`, which keeps the column names
/// next to the values for the embedding model. Without a header row, cells are joined with
/// ` | `. Quoted fields may contain commas and newlines.
pub struct CsvParser;

impl Parser for CsvParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let rows = parse_csv(&read_text(path)?);
        let mut rows = rows
            .into_iter()
            .filter(|row| row.iter().any(|c| !c.trim().is_empty()));
        let Some(header) = rows.next() else {
            return Ok(String::new());
        };

        let lines: Vec<String> = rows
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(_, cell)| !cell.trim().is_empty())
                    .map(|(i, cell)| match header.get(i).map(|h| h.trim()) {
                        Some(name) if !name.is_empty() => format!("{name}: {}", cell.trim()),
                        _ => cell.trim().to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .filter(|line| !line.is_empty())
            .collect();
        Ok(lines.join("\n"))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["csv"]
    }
}

/// Splits CSV text into rows of cells. Handles `""` escapes inside quoted fields.
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        match (in_quotes, ch) {
            (true, '"') if chars.peek() == Some(&'"') => {
                cell.push('"');
                chars.next();
            }
            (true, '"') => in_quotes = false,
            (true, _) => cell.push(ch),
            (false, '"') => in_quotes = true,
            (false, ',') => row.push(std::mem::take(&mut cell)),
            (false, '\r') => {}
            (false, '\n') => {
                row.push(std::mem::take(&mut cell));
                rows.push(std::mem::take(&mut row));
            }
            (false, _) => cell.push(ch),
        }
    }
    if !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        rows.push(row);
    }
    rows
}

/// Writes JSON as one `path: value` line per leaf, e.g. `user.tags[0]: rust`. Keys carry
/// the structure, so a line still makes sense on its own.
pub struct JsonParser;

impl Parser for JsonParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let value: serde_json::Value = serde_json::from_str(&read_text(path)?).map_err(|e| {
            RagError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid JSON: {e}"),
            ))
        })?;
        let mut lines = Vec::new();
        flatten_json("", &value, &mut lines);
        Ok(lines.join("\n"))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["json"]
    }
}

fn flatten_json(prefix: &str, value: &serde_json::Value, out: &mut Vec<String>) {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_json(&path, child, out);
            }
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                flatten_json(&format!("{prefix}[{i}]"), child, out);
            }
        }
        Value::String(s) => out.push(format!("{prefix}: {s}")),
        Value::Null => {}
        other => out.push(format!("{prefix}: {other}")),
    }
}

/// Source code and configuration files. The text is indexed as written, so identifiers and
/// comments stay searchable.
pub struct SourceParser;

const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "kt", "c", "h", "cc", "cpp", "hpp", "cs",
    "rb", "php", "swift", "sh", "sql", "toml", "yaml", "yml", "ini", "cfg", "log",
];

impl Parser for SourceParser {
    fn parse(&self, path: &Path) -> Result<String> {
        read_text(path)
    }

    fn supported_extensions(&self) -> &[&str] {
        SOURCE_EXTENSIONS
    }
}

/// Every built-in parser. Each file extension is declared by exactly one parser.
fn builtin_parsers() -> Vec<Box<dyn Parser>> {
    #[allow(unused_mut)]
    let mut parsers: Vec<Box<dyn Parser>> = vec![
        Box::new(PlainTextParser),
        Box::new(MarkdownParser),
        Box::new(HtmlParser),
        Box::new(CsvParser),
        Box::new(JsonParser),
        Box::new(DocxParser),
        Box::new(SourceParser),
    ];
    #[cfg(feature = "native")]
    parsers.push(Box::new(PdfParser));
    parsers
}

/// Chooses a parser by file extension. Parsers registered later take precedence, so a
/// custom parser can replace a built-in one for the same extension.
pub struct ParserRegistry {
    parsers: Vec<Box<dyn Parser>>,
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl ParserRegistry {
    /// The parsers that ship with the library.
    pub fn builtin() -> Self {
        Self {
            parsers: builtin_parsers(),
        }
    }

    /// Adds a parser. It is checked before every parser registered earlier.
    pub fn register(&mut self, parser: Box<dyn Parser>) {
        self.parsers.insert(0, parser);
    }

    /// Finds the parser for a path by its extension, ignoring case.
    pub fn for_path(&self, path: &Path) -> Option<&dyn Parser> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        self.parsers
            .iter()
            .find(|p| p.supported_extensions().contains(&ext.as_str()))
            .map(|p| p.as_ref())
    }
}

/// Returns the built-in parser for a path, if one handles its extension.
pub fn get_parser(path: &Path) -> Option<Box<dyn Parser>> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    builtin_parsers()
        .into_iter()
        .find(|p| p.supported_extensions().contains(&ext.as_str()))
}

/// Reads a text file as UTF-8. A leading byte order mark is removed, and invalid bytes
/// become U+FFFD instead of failing the whole file.
fn read_text(path: &Path) -> Result<String> {
    read_text_with_limit(path, MAX_FILE_BYTES)
}

fn read_text_with_limit(path: &Path, limit: u64) -> Result<String> {
    check_size(path, limit)?;
    let bytes = std::fs::read(path)?;
    let body = bytes.strip_prefix(UTF8_BOM).unwrap_or(bytes.as_slice());
    Ok(String::from_utf8_lossy(body).into_owned())
}

fn check_size(path: &Path, limit: u64) -> Result<()> {
    let size = std::fs::metadata(path)?.len();
    if size > limit {
        return Err(RagError::TooLarge {
            path: path.display().to_string(),
            size,
            limit,
        });
    }
    Ok(())
}

/// Keeps the readable text of a Markdown document: headings, paragraphs, list items,
/// code and link text. Markup such as emphasis markers, links and URLs is dropped.
/// Joins the text of each page into one document. Every page with text becomes a section
/// called `page N`, numbered from 1 in the PDF, so a chunk reports the page it starts on.
/// Pages without text add no section, and the chunks after them report the next page with text.
#[cfg(feature = "native")]
fn document_from_pages(pages: Vec<String>) -> Document {
    let mut text = String::new();
    let mut headings = Vec::new();
    for (index, page) in pages.iter().enumerate() {
        let page = page.trim();
        if page.is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        headings.push(Heading {
            offset: text.len(),
            path: format!("page {}", index + 1),
        });
        text.push_str(page);
    }
    Document {
        text,
        headings,
        ..Document::default()
    }
}

/// Parses text that is already in memory, for hosts that cannot read files, such as the browser
/// build. `kind` names the format: `"text"`, `"markdown"` or `"html"`. The result has the same
/// text, headings and boundaries that the matching file parser would give.
pub fn parse_content(kind: &str, content: &str) -> Result<Document> {
    match kind {
        "text" => Ok(Document {
            text: content.to_string(),
            ..Document::default()
        }),
        "markdown" => Ok(markdown_document(content)),
        "html" => Ok(Document {
            text: extract_text_from_html(content)?,
            ..Document::default()
        }),
        other => Err(RagError::UnsupportedFormat(other.to_string())),
    }
}

/// Plain text and headings from Markdown. Each heading's path is the chain of headings above it.
fn markdown_document(source: &str) -> Document {
    let mut out = String::with_capacity(source.len());
    let mut headings = Vec::new();
    // The start and end of every code block, so that no chunk runs from prose into code or
    // from one code block into another.
    let mut boundaries = Vec::new();
    // The headings that enclose the current position, outermost first.
    let mut outline: Vec<(HeadingLevel, String)> = Vec::new();
    // The heading being read: its level, the offset of its text in `out`, and its title so far.
    let mut open: Option<(HeadingLevel, usize, String)> = None;
    for event in CmarkParser::new(source) {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                open = Some((level, out.len(), String::new()));
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some((level, offset, title)) = open.take() {
                    while outline.last().is_some_and(|(outer, _)| *outer >= level) {
                        outline.pop();
                    }
                    let title = title.trim().to_string();
                    outline.push((level, title));
                    let path = outline
                        .iter()
                        .map(|(_, title)| title.as_str())
                        .collect::<Vec<_>>()
                        .join(" > ");
                    headings.push(Heading { offset, path });
                }
                out.push('\n');
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some((_, _, title)) = open.as_mut() {
                    title.push_str(&text);
                }
                out.push_str(&text);
            }
            Event::SoftBreak | Event::HardBreak => out.push('\n'),
            Event::Start(Tag::CodeBlock(_)) => boundaries.push(out.len()),
            Event::End(TagEnd::CodeBlock) => {
                out.push('\n');
                boundaries.push(out.len());
            }
            Event::End(
                TagEnd::Paragraph | TagEnd::Item | TagEnd::BlockQuote(_) | TagEnd::TableRow,
            ) => out.push('\n'),
            _ => {}
        }
    }
    // Trimming the start moves every offset back by the length of the removed whitespace.
    let leading = out.len() - out.trim_start().len();
    for heading in &mut headings {
        heading.offset = heading.offset.saturating_sub(leading);
    }
    for boundary in &mut boundaries {
        *boundary = boundary.saturating_sub(leading);
    }
    Document {
        text: out.trim().to_string(),
        headings,
        boundaries,
    }
}

fn extract_text_from_html(html: &str) -> Result<String> {
    use html5ever::driver::ParseOpts;
    use html5ever::parse_document;
    use html5ever::tendril::TendrilSink;
    use markup5ever_rcdom::RcDom;

    let dom = parse_document(RcDom::default(), ParseOpts::default())
        .from_utf8()
        .read_from(&mut html.as_bytes())?;

    // html5ever repairs malformed markup instead of failing. The repairs are not errors for
    // the reader, but they are logged so that odd documents can be found.
    let errors = dom.errors.borrow();
    if !errors.is_empty() {
        tracing::debug!(
            repairs = errors.len(),
            first = %errors[0],
            "HTML parser repaired malformed markup"
        );
    }

    let mut text = String::new();
    extract_text_from_node(&dom.document, &mut text);
    Ok(text.trim().to_string())
}

/// Tags whose contents are code or metadata, not reader-visible text.
const SKIPPED_HTML_TAGS: [&str; 3] = ["script", "style", "noscript"];

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
            if SKIPPED_HTML_TAGS.contains(&tag) {
                return;
            }
            let block_tags = [
                "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr",
            ];
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

    /// Each call gets its own directory, so parallel tests never share files.
    fn write_temp(name: &str, content: &[u8]) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tpt_rag_parser_test_{}", rand::random::<u64>()));
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
    fn test_plain_text_strips_bom_and_tolerates_invalid_utf8() {
        let path = write_temp("bom.txt", b"\xEF\xBB\xBFstart \xFF end");
        let text = PlainTextParser.parse(&path).unwrap();
        assert!(text.starts_with("start"));
        assert!(text.contains('\u{FFFD}'));
    }

    #[test]
    fn test_text_over_limit_is_rejected() {
        let path = write_temp("big.txt", &[b'a'; 100]);
        let err = read_text_with_limit(&path, 10).unwrap_err();
        assert!(matches!(err, RagError::TooLarge { limit: 10, .. }));
    }

    #[test]
    fn test_markdown_parser() {
        let path = write_temp(
            "test.md",
            b"# Title\n\nSome **bold** text with `code` and [a link](http://x.test).\n\n## Section\n\n- item 1\n- item 2",
        );
        let text = MarkdownParser.parse(&path).unwrap();
        assert!(text.contains("Title"));
        assert!(text.contains("Some bold text with code and a link."));
        assert!(text.contains("item 1\nitem 2"));
        assert!(!text.contains("http://x.test"));
        assert!(!text.contains("**"));
    }

    #[test]
    fn test_html_parser() {
        let path = write_temp(
            "test.html",
            b"<html><head><style>p { color: red }</style></head><body><h1>Title</h1><p>Paragraph one.</p><script>var secret = 1;</script><p>Paragraph two.</p></body></html>",
        );
        let parser = HtmlParser;
        let text = parser.parse(&path).unwrap();
        assert!(text.contains("Title"));
        assert!(text.contains("Paragraph one."));
        assert!(!text.contains("secret"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn test_each_extension_has_one_parser() {
        let mut seen = std::collections::HashSet::new();
        for parser in builtin_parsers() {
            for ext in parser.supported_extensions() {
                assert!(
                    seen.insert(ext.to_string()),
                    "extension {ext} declared twice"
                );
            }
        }
    }

    #[test]
    fn test_get_parser() {
        let p = get_parser(Path::new("foo.pdf"));
        assert!(p.is_some());
        let p = get_parser(Path::new("FOO.MD"));
        assert!(p.is_some());
        let p = get_parser(Path::new("foo.xyz"));
        assert!(p.is_none());
    }

    #[test]
    fn test_csv_rows_keep_column_names() {
        let path = write_temp(
            "people.csv",
            b"name,role\n\"Lovelace, Ada\",mathematician\nAlan,\"says \"\"hi\"\"\"\n",
        );
        let text = CsvParser.parse(&path).unwrap();
        assert_eq!(
            text,
            "name: Lovelace, Ada; role: mathematician\nname: Alan; role: says \"hi\""
        );
    }

    #[test]
    fn test_docx_text_in_document_order() {
        let dir = std::env::temp_dir().join(format!("tpt_rag_docx_test_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("doc.docx");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        zip.start_file(
            "word/document.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(
            br#"<?xml version="1.0"?><w:document><w:body>
            <w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> &amp; welcome</w:t></w:r></w:p>
            <w:p><w:r><w:t>Col A</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>Col B</w:t></w:r><w:r><w:br/></w:r><w:r><w:t>next line</w:t></w:r></w:p>
            </w:body></w:document>"#,
        )
        .unwrap();
        zip.finish().unwrap();

        assert_eq!(
            DocxParser.parse(&path).unwrap(),
            "Hello & welcome\nCol A\tCol B\nnext line"
        );
    }

    #[test]
    fn test_docx_rejects_non_zip_files() {
        let path = write_temp("fake.docx", b"plain text, not a zip");
        assert!(DocxParser.parse(&path).is_err());
    }

    #[test]
    fn test_csv_without_rows_is_empty() {
        let path = write_temp("empty.csv", b"\n\n");
        assert!(CsvParser.parse(&path).unwrap().is_empty());
    }

    #[test]
    fn test_json_flattens_leaves_with_paths() {
        let path = write_temp(
            "data.json",
            br#"{"user": {"name": "Ada", "tags": ["math", "code"]}, "age": 36, "note": null}"#,
        );
        let lines: Vec<String> = JsonParser
            .parse(&path)
            .unwrap()
            .lines()
            .map(String::from)
            .collect();
        assert_eq!(lines.len(), 4);
        assert!(lines.contains(&"user.name: Ada".to_string()));
        assert!(lines.contains(&"user.tags[0]: math".to_string()));
        assert!(lines.contains(&"user.tags[1]: code".to_string()));
        assert!(lines.contains(&"age: 36".to_string()));
    }

    #[test]
    fn test_invalid_json_is_an_error() {
        let path = write_temp("bad.json", b"{ nope");
        assert!(JsonParser.parse(&path).is_err());
    }

    #[test]
    fn test_registry_prefers_later_registrations() {
        struct Custom;
        impl Parser for Custom {
            fn parse(&self, _path: &Path) -> Result<String> {
                Ok("custom".into())
            }
            fn supported_extensions(&self) -> &[&str] {
                &["md"]
            }
        }

        let mut registry = ParserRegistry::builtin();
        assert!(registry.for_path(Path::new("a.md")).is_some());
        assert!(registry.for_path(Path::new("a.exe")).is_none());
        registry.register(Box::new(Custom));
        let path = write_temp("override.md", b"# ignored");
        assert_eq!(
            registry.for_path(&path).unwrap().parse(&path).unwrap(),
            "custom"
        );
    }

    #[test]
    fn markdown_headings_give_section_paths_and_offsets() {
        let path = write_temp(
            "guide.md",
            b"\n# Setup\n\nInstall it.\n\n## Backups\n\nRun nightly.\n\n# Other\n\nFinal.\n",
        );
        let doc = MarkdownParser.parse_document(&path).unwrap();

        assert!(
            doc.text.starts_with("Setup"),
            "leading blank line is trimmed"
        );
        let paths: Vec<&str> = doc.headings.iter().map(|h| h.path.as_str()).collect();
        assert_eq!(paths, ["Setup", "Setup > Backups", "Other"]);
        // Each heading offset points at its own title in the text, after the trim.
        assert!(doc.text[doc.headings[0].offset..].starts_with("Setup"));
        assert!(doc.text[doc.headings[1].offset..].starts_with("Backups"));
        assert!(doc.text[doc.headings[2].offset..].starts_with("Other"));

        assert_eq!(doc.section_at(0), Some("Setup".to_string()));
        let run = doc.text.find("Run nightly").unwrap();
        assert_eq!(doc.section_at(run), Some("Setup > Backups".to_string()));
        let last = doc.text.find("Final").unwrap();
        assert_eq!(doc.section_at(last), Some("Other".to_string()));
    }

    #[test]
    fn markdown_code_block_edges_are_boundaries() {
        let path = write_temp(
            "code.md",
            br#"Intro words here.

```
let x = 1;
```

Outro words."#,
        );
        let doc = MarkdownParser.parse_document(&path).unwrap();
        assert_eq!(doc.boundaries.len(), 2);
        assert!(doc.text[doc.boundaries[0]..].starts_with("let x"));
        assert!(doc.text[..doc.boundaries[1]]
            .trim_end()
            .ends_with("let x = 1;"));
        assert!(doc.text[doc.boundaries[1]..].starts_with("Outro"));
        assert!(doc.headings.is_empty());
    }

    #[test]
    fn documents_without_headings_have_no_section() {
        let path = write_temp("plain.md", b"Just a paragraph.");
        let doc = MarkdownParser.parse_document(&path).unwrap();
        assert!(doc.headings.is_empty());
        assert_eq!(doc.section_at(0), None);
        // Other formats use the default, which has no headings.
        let txt = write_temp("plain.txt", b"Just text.");
        assert_eq!(
            PlainTextParser.parse_document(&txt).unwrap().section_at(0),
            None
        );
    }

    #[test]
    #[cfg(feature = "native")]
    fn pdf_pages_become_sections() {
        let doc = document_from_pages(vec![
            "  First page.  ".to_string(),
            String::new(),
            "Third page.".to_string(),
        ]);
        assert_eq!(doc.text, "First page.\n\nThird page.");
        let paths: Vec<&str> = doc.headings.iter().map(|h| h.path.as_str()).collect();
        assert_eq!(paths, ["page 1", "page 3"]);
        assert_eq!(doc.section_at(0), Some("page 1".to_string()));
        assert_eq!(
            doc.section_at(doc.text.find("Third").unwrap()),
            Some("page 3".to_string())
        );
    }

    #[test]
    #[cfg(feature = "native")]
    fn scanned_pdf_pages_give_no_text() {
        let doc = document_from_pages(vec![String::new(), "  ".to_string()]);
        assert!(doc.text.trim().is_empty());
        assert!(doc.headings.is_empty());
    }
}
