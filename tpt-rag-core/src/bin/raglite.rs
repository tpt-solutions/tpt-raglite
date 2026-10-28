//! Command-line front end for tpt-raglite.
//!
//! The embedding model is found the same way as in the library: `RAGLITE_MODEL`, then
//! `<db>/models/all-MiniLM-L6-v2.onnx`. `RAGLITE_RERANKER` enables `--rerank`.

use std::path::Path;
use std::process::ExitCode;

use tpt_rag_core::{QueryFilter, RAGLite, SearchResult};

const USAGE: &str = "\
usage: raglite <command> <db> [arguments]

commands:
  ingest <db> <path|glob>...   add files, folders (recursive) or glob matches
      --tag <tag>              tag every ingested chunk (repeatable)
  query <db> <question...>     search the database
      -k <n>                   number of results (default 5)
      --hybrid <alpha>         vector + keyword search, alpha = vector weight
      --rerank <n>             rerank the best n vector matches with the cross-encoder
      --tag <tag>              only chunks with this tag (repeatable)
      --source-prefix <text>   only chunks whose source starts with this text
      --since <epoch>          only chunks ingested at or after this Unix time
  stats <db>                   chunk and vector counts, index size, model
  doctor <db>                  check the database for problems (exit 1 on failure)
  delete <db> <source>...      remove every chunk from the given sources
  rebuild <db>                 re-embed every chunk and rebuild the index
  eval <db> <cases.jsonl>      score queries: one {"query", "expected_source"} per line
      -k <n>                   cut-off for recall@k (default 5)
  help                         show this message";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type CliResult = Result<ExitCode, Box<dyn std::error::Error>>;

fn run(args: Vec<String>) -> CliResult {
    let Some((command, rest)) = args.split_first() else {
        eprintln!("{USAGE}");
        return Ok(ExitCode::from(2));
    };
    match command.as_str() {
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        "ingest" => ingest(rest),
        "query" => query(rest),
        "stats" => stats(rest),
        "doctor" => doctor(rest),
        "delete" => delete(rest),
        "rebuild" => rebuild(rest),
        "eval" => eval(rest),
        other => {
            eprintln!("unknown command {other:?}\n\n{USAGE}");
            Ok(ExitCode::from(2))
        }
    }
}

/// Options and positional arguments, parsed from one command's arguments.
#[derive(Default)]
struct Options {
    positional: Vec<String>,
    tags: Vec<String>,
    top_k: Option<usize>,
    hybrid: Option<f32>,
    rerank: Option<usize>,
    source_prefix: Option<String>,
    since: Option<usize>,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut opts = Options::default();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--tag" => opts.tags.push(value_for(&mut iter, arg)?),
            "-k" | "--top-k" => opts.top_k = Some(number_for(&mut iter, arg)?),
            "--rerank" => opts.rerank = Some(number_for(&mut iter, arg)?),
            "--hybrid" => {
                let alpha: f32 = value_for(&mut iter, arg)?
                    .parse()
                    .map_err(|_| "--hybrid needs a number between 0 and 1".to_string())?;
                opts.hybrid = Some(alpha);
            }
            "--source-prefix" => opts.source_prefix = Some(value_for(&mut iter, arg)?),
            "--since" => opts.since = Some(number_for(&mut iter, arg)?),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => opts.positional.push(arg.clone()),
        }
    }
    Ok(opts)
}

fn value_for(iter: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<String, String> {
    iter.next()
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn number_for(iter: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<usize, String> {
    value_for(iter, flag)?
        .parse()
        .map_err(|_| format!("{flag} needs a whole number"))
}

/// Splits `<db> <rest...>` and returns the database path and the remaining positionals.
fn database_and_rest(opts: &Options) -> Result<(&str, &[String]), String> {
    match opts.positional.split_first() {
        Some((db, rest)) => Ok((db.as_str(), rest)),
        None => Err("missing database folder".into()),
    }
}

fn open(db: &str) -> Result<RAGLite, Box<dyn std::error::Error>> {
    Ok(RAGLite::open(Path::new(db))?)
}

fn ingest(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, paths) = database_and_rest(&opts)?;
    if paths.is_empty() {
        return Err("ingest needs at least one path or glob".into());
    }
    let rag = open(db)?;
    let tags: Vec<&str> = opts.tags.iter().map(String::as_str).collect();
    let mut progress = |i: usize, total: usize, path: &Path| {
        eprintln!("[{i}/{total}] {}", path.display());
    };

    let mut failed = 0;
    for target in paths {
        let report = if Path::new(target).is_dir() {
            rag.add_directory(Path::new(target), &tags, &mut progress)?
        } else if target.contains(['*', '?', '[']) {
            rag.add_glob(target, &tags, &mut progress)?
        } else {
            rag.add_document(Path::new(target), &tags)?;
            println!("added {target}");
            continue;
        };
        println!("{target}: {} added, {} failed", report.added.len(), report.failed.len());
        for (path, reason) in &report.failed {
            println!("  failed {}: {reason}", path.display());
        }
        failed += report.failed.len();
    }
    rag.flush()?;
    Ok(if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn query(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, words) = database_and_rest(&opts)?;
    if words.is_empty() {
        return Err("query needs a question".into());
    }
    let question = words.join(" ");
    let rag = open(db)?;
    let top_k = opts.top_k.unwrap_or(5);

    let filter = QueryFilter {
        tags: opts.tags.clone(),
        source_prefix: opts.source_prefix.clone(),
        created_since: opts.since.map(|s| s as u64),
    };
    let filtered = !filter.tags.is_empty()
        || filter.source_prefix.is_some()
        || filter.created_since.is_some();

    let results: Vec<SearchResult> = match (opts.hybrid, opts.rerank) {
        (_, Some(_)) if filtered => {
            return Err("--rerank cannot be combined with --tag, --source-prefix or --since".into())
        }
        (_, Some(n)) => rag.query_reranked(&question, top_k, n)?,
        (Some(alpha), None) => rag.query_hybrid(&question, top_k, alpha)?,
        (None, None) if filtered => rag.query_filtered(&question, top_k, &filter)?,
        (None, None) => rag.query(&question, top_k)?,
    };

    if results.is_empty() {
        println!("no matching chunks");
    }
    for r in &results {
        println!("[{:.3}] {}", r.score, r.source);
        println!("        {}", preview(&r.text, 160));
    }
    Ok(ExitCode::SUCCESS)
}

/// One line of text, cut to `max` characters.
fn preview(text: &str, max: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}…")
    }
}

fn stats(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, _) = database_and_rest(&opts)?;
    let stats = open(db)?.stats()?;
    println!("chunks:      {}", stats.chunks);
    println!("vectors:     {}", stats.vectors);
    println!("index bytes: {}", stats.index_bytes);
    println!("model:       {}", stats.model);
    Ok(ExitCode::SUCCESS)
}

fn doctor(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, _) = database_and_rest(&opts)?;
    let checks = open(db)?.check()?;
    let mut all_ok = true;
    for check in &checks {
        let mark = if check.ok { "ok  " } else { "FAIL" };
        println!("{mark} {:<18} {}", check.name, check.detail);
        all_ok &= check.ok;
    }
    Ok(if all_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn delete(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, sources) = database_and_rest(&opts)?;
    if sources.is_empty() {
        return Err("delete needs at least one source".into());
    }
    let rag = open(db)?;
    for source in sources {
        let removed = rag.delete_document(source)?;
        println!("{source}: removed {removed} chunk(s)");
    }
    rag.flush()?;
    Ok(ExitCode::SUCCESS)
}

/// Scores retrieval against labelled cases. A case is a hit when one of the top `k` results
/// has a source containing `expected_source`. Reports recall@k and mean reciprocal rank.
fn eval(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, rest) = database_and_rest(&opts)?;
    let Some(file) = rest.first() else {
        return Err("eval needs a JSONL file of cases".into());
    };
    let k = opts.top_k.unwrap_or(5);
    let rag = open(db)?;
    let text = std::fs::read_to_string(file)?;

    let (mut total, mut hits, mut reciprocal_rank_sum) = (0usize, 0usize, 0.0f64);
    let mut misses = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_no = line_no + 1;
        let case: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("line {line_no}: {e}"))?;
        let query = case["query"]
            .as_str()
            .ok_or_else(|| format!("line {line_no}: missing \"query\""))?;
        let expected = case["expected_source"]
            .as_str()
            .ok_or_else(|| format!("line {line_no}: missing \"expected_source\""))?;

        total += 1;
        let results = rag.query(query, k)?;
        match results.iter().position(|r| r.source.contains(expected)) {
            Some(rank) => {
                hits += 1;
                reciprocal_rank_sum += 1.0 / (rank as f64 + 1.0);
            }
            None => misses.push(query.to_string()),
        }
    }
    if total == 0 {
        return Err("the file has no cases".into());
    }

    println!("cases:    {total}");
    println!("recall@{k}: {:.3}", hits as f64 / total as f64);
    println!("MRR:      {:.3}", reciprocal_rank_sum / total as f64);
    for query in misses.iter().take(10) {
        println!("  miss: {query}");
    }
    Ok(ExitCode::SUCCESS)
}

fn rebuild(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, _) = database_and_rest(&opts)?;
    let count = open(db)?.rebuild_index()?;
    println!("re-embedded {count} chunk(s)");
    Ok(ExitCode::SUCCESS)
}
