//! Command-line front end for tpt-raglite.
//!
//! The embedding model is found the same way as in the library: `RAGLITE_MODEL`, then
//! `<db>/models/all-MiniLM-L6-v2.onnx`. `RAGLITE_RERANKER` enables `--rerank`.

#[path = "raglite/mcp.rs"]
mod mcp;
#[path = "raglite/models.rs"]
mod models;
#[path = "raglite/serve.rs"]
mod serve;

use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use tpt_rag_core::{QueryFilter, RAGLite, SearchResult, SyncReport};

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
      --diverse <w>            spread the results out: w = 1 is plain relevance, 0 is most varied
  stats <db>                   chunk and vector counts, index size, model
  doctor <db>                  check the database for problems (exit 1 on failure)
  sync <db> <folder>           index new and changed files, remove deleted ones
      --tag <tag>              tag every ingested chunk (repeatable)
  watch <db> <folder>          sync every few seconds until stopped
      --interval <secs>        seconds between syncs (default 30)
  serve <db>                   JSON API over HTTP (no authentication; see README)
      --addr <host:port>       address to listen on (default 127.0.0.1:8080)
      --allow-remote           allow a non-loopback address
      --mcp                    speak MCP over stdin and stdout instead of HTTP
  delete <db> <source>...      remove every chunk from the given sources
  rebuild <db>                 re-embed every chunk and rebuild the index
  eval <db> <cases.jsonl>      score queries: one {\"query\", \"expected_source\"} per line
      -k <n>                   cut-off for recall@k (default 5); --hybrid <alpha> or --rerank <n> score those modes
  models pull [folder]         download the embedding and reranker models (default ./models)
  synth <db>                   print eval cases made from the chunks, as JSONL
      --per-source <n>         cases per source (default 5)
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
        "sync" => sync(rest),
        "watch" => watch(rest),
        "serve" => serve_command(rest),
        "models" => models_command(rest),
        "delete" => delete(rest),
        "rebuild" => rebuild(rest),
        "eval" => eval(rest),
        "synth" => synth(rest),
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
    diverse: Option<f32>,
    rerank: Option<usize>,
    source_prefix: Option<String>,
    since: Option<usize>,
    interval: Option<usize>,
    addr: Option<String>,
    allow_remote: bool,
    mcp: bool,
    per_source: Option<usize>,
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
            "--diverse" => {
                let weight: f32 = value_for(&mut iter, arg)?
                    .parse()
                    .map_err(|_| "--diverse needs a number between 0 and 1".to_string())?;
                if !(0.0..=1.0).contains(&weight) {
                    return Err("--diverse needs a number between 0 and 1".into());
                }
                opts.diverse = Some(weight);
            }
            "--source-prefix" => opts.source_prefix = Some(value_for(&mut iter, arg)?),
            "--since" => opts.since = Some(number_for(&mut iter, arg)?),
            "--interval" => opts.interval = Some(number_for(&mut iter, arg)?),
            "--addr" => opts.addr = Some(value_for(&mut iter, arg)?),
            "--allow-remote" => opts.allow_remote = true,
            "--mcp" => opts.mcp = true,
            "--per-source" => opts.per_source = Some(number_for(&mut iter, arg)?),
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
        println!(
            "{target}: {} added, {} failed",
            report.added.len(),
            report.failed.len()
        );
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
    let filtered =
        !filter.tags.is_empty() || filter.source_prefix.is_some() || filter.created_since.is_some();

    let results: Vec<SearchResult> = if let Some(weight) = opts.diverse {
        if filtered || opts.hybrid.is_some() || opts.rerank.is_some() {
            return Err(
                "--diverse cannot be combined with --hybrid, --rerank, --tag, --source-prefix or --since"
                    .into(),
            );
        }
        rag.query_diverse(&question, top_k, weight)?
    } else {
        match (opts.hybrid, opts.rerank) {
            (_, Some(_)) if filtered => {
                return Err(
                    "--rerank cannot be combined with --tag, --source-prefix or --since".into(),
                )
            }
            (_, Some(n)) => rag.query_reranked(&question, top_k, n)?,
            (Some(alpha), None) => rag.query_hybrid(&question, top_k, alpha)?,
            (None, None) if filtered => rag.query_filtered(&question, top_k, &filter)?,
            (None, None) => rag.query(&question, top_k)?,
        }
    };

    if results.is_empty() {
        println!("no matching chunks");
    }
    for r in &results {
        let section = r
            .section
            .as_deref()
            .map(|s| format!(" | {s}"))
            .unwrap_or_default();
        let span = match (r.start_offset, r.end_offset) {
            (Some(start), Some(end)) => format!(" (bytes {start}-{end})"),
            _ => String::new(),
        };
        println!("[{:.3}] {}{section}{span}", r.score, r.source);
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

/// Prints one eval case per line as `{"query", "expected_source"}`, the format `eval` reads.
fn synth(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, _) = database_and_rest(&opts)?;
    let rag = open(db)?;
    let cases = rag.eval_cases(opts.per_source.unwrap_or(5))?;
    if cases.is_empty() {
        return Err("no chunk has enough words to make a query".into());
    }
    for case in &cases {
        let line = serde_json::json!({
            "query": case.query,
            "expected_source": case.expected_source,
        });
        println!("{line}");
    }
    Ok(ExitCode::SUCCESS)
}

/// `models pull [folder]`: downloads the models into `folder` (default `models`).
fn models_command(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    if opts.positional.first().map(String::as_str) != Some("pull") {
        return Err("usage: raglite models pull [folder]".into());
    }
    let dir = Path::new(
        opts.positional
            .get(1)
            .map(String::as_str)
            .unwrap_or("models"),
    );
    let report = models::pull(dir, models::MODEL_FILES)?;
    for name in &report.downloaded {
        println!("downloaded {name}");
    }
    for name in &report.kept {
        println!("kept       {name}");
    }
    println!("models are in {}", dir.display());
    println!("point RAGLITE_MODEL and RAGLITE_RERANKER at them, or copy them into <db>/models/");
    Ok(ExitCode::SUCCESS)
}

fn serve_command(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, _) = database_and_rest(&opts)?;
    let rag = open(db)?;
    if opts.mcp {
        if opts.addr.is_some() || opts.allow_remote {
            return Err("--mcp cannot be combined with --addr or --allow-remote".into());
        }
        mcp::run(&rag)?;
        return Ok(ExitCode::SUCCESS);
    }
    let addr = opts.addr.as_deref().unwrap_or("127.0.0.1:8080");
    serve::serve(&rag, addr, opts.allow_remote)?;
    Ok(ExitCode::SUCCESS)
}

fn sync(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, rest) = database_and_rest(&opts)?;
    let Some(folder) = rest.first() else {
        return Err("sync needs a folder".into());
    };
    let rag = open(db)?;
    let tags: Vec<&str> = opts.tags.iter().map(String::as_str).collect();
    let report = rag.sync_directory(Path::new(folder), &tags)?;
    print_sync(&report);
    Ok(if report.failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// Syncs the folder on a fixed interval until the process is stopped. Changes and failures are
/// printed as they happen; a file that keeps failing is reported on every sync.
fn watch(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, rest) = database_and_rest(&opts)?;
    let Some(folder) = rest.first() else {
        return Err("watch needs a folder".into());
    };
    let rag = open(db)?;
    let tags: Vec<&str> = opts.tags.iter().map(String::as_str).collect();
    let interval = Duration::from_secs(opts.interval.unwrap_or(30) as u64);
    loop {
        let report = rag.sync_directory(Path::new(folder), &tags)?;
        if !report.indexed.is_empty() || !report.removed.is_empty() || !report.failed.is_empty() {
            print_sync(&report);
        }
        std::thread::sleep(interval);
    }
}

fn print_sync(report: &SyncReport) {
    for path in &report.indexed {
        println!("indexed {}", path.display());
    }
    for source in &report.removed {
        println!("removed {source}");
    }
    for (path, reason) in &report.failed {
        println!("failed {}: {reason}", path.display());
    }
    println!(
        "{} indexed, {} unchanged, {} removed, {} failed",
        report.indexed.len(),
        report.unchanged,
        report.removed.len(),
        report.failed.len()
    );
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
/// has a source containing `expected_source`. Reports recall@k, mean reciprocal rank, and nDCG@k.
/// Each case has one relevant source, so the ideal DCG is 1 and a hit at `rank` (0-based) scores
/// `1 / log2(rank + 2)`.
fn eval(args: &[String]) -> CliResult {
    let opts = parse_options(args)?;
    let (db, rest) = database_and_rest(&opts)?;
    let Some(file) = rest.first() else {
        return Err("eval needs a JSONL file of cases".into());
    };
    let k = opts.top_k.unwrap_or(5);
    let rag = open(db)?;
    let text = std::fs::read_to_string(file)?;

    let (mut total, mut hits, mut reciprocal_rank_sum, mut ndcg_sum) =
        (0usize, 0usize, 0.0f64, 0.0f64);
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
        let results = match (opts.hybrid, opts.rerank) {
            (_, Some(n)) => rag.query_reranked(query, k, n)?,
            (Some(alpha), None) => rag.query_hybrid(query, k, alpha)?,
            (None, None) => rag.query(query, k)?,
        };
        match results.iter().position(|r| r.source.contains(expected)) {
            Some(rank) => {
                hits += 1;
                reciprocal_rank_sum += 1.0 / (rank as f64 + 1.0);
                ndcg_sum += 1.0 / (rank as f64 + 2.0).log2();
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
    println!("nDCG@{k}:   {:.3}", ndcg_sum / total as f64);
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
