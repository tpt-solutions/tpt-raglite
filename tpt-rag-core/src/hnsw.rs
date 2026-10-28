use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use ordered_float::NotNan;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::embedding::{dot, normalize};
use crate::error::{RagError, Result};

/// Highest layer a node can be assigned to.
const MAX_LEVEL: usize = 16;
const MAGIC: &[u8; 8] = b"TPTHNSW\0";
const FORMAT_VERSION: u64 = 4;

pub struct HnswIndex {
    dim: usize,
    m: usize,
    ef_construction: usize,
    /// Random id chosen when the index is (re)built. The metadata store records the same
    /// value, so an index file left over from a rebuild that did not finish is detected.
    epoch: u64,
    /// `layers[l]` maps a node id to its neighbor ids on layer `l`. A node is present
    /// on every layer from 0 up to its own level.
    layers: Vec<HashMap<usize, Vec<usize>>>,
    entry_point: Option<usize>,
    max_layer: usize,
    vectors: Vec<Vec<f32>>,
    /// Removed ids. They stay in the graph so routing through them still works, but
    /// search never returns them.
    deleted: HashSet<usize>,
    rng: StdRng,
}

/// Ordered by distance, so a `BinaryHeap<Candidate>` has the farthest candidate on top
/// and a `BinaryHeap<Reverse<Candidate>>` has the closest one on top.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    distance: NotNan<f32>,
    id: usize,
}

/// Cosine distance as a negated similarity: smaller means closer.
/// Stored vectors are unit length (see `insert` and `search`), so the dot product is the
/// cosine similarity and the negated dot product orders them as a distance.
fn distance(a: &[f32], b: &[f32]) -> NotNan<f32> {
    NotNan::new(-dot(a, b)).unwrap_or_default()
}

impl HnswIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            dim,
            // A level multiplier of 1 / ln(M) needs M >= 2.
            m: m.max(2),
            ef_construction,
            epoch: rand::random(),
            layers: Vec::new(),
            entry_point: None,
            max_layer: 0,
            vectors: Vec::new(),
            deleted: HashSet::new(),
            rng: StdRng::from_entropy(),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Ids of removed vectors, which search never returns.
    pub fn removed_count(&self) -> usize {
        self.deleted.len()
    }

    /// Hides `id` from search results. Returns `false` if the id is unknown or was already
    /// removed.
    pub fn remove(&mut self, id: usize) -> bool {
        id < self.vectors.len() && self.deleted.insert(id)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn insert(&mut self, mut vector: Vec<f32>) -> Result<usize> {
        self.check_dim(vector.len())?;
        normalize(&mut vector);
        let id = self.vectors.len();
        let level = self.random_level();

        let Some(entry) = self.entry_point else {
            self.vectors.push(vector);
            self.add_node(id, level);
            self.entry_point = Some(id);
            self.max_layer = level;
            return Ok(id);
        };

        // Greedy descent through the layers above the new node's level.
        let mut current = entry;
        for layer in (level + 1..=self.max_layer).rev() {
            current = self.closest(&vector, current, layer);
        }

        // Find neighbors on every layer the new node joins, top-down.
        let top = level.min(self.max_layer);
        let mut plans = Vec::with_capacity(top + 1);
        for layer in (0..=top).rev() {
            let candidates = self.search_layer(&vector, current, self.ef_construction, layer);
            let selected = self.select_neighbors(&candidates, self.m);
            current = candidates.first().map_or(current, |c| c.id);
            plans.push((layer, selected));
        }

        self.vectors.push(vector);
        self.add_node(id, level);
        for (layer, selected) in plans {
            for &neighbor in &selected {
                self.layers[layer].entry(neighbor).or_default().push(id);
                self.shrink_connections(layer, neighbor);
            }
            self.layers[layer].insert(id, selected);
        }

        if level > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = level;
        }

        Ok(id)
    }

    /// Returns up to `top_k` `(id, cosine similarity)` pairs, best first.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<(usize, f32)>> {
        self.check_dim(query.len())?;
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        let mut unit_query = query.to_vec();
        normalize(&mut unit_query);
        let query = unit_query.as_slice();
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let mut current = entry;
        for layer in (1..=self.max_layer).rev() {
            current = self.closest(query, current, layer);
        }

        // Removed ids can take slots in the candidate list, so widen the search by their count.
        let ef = ef_search.max(top_k).saturating_add(self.deleted.len());
        let candidates = self.search_layer(query, current, ef, 0);
        Ok(candidates
            .into_iter()
            .filter(|c| !self.deleted.contains(&c.id))
            .take(top_k)
            .map(|c| (c.id, -c.distance.into_inner()))
            .collect())
    }

    fn check_dim(&self, len: usize) -> Result<()> {
        if len != self.dim {
            return Err(RagError::Hnsw(format!(
                "Vector dimension mismatch: index has {}, got {}",
                self.dim, len
            )));
        }
        Ok(())
    }

    fn closest(&self, query: &[f32], entry: usize, layer: usize) -> usize {
        self.search_layer(query, entry, 1, layer)
            .first()
            .map_or(entry, |c| c.id)
    }

    fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            2 * self.m
        } else {
            self.m
        }
    }

    fn add_node(&mut self, id: usize, level: usize) {
        while self.layers.len() <= level {
            self.layers.push(HashMap::new());
        }
        for layer in &mut self.layers[..=level] {
            layer.insert(id, Vec::new());
        }
    }

    /// Best-first search on one layer. Returns up to `ef` candidates, closest first.
    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<Candidate> {
        let ef = ef.max(1);
        let entry_candidate = Candidate {
            distance: distance(query, &self.vectors[entry]),
            id: entry,
        };

        let mut visited = HashSet::new();
        visited.insert(entry);
        let mut to_visit = BinaryHeap::new();
        to_visit.push(Reverse(entry_candidate));
        let mut results = BinaryHeap::new();
        results.push(entry_candidate);

        while let Some(Reverse(current)) = to_visit.pop() {
            if let Some(worst) = results.peek() {
                if results.len() >= ef && current.distance > worst.distance {
                    break;
                }
            }

            let Some(neighbors) = self.layers.get(layer).and_then(|l| l.get(&current.id)) else {
                continue;
            };
            for &neighbor in neighbors {
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = Candidate {
                    distance: distance(query, &self.vectors[neighbor]),
                    id: neighbor,
                };
                let improves = results
                    .peek()
                    .is_none_or(|worst| candidate.distance < worst.distance);
                if results.len() < ef || improves {
                    to_visit.push(Reverse(candidate));
                    results.push(candidate);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.into_sorted_vec()
    }

    /// Heuristic from the HNSW paper: keep a candidate only if it is closer to the base
    /// node than to every neighbor already kept, which spreads edges across directions.
    /// `candidates` must be sorted closest first. Rejected candidates fill any remaining
    /// slots so that nodes keep their degree.
    fn select_neighbors(&self, candidates: &[Candidate], max_count: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(max_count);
        let mut rejected: Vec<usize> = Vec::new();
        for candidate in candidates {
            if selected.len() >= max_count {
                break;
            }
            let diverse = selected.iter().all(|&s| {
                distance(&self.vectors[candidate.id], &self.vectors[s]) > candidate.distance
            });
            if diverse {
                selected.push(candidate.id);
            } else {
                rejected.push(candidate.id);
            }
        }
        for id in rejected {
            if selected.len() >= max_count {
                break;
            }
            selected.push(id);
        }
        selected
    }

    /// Re-prunes a node's neighbor list on `layer` once it grows past the connection cap.
    fn shrink_connections(&mut self, layer: usize, node: usize) {
        let max_conn = self.max_connections(layer);
        let Some(neighbors) = self.layers[layer].get(&node) else {
            return;
        };
        if neighbors.len() <= max_conn {
            return;
        }

        let base = &self.vectors[node];
        let mut candidates: Vec<Candidate> = neighbors
            .iter()
            .map(|&n| Candidate {
                distance: distance(base, &self.vectors[n]),
                id: n,
            })
            .collect();
        candidates.sort();
        let kept = self.select_neighbors(&candidates, max_conn);
        self.layers[layer].insert(node, kept);
    }

    fn random_level(&mut self) -> usize {
        // Level sampled from an exponential distribution with mL = 1 / ln(M).
        let u = self.rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let level = (-u.ln() / (self.m as f64).ln()).floor() as usize;
        level.min(MAX_LEVEL)
    }

    /// Writes the index atomically: data goes to a temp file, is synced, then renamed
    /// over `path`, so a crash never leaves a half-written index behind.
    pub fn flush(&self, path: &Path) -> Result<()> {
        let tmp_path = path.with_extension("bin.tmp");
        {
            let file = std::fs::File::create(&tmp_path)?;
            let mut writer = HashingWriter::new(BufWriter::new(file));

            writer.write_all(MAGIC)?;
            write_u64(&mut writer, FORMAT_VERSION)?;
            write_u64(&mut writer, self.dim as u64)?;
            write_u64(&mut writer, self.m as u64)?;
            write_u64(&mut writer, self.ef_construction as u64)?;
            write_u64(&mut writer, self.vectors.len() as u64)?;
            write_u64(&mut writer, self.max_layer as u64)?;
            write_u64(&mut writer, self.entry_point.map_or(u64::MAX, |e| e as u64))?;
            write_u64(&mut writer, self.epoch)?;

            write_u64(&mut writer, self.layers.len() as u64)?;
            for layer in &self.layers {
                write_u64(&mut writer, layer.len() as u64)?;
                for (&id, neighbors) in layer {
                    write_u64(&mut writer, id as u64)?;
                    write_u64(&mut writer, neighbors.len() as u64)?;
                    for &nid in neighbors {
                        write_u64(&mut writer, nid as u64)?;
                    }
                }
            }

            for vector in &self.vectors {
                for &v in vector {
                    writer.write_all(&v.to_le_bytes())?;
                }
            }

            let mut deleted: Vec<usize> = self.deleted.iter().copied().collect();
            deleted.sort_unstable();
            write_u64(&mut writer, deleted.len() as u64)?;
            for id in deleted {
                write_u64(&mut writer, id as u64)?;
            }

            let checksum = writer.hash;
            let mut file = writer.inner;
            file.write_all(&checksum.to_le_bytes())?;
            file.flush()?;
            file.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Loads an index written by [`HnswIndex::flush`], validating the header, the
    /// checksum and every node and neighbor id before the index is returned.
    pub fn load_from_disk(
        path: &Path,
        dim: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = HashingReader::new(BufReader::new(file));

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(corrupt("not an HNSW index file"));
        }
        let version = read_u64(&mut reader)?;
        if version != FORMAT_VERSION {
            return Err(corrupt(&format!("unsupported format version {version}")));
        }

        let file_dim = read_u64(&mut reader)? as usize;
        let file_m = read_u64(&mut reader)? as usize;
        let _file_ef = read_u64(&mut reader)?;
        if file_dim != dim {
            return Err(RagError::Hnsw(format!(
                "Dimension mismatch: file has {}, expected {}",
                file_dim, dim
            )));
        }
        if file_m != m.max(2) {
            return Err(RagError::Hnsw(format!(
                "M mismatch: file has {}, expected {}",
                file_m, m
            )));
        }

        let vector_count = read_u64(&mut reader)? as usize;
        let max_layer = read_u64(&mut reader)? as usize;
        let ep_raw = read_u64(&mut reader)?;
        let entry_point = if ep_raw == u64::MAX {
            None
        } else {
            Some(ep_raw as usize)
        };
        if entry_point.is_some_and(|ep| ep >= vector_count) {
            return Err(corrupt("entry point out of range"));
        }
        let epoch = read_u64(&mut reader)?;

        let layer_count = read_u64(&mut reader)? as usize;
        if layer_count == 0 || layer_count > MAX_LEVEL + 1 || max_layer >= layer_count {
            return Err(corrupt("invalid layer count"));
        }

        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let node_count = read_u64(&mut reader)? as usize;
            if node_count > vector_count {
                return Err(corrupt("layer has more nodes than vectors"));
            }
            let mut layer = HashMap::with_capacity(node_count.min(4096));
            for _ in 0..node_count {
                let id = read_index(&mut reader, vector_count)?;
                let neighbor_count = read_u64(&mut reader)? as usize;
                if neighbor_count > vector_count {
                    return Err(corrupt("neighbor list longer than index"));
                }
                let mut neighbors = Vec::with_capacity(neighbor_count.min(4096));
                for _ in 0..neighbor_count {
                    neighbors.push(read_index(&mut reader, vector_count)?);
                }
                layer.insert(id, neighbors);
            }
            layers.push(layer);
        }

        // Vectors are read one at a time instead of pre-allocating `vector_count`, so a
        // corrupt count fails on EOF rather than exhausting memory.
        let mut vectors = Vec::new();
        let mut buf = vec![0u8; dim * 4];
        for _ in 0..vector_count {
            reader.read_exact(&mut buf)?;
            let vector = buf
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            vectors.push(vector);
        }

        let deleted_count = read_u64(&mut reader)? as usize;
        if deleted_count > vector_count {
            return Err(corrupt("more removed ids than vectors"));
        }
        let mut deleted = HashSet::with_capacity(deleted_count.min(4096));
        for _ in 0..deleted_count {
            deleted.insert(read_index(&mut reader, vector_count)?);
        }

        let computed = reader.hash;
        let mut file = reader.inner;
        let mut stored = [0u8; 8];
        file.read_exact(&mut stored)?;
        if u64::from_le_bytes(stored) != computed {
            return Err(corrupt("checksum mismatch"));
        }
        let mut trailing = [0u8; 1];
        if file.read(&mut trailing)? != 0 {
            return Err(corrupt("unexpected trailing data"));
        }

        Ok(Self {
            dim,
            m: file_m,
            ef_construction,
            epoch,
            layers,
            entry_point,
            max_layer,
            vectors,
            deleted,
            rng: StdRng::from_entropy(),
        })
    }
}

fn corrupt(reason: &str) -> RagError {
    RagError::Hnsw(format!("corrupt index file: {reason}"))
}

/// Reads a node id and checks that it refers to an existing vector.
fn read_index(reader: &mut impl Read, vector_count: usize) -> Result<usize> {
    let id = read_u64(reader)? as usize;
    if id >= vector_count {
        return Err(corrupt("node id out of range"));
    }
    Ok(id)
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(FNV_PRIME))
}

/// Writer that keeps a running checksum of everything written through it.
struct HashingWriter<W: Write> {
    inner: W,
    hash: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hash: FNV_OFFSET,
        }
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hash = fnv1a(self.hash, &buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Reader that keeps a running checksum of everything read through it.
struct HashingReader<R: Read> {
    inner: R,
    hash: u64,
}

impl<R: Read> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hash: FNV_OFFSET,
        }
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hash = fnv1a(self.hash, &buf[..n]);
        Ok(n)
    }
}

fn write_u64(f: &mut impl Write, v: u64) -> Result<()> {
    f.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_u64(f: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::cosine_similarity;

    fn random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    fn temp_file(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("tpt_hnsw_{}_{}.bin", name, rand::random::<u32>()))
    }

    #[test]
    fn test_insert_and_search() {
        // M=2 with a tiny ef_construction gives a sparse graph that misses the exact
        // nearest neighbor about a quarter of the time, so use realistic parameters.
        let mut index = HnswIndex::new(4, 8, 32);
        let vectors = random_vectors(100, 4);
        for v in &vectors {
            index.insert(v.clone()).unwrap();
        }
        assert_eq!(index.len(), 100);

        let query = &vectors[0];
        let results = index.search(query, 5, 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // closest to itself
    }

    #[test]
    fn test_recall_synthetic() {
        let mut index = HnswIndex::new(32, 16, 200);
        let vectors = random_vectors(500, 32);
        for v in &vectors {
            index.insert(v.clone()).unwrap();
        }

        let mut correct = 0;
        let total = 50;
        let mut rng = rand::thread_rng();
        for _ in 0..total {
            let qi = rng.gen_range(0..500);
            let results = index.search(&vectors[qi], 10, 100).unwrap();

            let mut brute_force: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_similarity(&vectors[qi], v)))
                .collect();
            brute_force.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_10_bf: std::collections::HashSet<usize> =
                brute_force.iter().take(10).map(|(id, _)| *id).collect();

            let top_10_hnsw: std::collections::HashSet<usize> =
                results.iter().map(|(id, _)| *id).collect();

            let intersection = top_10_bf.intersection(&top_10_hnsw).count();
            correct += intersection;
        }

        let recall = correct as f64 / (total * 10) as f64;
        assert!(recall > 0.95, "Recall was {recall}, expected > 0.95");
    }

    #[test]
    fn test_layers_are_nested() {
        let mut index = HnswIndex::new(8, 4, 16);
        for v in random_vectors(300, 8) {
            index.insert(v).unwrap();
        }
        assert!(index.layers.len() > 1, "expected multiple layers");
        for l in 1..index.layers.len() {
            for id in index.layers[l].keys() {
                assert!(
                    index.layers[l - 1].contains_key(id),
                    "node {id} missing from layer {}",
                    l - 1
                );
            }
        }
    }

    #[test]
    fn test_dimension_mismatch_is_error() {
        let mut index = HnswIndex::new(4, 2, 4);
        assert!(index.insert(vec![0.0; 3]).is_err());
        assert!(index.search(&[0.0; 5], 1, 1).is_err());
        assert!(index.is_empty());
    }

    #[test]
    fn test_empty_index_search() {
        let index = HnswIndex::new(4, 2, 4);
        let results = index.search(&[0.0; 4], 5, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_flush_and_load_round_trip() {
        let mut index = HnswIndex::new(16, 8, 64);
        let vectors = random_vectors(200, 16);
        for v in &vectors {
            index.insert(v.clone()).unwrap();
        }

        let path = temp_file("roundtrip");
        index.flush(&path).unwrap();
        let loaded = HnswIndex::load_from_disk(&path, 16, 8, 64).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.len(), index.len());
        let before = index.search(&vectors[7], 5, 50).unwrap();
        let after = loaded.search(&vectors[7], 5, 50).unwrap();
        assert_eq!(
            before.iter().map(|r| r.0).collect::<Vec<_>>(),
            after.iter().map(|r| r.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_removed_ids_are_not_returned_and_stay_removed() {
        let mut index = HnswIndex::new(16, 8, 64);
        let vectors = random_vectors(100, 16);
        for v in &vectors {
            index.insert(v.clone()).unwrap();
        }
        assert!(index.remove(7));
        assert!(!index.remove(7), "second removal should report no change");
        assert!(!index.remove(1000), "unknown id should report no change");

        let results = index.search(&vectors[7], 10, 50).unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|(id, _)| *id != 7));

        let path = temp_file("tombstone");
        index.flush(&path).unwrap();
        let loaded = HnswIndex::load_from_disk(&path, 16, 8, 64).unwrap();
        std::fs::remove_file(&path).ok();
        let reloaded = loaded.search(&vectors[7], 10, 50).unwrap();
        assert!(reloaded.iter().all(|(id, _)| *id != 7));
    }

    #[test]
    fn test_epoch_survives_flush_and_load() {
        let mut index = HnswIndex::new(4, 4, 16);
        index.insert(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let epoch = index.epoch();

        let path = temp_file("epoch");
        index.flush(&path).unwrap();
        let loaded = HnswIndex::load_from_disk(&path, 4, 4, 16).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.epoch(), epoch);
    }

    #[test]
    fn test_load_detects_corruption() {
        let mut index = HnswIndex::new(4, 4, 16);
        for v in random_vectors(20, 4) {
            index.insert(v).unwrap();
        }
        let path = temp_file("corrupt");
        index.flush(&path).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let result = HnswIndex::load_from_disk(&path, 4, 4, 16);
        std::fs::remove_file(&path).ok();

        assert!(result.is_err());
    }
}
