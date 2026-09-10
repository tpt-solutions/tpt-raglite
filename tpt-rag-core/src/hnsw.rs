use std::collections::BinaryHeap;
use std::io::{Read, Write};
use std::path::Path;

use ordered_float::NotNan;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::embedding::cosine_similarity;
use crate::error::{RagError, Result};

pub struct HnswIndex {
    dim: usize,
    m: usize,
    ef_construction: usize,
    layers: Vec<Vec<HnswNode>>,
    entry_point: Option<usize>,
    max_layer: usize,
    vectors: Vec<Vec<f32>>,
    rng: StdRng,
}

#[derive(Clone)]
struct HnswNode {
    id: usize,
    neighbors: Vec<usize>,
}

#[derive(Clone)]
struct SearchResult {
    id: usize,
    distance: NotNan<f32>,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl std::cmp::Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance).reverse()
    }
}

impl std::cmp::PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            dim,
            m,
            ef_construction,
            layers: Vec::new(),
            entry_point: None,
            max_layer: 0,
            vectors: Vec::new(),
            rng: StdRng::from_entropy(),
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");
        let id = self.vectors.len();
        self.vectors.push(vector.clone());

        let level = self.random_level();
        while self.layers.len() <= level {
            self.layers.push(Vec::new());
        }

        let node = HnswNode {
            id,
            neighbors: Vec::new(),
        };
        self.layers[level].push(node);

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = level;
            return id;
        }

        let entry = self.entry_point.unwrap();
        let mut current_closest = entry;
        let mut current_layer = self.max_layer;

        while current_layer > level {
            let neighbors = self.get_neighbors(current_layer, current_closest);
            let best = neighbors
                .iter()
                .map(|&n| {
                    let dist = cosine_similarity(&vector, &self.vectors[n]);
                    (n, NotNan::new(-dist).unwrap_or(NotNan::new(0.0).unwrap()))
                })
                .min_by_key(|(_, d)| *d)
                .map(|(n, _)| n)
                .unwrap_or(current_closest);
            current_closest = best;
            current_layer -= 1;
        }

        for layer in (level..=current_layer).rev() {
            let candidates = self.search_layer(&vector, current_closest, self.ef_construction, layer);
            let selected = self.select_neighbors(&candidates, self.m);
            self.connect_neighbors(layer, id, &selected);
            current_closest = selected.first().copied().unwrap_or(current_closest);
        }

        for layer in (0..level).rev() {
            let candidates = self.search_layer(&vector, current_closest, self.ef_construction, layer);
            let selected = self.select_neighbors(&candidates, self.m * 2);
            self.connect_neighbors(layer, id, &selected);
            current_closest = selected.first().copied().unwrap_or(current_closest);
        }

        if level > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = level;
        }

        id
    }

    pub fn search(&self, query: &[f32], top_k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dim);

        let entry = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let mut current_closest = entry;

        for layer in (1..=self.max_layer).rev() {
            let neighbors = self.get_neighbors(layer, current_closest);
            current_closest = neighbors
                .iter()
                .map(|&n| {
                    let dist = cosine_similarity(query, &self.vectors[n]);
                    (n, dist)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(n, _)| n)
                .unwrap_or(current_closest);
        }

        let candidates = self.search_layer(query, current_closest, ef_search, 0);
        let mut results: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(id, dist)| (id, -dist.into_inner()))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, NotNan<f32>)> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates: BinaryHeap<SearchResult> = BinaryHeap::new();
        let mut results: BinaryHeap<SearchResult> = BinaryHeap::new();

        let entry_dist = NotNan::new(-cosine_similarity(query, &self.vectors[entry_point]))
            .unwrap_or(NotNan::new(0.0).unwrap());
        candidates.push(SearchResult {
            id: entry_point,
            distance: entry_dist,
        });
        results.push(SearchResult {
            id: entry_point,
            distance: entry_dist,
        });
        visited.insert(entry_point);

        while let Some(current) = candidates.pop() {
            let worst_in_results = results.peek().map(|r| r.distance).unwrap_or(NotNan::new(f32::MAX).unwrap());
            if current.distance > worst_in_results && results.len() >= ef {
                break;
            }

            let neighbors = self.get_neighbors(layer, current.id);
            for neighbor_id in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let dist = NotNan::new(-cosine_similarity(query, &self.vectors[neighbor_id]))
                    .unwrap_or(NotNan::new(0.0).unwrap());
                let worst = results.peek().map(|r| r.distance).unwrap_or(NotNan::new(f32::MAX).unwrap());

                if dist < worst || results.len() < ef {
                    candidates.push(SearchResult {
                        id: neighbor_id,
                        distance: dist,
                    });
                    results.push(SearchResult {
                        id: neighbor_id,
                        distance: dist,
                    });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect()
    }

    fn select_neighbors(&self, candidates: &[(usize, NotNan<f32>)], max_count: usize) -> Vec<usize> {
        candidates
            .iter()
            .take(max_count)
            .map(|(id, _)| *id)
            .collect()
    }

    fn connect_neighbors(&mut self, layer: usize, node_id: usize, neighbors: &[usize]) {
        if let Some(node) = self.layers[layer].iter_mut().find(|n| n.id == node_id) {
            node.neighbors.extend_from_slice(neighbors);
        }

        for &neighbor_id in neighbors {
            if let Some(node) = self.layers[layer].iter_mut().find(|n| n.id == neighbor_id) {
                node.neighbors.push(node_id);
            }
        }
    }

    fn get_neighbors(&self, layer: usize, node_id: usize) -> Vec<usize> {
        self.layers
            .get(layer)
            .and_then(|l| l.iter().find(|n| n.id == node_id))
            .map(|n| n.neighbors.clone())
            .unwrap_or_default()
    }

    fn random_level(&mut self) -> usize {
        let mut level = 0;
        let m = self.m as f64;
        while self.rng.gen::<f64>() < 1.0 / m && level < 16 {
            level += 1;
        }
        level
    }

    pub fn flush(&self, path: &Path) -> Result<()> {
        let mut f = std::fs::File::create(path)?;

        // Header
        write_u64(&mut f, self.dim as u64)?;
        write_u64(&mut f, self.m as u64)?;
        write_u64(&mut f, self.ef_construction as u64)?;
        write_u64(&mut f, self.vectors.len() as u64)?;
        write_u64(&mut f, self.max_layer as u64)?;
        write_u64(
            &mut f,
            self.entry_point.map(|e| e as u64).unwrap_or(u64::MAX),
        )?;

        // Layers: for each layer, write node count, then each node's id, neighbor count, neighbor ids
        write_u64(&mut f, self.layers.len() as u64)?;
        for layer in &self.layers {
            write_u64(&mut f, layer.len() as u64)?;
            for node in layer {
                write_u64(&mut f, node.id as u64)?;
                write_u64(&mut f, node.neighbors.len() as u64)?;
                for &nid in &node.neighbors {
                    write_u64(&mut f, nid as u64)?;
                }
            }
        }

        // Vectors
        for vec in &self.vectors {
            for &v in vec {
                f.write_all(&v.to_le_bytes())?;
            }
        }

        Ok(())
    }

    pub fn load_from_disk(path: &Path, dim: usize, m: usize, ef_construction: usize) -> Result<Self> {
        let mut f = std::fs::File::open(path)?;

        // Header
        let file_dim = read_u64(&mut f)? as usize;
        let file_m = read_u64(&mut f)? as usize;
        let _file_ef = read_u64(&mut f)? as usize;
        let vector_count = read_u64(&mut f)? as usize;
        let max_layer = read_u64(&mut f)? as usize;
        let ep_raw = read_u64(&mut f)?;
        let entry_point = if ep_raw == u64::MAX {
            None
        } else {
            Some(ep_raw as usize)
        };

        if file_dim != dim {
            return Err(RagError::Hnsw(format!(
                "Dimension mismatch: file has {}, expected {}",
                file_dim, dim
            )));
        }
        if file_m != m {
            return Err(RagError::Hnsw(format!(
                "M mismatch: file has {}, expected {}",
                file_m, m
            )));
        }

        // Layers
        let layer_count = read_u64(&mut f)? as usize;
        let mut layers: Vec<Vec<HnswNode>> = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let node_count = read_u64(&mut f)? as usize;
            let mut layer = Vec::with_capacity(node_count);
            for _ in 0..node_count {
                let id = read_u64(&mut f)? as usize;
                let neighbor_count = read_u64(&mut f)? as usize;
                let mut neighbors = Vec::with_capacity(neighbor_count);
                for _ in 0..neighbor_count {
                    neighbors.push(read_u64(&mut f)? as usize);
                }
                layer.push(HnswNode { id, neighbors });
            }
            layers.push(layer);
        }

        // Vectors
        let mut vectors = Vec::with_capacity(vector_count);
        for _ in 0..vector_count {
            let mut vec = vec![0.0f32; dim];
            for v in &mut vec {
                let mut buf = [0u8; 4];
                f.read_exact(&mut buf)?;
                *v = f32::from_le_bytes(buf);
            }
            vectors.push(vec);
        }

        Ok(Self {
            dim,
            m,
            ef_construction,
            layers,
            entry_point,
            max_layer,
            vectors,
            rng: StdRng::from_entropy(),
        })
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

    fn random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = HnswIndex::new(4, 2, 4);
        let vectors = random_vectors(100, 4);
        for v in &vectors {
            index.insert(v.clone());
        }
        assert_eq!(index.len(), 100);

        let query = &vectors[0];
        let results = index.search(query, 5, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // closest to itself
    }

    #[test]
    fn test_recall_synthetic() {
        let mut index = HnswIndex::new(32, 4, 8);
        let vectors = random_vectors(500, 32);
        for v in &vectors {
            index.insert(v.clone());
        }

        let mut correct = 0;
        let total = 50;
        let mut rng = rand::thread_rng();
        for _ in 0..total {
            let qi = rng.gen_range(0..500);
            let results = index.search(&vectors[qi], 10, 20);

            let mut brute_force: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_similarity(&vectors[qi], v)))
                .collect();
            brute_force.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_10_bf: std::collections::HashSet<usize> = brute_force
                .iter()
                .take(10)
                .map(|(id, _)| *id)
                .collect();

            let top_10_hnsw: std::collections::HashSet<usize> =
                results.iter().map(|(id, _)| *id).collect();

            let intersection = top_10_bf.intersection(&top_10_hnsw).count();
            correct += intersection;
        }

        let recall = correct as f64 / (total * 10) as f64;
        assert!(recall > 0.8, "Recall was {recall}, expected > 0.8");
    }

    #[test]
    fn test_empty_index_search() {
        let index = HnswIndex::new(4, 2, 4);
        let results = index.search(&vec![0.0; 4], 5, 10);
        assert!(results.is_empty());
    }
}
