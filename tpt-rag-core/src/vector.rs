//! Vector arithmetic shared by the index and the embedding code. Pure Rust, so it builds for
//! every target, including WebAssembly.

/// Scales `vector` to unit length, so cosine similarity is a plain dot product.
pub(crate) fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two vectors. Returns 0 when the lengths differ or a vector is empty.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot(a, b) / (norm_a * norm_b)
    }
}

/// Dot product of the common prefix of `a` and `b`. The sum is kept in eight independent
/// lanes. A single running sum is a chain of dependent additions that the compiler cannot
/// vectorize, and this is the hot loop of the index.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let main = n - n % 8;
    let (a_main, a_tail) = a[..n].split_at(main);
    let (b_main, b_tail) = b[..n].split_at(main);

    let mut lanes = [0.0f32; 8];
    for (ca, cb) in a_main.chunks_exact(8).zip(b_main.chunks_exact(8)) {
        for (lane, (x, y)) in lanes.iter_mut().zip(ca.iter().zip(cb)) {
            *lane += x * y;
        }
    }
    let tail: f32 = a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
}
