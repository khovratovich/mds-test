// MDS test for a random N×N matrix over the prime field p = 2^31 - 2^24 + 1
// p = 2^64-2^32+1
//
// Definition used (common in block-cipher diffusion):
// A is MDS iff every k×k square submatrix (minor) is nonsingular for all k = 1..N.
//
// WARNING: This is exponential-time in N in the worst case (checks all minors).
//
// Build/run (example):
//   cargo new mds_test && cd mds_test
//   (replace src/main.rs with this file)
//   add dependency: rand = "0.8" in Cargo.toml
//   cargo run --release -- 4
//
// You can also pass an optional seed:
//   cargo run --release -- 8 12345

use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use std::env;

const P: u64 = 2130706433; // KoalaBear: 2^31 - 2^24 + 1

#[inline]
fn mod_add(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P { s - P } else { s }
}

#[inline]
fn mod_sub(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + P - b }
}

#[inline]
fn mod_mul(a: u64, b: u64) -> u64 {
    // NOTE: Only correct for primes P <= 2^32 - 1 (32-bit or smaller),
    // so that a*b fits in u64 without overflow. For larger primes use u128.
    (a * b) % P
}

fn mod_pow(mut a: u64, mut e: u64) -> u64 {
    let mut r: u64 = 1;
    while e > 0 {
        if (e & 1) == 1 {
            r = mod_mul(r, a);
        }
        a = mod_mul(a, a);
        e >>= 1;
    }
    r
}

#[inline]
fn mod_inv(a: u64) -> u64 {
    // Fermat inverse since P is prime: a^(P-2) mod P
    // Caller must ensure a != 0.
    mod_pow(a, P - 2)
}

/// Returns true iff the k×k submatrix A[rows][cols] is nonsingular.
fn nonsingular_minor(matrix: &[Vec<u64>], rows: &[usize], cols: &[usize]) -> bool {
    let k = rows.len();
    debug_assert_eq!(k, cols.len());

    // Copy into a temporary k×k for elimination (simpler + usually faster than lots of indexing)
    let mut a = vec![vec![0u64; k]; k];
    for i in 0..k {
        for j in 0..k {
            a[i][j] = matrix[rows[i]][cols[j]];
        }
    }

    // Gaussian elimination to check full rank.
    // We don't need the actual determinant value, only whether it's zero.
    for col in 0..k {
        // Find pivot row at/under col with nonzero in this column.
        let mut pivot = None;
        for r in col..k {
            if a[r][col] != 0 {
                pivot = Some(r);
                break;
            }
        }
        let Some(piv) = pivot else {
            return false; // singular
        };

        // Swap into place if needed.
        if piv != col {
            a.swap(piv, col);
        }

        // Eliminate rows below.
        let inv_pivot = mod_inv(a[col][col]);
        for r in (col + 1)..k {
            if a[r][col] == 0 {
                continue;
            }
            let factor = mod_mul(a[r][col], inv_pivot);
            // row_r = row_r - factor * row_col
            a[r][col] = 0;
            for c in (col + 1)..k {
                let t = mod_mul(factor, a[col][c]);
                a[r][c] = mod_sub(a[r][c], t);
            }
        }
    }

    true
}

/// Calls `f` on every k-combination of indices 0..n-1. Early exit if f returns false.
fn for_each_combination<F>(n: usize, k: usize, mut f: F) -> bool
where
    F: FnMut(&[usize]) -> bool,
{
    let mut comb = Vec::with_capacity(k);

    fn rec<F>(start: usize, n: usize, k: usize, comb: &mut Vec<usize>, f: &mut F) -> bool
    where
        F: FnMut(&[usize]) -> bool,
    {
        if comb.len() == k {
            return f(comb);
        }
        // Need enough room to fill remaining positions
        let remaining = k - comb.len();
        for i in start..=(n - remaining) {
            comb.push(i);
            if !rec(i + 1, n, k, comb, f) {
                return false;
            }
            comb.pop();
        }
        true
    }

    rec(0, n, k, &mut comb, &mut f)
}

/// Full MDS test: checks all square minors of all sizes 1..N.
fn is_mds(matrix: &[Vec<u64>]) -> bool {
    use std::io::Write;

    fn binom(n: usize, k: usize) -> u64 {
        if k > n { return 0; }
        let k = k.min(n - k);
        let mut r: u64 = 1;
        for i in 0..k {
            r = r * (n - i) as u64 / (i + 1) as u64;
        }
        r
    }

    let n = matrix.len();
    // Total minors = sum_{k=1}^{n} C(n,k)^2  (= C(2n,n) - 1 by Vandermonde)
    let total: u64 = (1..=n).map(|k| binom(n, k) * binom(n, k)).sum();

    let mut count: u64 = 0;
    let mut last_pct: u64 = u64::MAX;

    for k in 1..=n {
        let ok_rows = for_each_combination(n, k, |rows| {
            for_each_combination(n, k, |cols| {
                count += 1;
                let pct = count * 100 / total;
                if pct != last_pct {
                    last_pct = pct;
                    eprint!("\r{:3}%  ({}/{} minors tested)", pct, count, total);
                    let _ = std::io::stderr().flush();
                }
                nonsingular_minor(matrix, rows, cols)
            })
        });
        if !ok_rows {
            eprintln!();
            return false;
        }
    }
    eprintln!("\r100%  ({}/{} minors tested)", total, total);
    true
}

fn random_matrix(n: usize, rng: &mut impl Rng) -> Vec<Vec<u64>> {
    let mut m = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = rng.gen_range(0..64);
        }
    }
    m
}

fn random_circulant_matrix(n: usize, rng: &mut impl Rng) -> Vec<Vec<u64>> {
    let mut first_row = vec![0u64; n];
    for j in 0..n {
        first_row[j] = rng.gen_range(1..8); // Avoid zero to increase chance of MDS (not guaranteed)
    }
    let mut m = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = first_row[(j + i) % n];
        }
    }
    m
}

#[test]
fn test_matrix() -> Result<(), ()> {
    let mut first_row = vec![1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];
    let n = first_row.len();
    let mut m = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = first_row[(j + i) % n];
        }
    }

    let ok = is_mds(&m);
        if ok {
            println!("Test success!");
            // Print a small matrix (optional; comment out if you want speed only)
            if n <= 16 {
                println!("Matrix (mod P={}):", P);
                for row in &m {
                    println!("{:?}", row);
                }
                println!();
            }
            Ok(())
        }
        else {
            Err(())
        }
}

#[test]
fn test_kb_mds_first_row_16() {
    const FIRST_ROW: [u64; 16] = [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];
    let n = FIRST_ROW.len();
    let mut m = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = FIRST_ROW[(j + n - i) % n];
        }
    }
    assert!(is_mds(&m), "circulant matrix with first row {:?} is not MDS", FIRST_ROW);
}

fn random_small_weight_circulant_matrix(n: usize, rng: &mut impl Rng) -> Vec<Vec<u64>> {
    //generate a vector of powers of two up to 2^(w-1) where w is the desired weight (number of nonzero entries in the first row)
    let w = 5; // Desired weight (number of nonzero entries in the first row)
    let mut powers_of_two = Vec::with_capacity(w);
    for i in 0..w {
        powers_of_two.push(1 << i);
    }
    //for each position in the first row, choose a random sum of k distinct powers of two, where k is a random number between 1 and w (inclusive)
    let k = 2; // Max number of nonzero entries in the first row
    let mut first_row = vec![0u64; n];
    for j in 0..n {
        for _ in 0..rng.gen_range(1..=k) {
            let pow = powers_of_two[rng.gen_range(0..w)];
            first_row[j] = mod_add(first_row[j], pow);
        }
    }
    let mut m = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = first_row[(j + i) % n];
        }
    }
    m
}




fn main() {


    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} N [seed]", args[0]);
        eprintln!("Example: {} 4", args[0]);
        std::process::exit(2);
    }

    let n: usize = args[1].parse().expect("N must be an integer");
    if n == 0 {
        eprintln!("N must be >= 1");
        std::process::exit(2);
    }

    let mut rng: StdRng = if args.len() >= 3 {
        let seed: u64 = args[2].parse().expect("seed must be an integer");
        StdRng::seed_from_u64(seed)
    } else {
        // Use an OS-seeded RNG to create a deterministic StdRng seed.
        let seed: u64 = rand::random();
        StdRng::seed_from_u64(seed)
    };

    for i in 0..10000 {
        let m = random_small_weight_circulant_matrix(n, &mut rng);
        let ok = is_mds(&m);
        if ok {
            println!("Found MDS matrix on iteration {}!", i);
            // Print a small matrix (optional; comment out if you want speed only)
            if n <= 16 {
                println!("Matrix (mod P={}):", P);
                for row in &m {
                    println!("{:?}", row);
                }
                println!();
            }
            break;
        }
    }



    //println!("N={}  =>  MDS: {}", n, ok);
}