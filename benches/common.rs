//! Shared bench harness helpers for Divan & ratio-comparison benchmarks.
#![allow(dead_code, reason = "Each benchmark imports only the helpers it needs.")]

use std::{hint::black_box, sync::LazyLock, time::Instant};

/// Configures Rayon's global thread pool to use half of available CPU cores (at
/// least 1 thread).
pub fn init_rayon() {
	static INIT: LazyLock<()> = LazyLock::new(|| {
		let threads = std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1));
		let _ = rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build_global();
	});
	let () = *INIT;
}

/// Fast-mode configuration: warm-up time and measurement run count.
/// When `--fast` is passed, warm-up uses 0.01s and only 2 runs instead of 8.
static FAST: LazyLock<Fast> = LazyLock::new(|| {
	let fast = std::env::args().any(|a| a == "--fast");
	Fast { warmup_secs: if fast { 0.01 } else { 0.03 }, runs: if fast { 2 } else { 8 } }
});
/// Loss-only mode: `--loss-only` arg skips rows where xutf (index 0) wins or
/// ties.
static LOSS_ONLY: LazyLock<bool> = LazyLock::new(|| std::env::args().any(|a| a == "--loss-only"));

struct Fast {
	warmup_secs: f64,
	runs:        u32,
}

/// Repeats `seed` until at least `target_bytes` of UTF-8.
pub fn build_input(seed: &str, target_bytes: usize) -> String {
	let mut s = String::with_capacity(target_bytes + seed.len());
	while s.len() < target_bytes {
		s.push_str(seed);
	}
	s
}

/// Measures throughput in GB/s of `bytes` processed per call.
pub fn measure(bytes: usize, mut f: impl FnMut() -> usize) -> f64 {
	for _ in 0..3 {
		black_box(f());
	}
	let mut iters = 1u32;
	loop {
		let t = Instant::now();
		for _ in 0..iters {
			black_box(f());
		}
		if t.elapsed().as_secs_f64() >= FAST.warmup_secs || iters >= 1 << 24 {
			break;
		}
		iters *= 4;
	}
	let mut best = f64::INFINITY;
	for _ in 0..FAST.runs {
		let t = Instant::now();
		for _ in 0..iters {
			black_box(f());
		}
		best = best.min(t.elapsed().as_secs_f64() / iters as f64);
	}
	bytes as f64 / best / 1e9
}

/// Boilerplate main entry point for benchmark binaries.
/// Runs Divan benchmarks if DIVAN env var is set or `--divan` arg is passed;
/// otherwise runs the custom ratio table benchmark.
#[macro_export]
macro_rules! bench_main {
	() => {
		fn main() {
			if std::env::var("DIVAN").is_ok() || std::env::args().any(|a| a == "--divan") {
				divan::main();
			} else {
				common::init_rayon();
				run_ratio_table();
			}
		}
	};
}
pub use bench_main;

/// Appends one JSON line per printed table to the file named by `BENCH_JSON`,
/// tagged with `BENCH_HOST` (default: `{arch} {os}`), so raw runs from
/// several machines can be committed under `benches/data/` and graphed with
/// `scripts/bench_viz.py`. Repeated runs append; the grapher keeps the last
/// record per `(host, bench)`.
fn emit_json(title: &str, impls: &[&str], rows: &[(&str, Vec<f64>)]) {
	use std::{fmt::Write as _, io::Write as _};

	let Ok(path) = std::env::var("BENCH_JSON") else {
		return;
	};
	let host = std::env::var("BENCH_HOST")
		.unwrap_or_else(|_| format!("{} {}", std::env::consts::ARCH, std::env::consts::OS));
	let unix_time = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_secs());

	fn quoted(out: &mut String, s: &str) {
		out.push('"');
		for c in s.chars() {
			match c {
				'"' | '\\' => {
					out.push('\\');
					out.push(c);
				},
				c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).expect("writing to String cannot fail"),
				c => out.push(c),
			}
		}
		out.push('"');
	}

	let mut line = String::new();
	line.push_str("{\"bench\":");
	quoted(&mut line, title);
	line.push_str(",\"unit\":\"GB/s\",\"host\":");
	quoted(&mut line, &host);
	write!(line, ",\"unix_time\":{unix_time},\"impls\":[").expect("writing to String cannot fail");
	for (i, name) in impls.iter().enumerate() {
		if i > 0 {
			line.push(',');
		}
		quoted(&mut line, name);
	}
	line.push_str("],\"rows\":[");
	for (i, (input, vals)) in rows.iter().enumerate() {
		if i > 0 {
			line.push(',');
		}
		line.push_str("{\"input\":");
		quoted(&mut line, input);
		line.push_str(",\"values\":[");
		for (j, v) in vals.iter().enumerate() {
			if j > 0 {
				line.push(',');
			}
			let v = if v.is_finite() { *v } else { 0.0 };
			write!(line, "{v:.6}").expect("writing to String cannot fail");
		}
		line.push_str("]}");
	}
	line.push_str("]}");

	let mut file = std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(&path)
		.expect("BENCH_JSON must name a writable file");
	writeln!(file, "{line}").expect("BENCH_JSON write failed");
}

/// Prints a comparison table with performance deltas relative to xutf (index
/// 0). Formats deltas with green ▼ +XX.X% (when xutf is faster) and red ▲
/// -XX.X% (when competitor is faster). Bright colors are used when diff > 5%,
/// and standard colors when diff is within ±5%.
/// Separators: `>` when xutf is winning (> 5%), `<` when competitor is winning
/// (> 5%), `~` when diff is within ±5%.
pub fn print_ratio_table(title: &str, impls: &[&str], rows: &[(&str, Vec<f64>)]) {
	init_rayon();
	let detail = std::env::args().any(|a| a == "--detail");
	print_ratio_table_inner(title, impls, rows, detail);
}

fn print_ratio_table_inner(title: &str, impls: &[&str], rows: &[(&str, Vec<f64>)], detail: bool) {
	emit_json(title, impls, rows);
	const COL_WIDTH: usize = 23;
	println!("\n## {title} (GB/s)");
	if detail {
		print!("{:<14}", "input");
		for i in impls {
			print!("{i:>COL_WIDTH$}");
		}
		println!();
	}

	for (name, vals) in rows {
		let base = vals[0];
		// In summary mode (detail=false), index 1 (simdutf) is the target shown.
		// In detail mode, compare against the best non-xutf implementation.
		if *LOSS_ONLY {
			let target_val = if detail {
				vals[1..]
					.iter()
					.copied()
					.filter(|&v| v > 0.0 && v.is_finite())
					.fold(0.0f64, f64::max)
			} else {
				vals.get(1).copied().unwrap_or(0.0)
			};
			if target_val <= base {
				continue;
			}
		}
		if detail {
			let is_winning = vals[1..]
				.iter()
				.copied()
				.filter(|&v| v > 0.0 && v.is_finite())
				.all(|v| base > v);
			let mut line = format!("{name:<14}");
			for (idx, &val) in vals.iter().enumerate() {
				if idx == 0 {
					let cell = format!("{val:.1} GB/s (1.00x)");
					use std::fmt::Write;
					let _ = write!(line, "{cell:>COL_WIDTH$}");
				} else if val.is_nan() || val == 0.0 {
					use std::fmt::Write;
					let _ = write!(line, "{:>COL_WIDTH$}", "N/A");
				} else {
					let (delta_str, _, ansi_len) = format_delta(base, val);
					let cell = format!("{val:.1} GB/s ({delta_str})");
					let target_width = COL_WIDTH + ansi_len;
					use std::fmt::Write;
					let _ = write!(line, "{cell:>target_width$}");
				}
			}
			if is_winning {
				println!("\x1b[2m{line}\x1b[0m");
			} else {
				println!("{line}");
			}
		} else {
			// Always show simdutf (index 1) as the comparison target
			let best_idx = 1;
			let best_val = vals.get(1).copied().unwrap_or(0.0);
			let (delta_str, sep, _) = format_delta(base, best_val);
			let winner_name = impls.get(best_idx).unwrap_or(&"N/A");
			let is_winning = base > best_val;
			let line = format!(
				"{name:<15} {base:>5.1} GB/s {sep} [{winner_name}] {best_val:>5.1} GB/s ({delta_str})"
			);
			if is_winning {
				println!("\x1b[2m{line}\x1b[0m");
			} else {
				println!("{line}");
			}
		}
	}
}

fn format_delta(base: f64, val: f64) -> (String, char, usize) {
	if val == base {
		(String::from("1.00x"), '~', 0)
	} else if val > base {
		let ratio = val / base;
		let pct = -(ratio - 1.0) * 100.0;
		let is_close = pct.abs() <= 5.0;
		let sep = if is_close { '~' } else { '<' };
		let color = if is_close { "31" } else { "91" };
		(format!("\x1b[{color}m▲ {pct:.1}%\x1b[0m"), sep, 9)
	} else {
		let slowdown = base / val;
		let pct = (slowdown - 1.0) * 100.0;
		let is_close = pct.abs() <= 5.0;
		let sep = if is_close { '~' } else { '>' };
		let color = if is_close { "32" } else { "92" };
		(format!("\x1b[{color}m▼ +{pct:.1}%\x1b[0m"), sep, 9)
	}
}

#[cfg(test)]
mod tests {

	#[test]
	fn test_format_delta_exact_tie() {
		let (delta, sep, len) = format_delta(100.0, 100.0);
		assert_eq!(delta, "1.00x");
		assert_eq!(sep, '~');
		assert_eq!(len, 0);
	}

	#[test]
	fn test_format_delta_sub_1pct_competitor_win() {
		let (delta, sep, len) = format_delta(100.0, 100.5);
		assert_eq!(delta, "\x1b[31m▲ -0.5%\x1b[0m");
		assert_eq!(sep, '~');
		assert_eq!(len, 9);
	}

	#[test]
	fn test_format_delta_sub_1pct_xutf_win() {
		let (delta, sep, len) = format_delta(100.5, 100.0);
		assert_eq!(delta, "\x1b[32m▼ +0.5%\x1b[0m");
		assert_eq!(sep, '~');
		assert_eq!(len, 9);
	}

	#[test]
	fn test_format_delta_5pct_boundary_competitor_win() {
		let (delta50, sep50, _) = format_delta(100.0, 105.0);
		assert_eq!(delta50, "\x1b[31m▲ -5.0%\x1b[0m");
		assert_eq!(sep50, '~');

		let (delta51, sep51, _) = format_delta(100.0, 105.1);
		assert_eq!(delta51, "\x1b[91m▲ -5.1%\x1b[0m");
		assert_eq!(sep51, '<');
	}

	#[test]
	fn test_format_delta_5pct_boundary_xutf_win() {
		let (delta50, sep50, _) = format_delta(105.0, 100.0);
		assert_eq!(delta50, "\x1b[32m▼ +5.0%\x1b[0m");
		assert_eq!(sep50, '~');

		let (delta51, sep51, _) = format_delta(105.1, 100.0);
		assert_eq!(delta51, "\x1b[92m▼ +5.1%\x1b[0m");
		assert_eq!(sep51, '>');
	}
}
