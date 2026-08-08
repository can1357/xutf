//! Permissive UTF-8 / UTF-16 / UTF-32 transcoding, comparison and BOM
//! detection, with SIMD ASCII fast paths.
//!
//! A Rust port of the `xstd::codepoint_cvt` / `utf_*` family. Design points
//! carried over:
//!
//! - **Permissive, non-validating codecs.** Decoding never fails: truncated
//!   UTF-8 sequences decode to `0` (consuming the remainder), lone surrogates
//!   pass through, garbage-in produces garbage-out. Values are raw `u32`
//!   codepoints, not `char`.
//! - **ASCII-only case folding** for the `*_ignore_ascii_case` operations and
//!   [`AsciiCase`] transcoding.
//! - **Foreign endianness** as a type parameter: [`Utf16<true>`](Utf16) is
//!   byte-swapped relative to the native byte order (see [`Utf16Be`] /
//!   [`Utf16Le`] aliases).
//! - **Terminal text primitives**, allocation-free and generic over the
//!   encoding: grapheme segmentation ([`graphemes`], UAX #29 incl. `GB9c` and
//!   GB11), visible cell width ([`width`]), width truncation ([`truncate`],
//!   returns a borrowed prefix) and greedy word wrap ([`wrap`], yields borrowed
//!   sublines), backed by a generated Unicode [`UNICODE_VERSION`] property trie
//!   (`scripts/gen_props.py`).
//!
//! Special-op mapping from the C++ original:
//!
//! | C++ (`xstd`)                          | Rust                                        |
//! |---------------------------------------|---------------------------------------------|
//! | clang vector extensions (`xvec`)      | `core::simd` (LLVM vector IR)               |
//! | `bit_pdep` / `bit_pext` (BMI2)        | `_pdep_u32` / `_pext_u32` under `cfg(bmi2)`  |
//! | `assume(..)`                          | `core::hint::assert_unchecked`              |
//! | `bswapw` / `bswapd`                   | `u16::swap_bytes` / `u32::swap_bytes`       |
//! | `lsb(mask())`                         | `Mask::to_bitmask().trailing_zeros()`       |
//! | overlapped tail vector load/store     | `Simd::from_slice` / `copy_to_slice` on an  |
//! |                                       | overlapping window at `limit - N`           |
//!
//! Deliberate fixes over the original:
//! - UTF-16 encode computes the low surrogate as `0xDC00 | (cp & 0x3FF)`; the
//!   C++ `0xDC00 | uint16_t(cp)` corrupts pairs when bit 13 of `cp - 0x10000`
//!   is set (e.g. U+12000).
//! - `compare` orders by decoded codepoint everywhere (the C++ mixed UTF-16
//!   code-unit order with codepoint order) and a longer string with a trailing
//!   `NUL` codepoint no longer compares equal to its prefix.

#![feature(portable_simd)]
#![feature(stmt_expr_attributes)]

extern crate alloc;

mod bytes;
mod compare;
mod convert;
mod encoding;
mod grapheme;
mod kernel;
mod native;
mod props;
#[cfg(test)]
mod props_tests;
mod simd;
mod tables;
mod truncate;
mod unit;
mod utf16;
mod utf32;
mod utf8;
mod width;
mod wrap;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use bytes::{Bom, detect_bom, from_bytes};
pub use compare::{compare, compare_ignore_ascii_case, equals, equals_ignore_ascii_case};
pub use convert::{
	AsciiCase, to_string, transcode, transcode_into, transcode_with_case, transcoded_len,
};
pub use encoding::{Codepoints, Encoding, Kind, chars, codepoints};
pub use grapheme::{Grapheme, Graphemes, graphemes, graphemes_str};
pub use props::UNICODE_VERSION;
pub use truncate::{truncate, truncate_str};
pub use unit::Unit;
pub use utf8::Utf8;
pub use utf16::Utf16;
pub use utf32::Utf32;
pub use width::{width, width_str};
pub use wrap::{Wrapped, wrap, wrap_str};

/// Little-endian UTF-16, regardless of the native byte order.
#[cfg(target_endian = "little")]
pub type Utf16Le = Utf16<false>;
/// Big-endian UTF-16, regardless of the native byte order.
#[cfg(target_endian = "little")]
pub type Utf16Be = Utf16<true>;
/// Little-endian UTF-32, regardless of the native byte order.
#[cfg(target_endian = "little")]
pub type Utf32Le = Utf32<false>;
/// Big-endian UTF-32, regardless of the native byte order.
#[cfg(target_endian = "little")]
pub type Utf32Be = Utf32<true>;

#[cfg(target_endian = "big")]
pub type Utf16Le = Utf16<true>;
#[cfg(target_endian = "big")]
pub type Utf16Be = Utf16<false>;
#[cfg(target_endian = "big")]
pub type Utf32Le = Utf32<true>;
#[cfg(target_endian = "big")]
pub type Utf32Be = Utf32<false>;
