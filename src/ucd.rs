//! `General_Category` and `Script` lookups per Unicode scalar.
//!
//! Covers the `unicode-properties` and `unicode-script` crates' lookup
//! surface with the same permissive `u32` conventions as the rest of the
//! crate: surrogates report [`GeneralCategory::Surrogate`], out-of-range
//! values report `Unassigned` / [`Script::Unknown`]. Data comes from the
//! generated direct-BMP + astral-trie tables in `ucd_data.rs`
//! (`scripts/gen_props.py`), at the same [`crate::UNICODE_VERSION`] as the
//! width and grapheme tables.

#[path = "ucd_data.rs"]
mod data;

pub use data::Script;

/// `General_Category` property value (UAX #44), e.g. `\p{Lu}`.
///
/// Discriminants follow the UAX #44 table order and are grouped by
/// [`GeneralCategoryGroup`]; the generator (`scripts/gen_props.py`) writes
/// them into the low 5 bits of each ucd word under the same contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GeneralCategory {
	/// `Lu`.
	UppercaseLetter      = 0,
	/// `Ll`.
	LowercaseLetter      = 1,
	/// `Lt`.
	TitlecaseLetter      = 2,
	/// `Lm`.
	ModifierLetter       = 3,
	/// `Lo`.
	OtherLetter          = 4,
	/// `Mn`.
	NonspacingMark       = 5,
	/// `Mc`.
	SpacingMark          = 6,
	/// `Me`.
	EnclosingMark        = 7,
	/// `Nd`.
	DecimalNumber        = 8,
	/// `Nl`.
	LetterNumber         = 9,
	/// `No`.
	OtherNumber          = 10,
	/// `Pc`.
	ConnectorPunctuation = 11,
	/// `Pd`.
	DashPunctuation      = 12,
	/// `Ps`.
	OpenPunctuation      = 13,
	/// `Pe`.
	ClosePunctuation     = 14,
	/// `Pi`.
	InitialPunctuation   = 15,
	/// `Pf`.
	FinalPunctuation     = 16,
	/// `Po`.
	OtherPunctuation     = 17,
	/// `Sm`.
	MathSymbol           = 18,
	/// `Sc`.
	CurrencySymbol       = 19,
	/// `Sk`.
	ModifierSymbol       = 20,
	/// `So`.
	OtherSymbol          = 21,
	/// `Zs`.
	SpaceSeparator       = 22,
	/// `Zl`.
	LineSeparator        = 23,
	/// `Zp`.
	ParagraphSeparator   = 24,
	/// `Cc`.
	Control              = 25,
	/// `Cf`.
	Format               = 26,
	/// `Cs`.
	Surrogate            = 27,
	/// `Co`.
	PrivateUse           = 28,
	/// `Cn`.
	Unassigned           = 29,
}

/// The seven `General_Category` groups: `\p{L}`, `\p{M}`, `\p{N}`, `\p{P}`,
/// `\p{S}`, `\p{Z}`, `\p{C}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneralCategoryGroup {
	/// `L`: `Lu | Ll | Lt | Lm | Lo`.
	Letter,
	/// `M`: `Mn | Mc | Me`.
	Mark,
	/// `N`: `Nd | Nl | No`.
	Number,
	/// `P`: `Pc | Pd | Ps | Pe | Pi | Pf | Po`.
	Punctuation,
	/// `S`: `Sm | Sc | Sk | So`.
	Symbol,
	/// `Z`: `Zs | Zl | Zp`.
	Separator,
	/// `C`: `Cc | Cf | Cs | Co | Cn`.
	Other,
}

impl GeneralCategory {
	/// Group (major class) of this category — the first letter of its
	/// UAX #44 abbreviation.
	#[inline]
	pub const fn group(self) -> GeneralCategoryGroup {
		use GeneralCategoryGroup as G;
		match self as u8 {
			0..=4 => G::Letter,
			5..=7 => G::Mark,
			8..=10 => G::Number,
			11..=17 => G::Punctuation,
			18..=21 => G::Symbol,
			22..=24 => G::Separator,
			_ => G::Other,
		}
	}
}

/// Category of each 5-bit ucd word index. 32 entries (two `Cn` pads) so the
/// masked index never bounds-checks.
static GC_BY_INDEX: [GeneralCategory; 32] = {
	use GeneralCategory as GC;
	[
		GC::UppercaseLetter,
		GC::LowercaseLetter,
		GC::TitlecaseLetter,
		GC::ModifierLetter,
		GC::OtherLetter,
		GC::NonspacingMark,
		GC::SpacingMark,
		GC::EnclosingMark,
		GC::DecimalNumber,
		GC::LetterNumber,
		GC::OtherNumber,
		GC::ConnectorPunctuation,
		GC::DashPunctuation,
		GC::OpenPunctuation,
		GC::ClosePunctuation,
		GC::InitialPunctuation,
		GC::FinalPunctuation,
		GC::OtherPunctuation,
		GC::MathSymbol,
		GC::CurrencySymbol,
		GC::ModifierSymbol,
		GC::OtherSymbol,
		GC::SpaceSeparator,
		GC::LineSeparator,
		GC::ParagraphSeparator,
		GC::Control,
		GC::Format,
		GC::Surrogate,
		GC::PrivateUse,
		GC::Unassigned,
		GC::Unassigned,
		GC::Unassigned,
	]
};

/// `General_Category` of a codepoint.
///
/// Permissive: surrogate values report [`GeneralCategory::Surrogate`],
/// values above `0x10FFFF` report `Unassigned`.
#[inline]
pub fn general_category(cp: u32) -> GeneralCategory {
	GC_BY_INDEX[(data::word(cp) & 31) as usize]
}

/// [`general_category`] reduced to its group: the `\p{L}`/`\p{N}`/`\p{M}`
/// classes of the reference tokenizer regexes in one lookup.
#[inline]
pub fn general_category_group(cp: u32) -> GeneralCategoryGroup {
	general_category(cp).group()
}

/// Script property of a codepoint (UAX #24), e.g. `\p{Han}`.
///
/// Permissive: unassigned, surrogate and out-of-range values report
/// [`Script::Unknown`]; inherited marks report [`Script::Inherited`].
#[inline]
pub fn script(cp: u32) -> Script {
	// The generator asserts the script index fits 8 bits, so the `as u8`
	// cast is lossless and indexes the 256-entry table without a check.
	data::SCRIPT_BY_INDEX[(data::word(cp) >> 5) as u8 as usize]
}

mod sealed {
	pub trait Sealed {}
	impl Sealed for u32 {}
	impl Sealed for char {}
}

/// [`general_category`], [`general_category_group`] and [`script`] as
/// methods on the scalar itself.
///
/// Sealed; implemented for `char` and for raw `u32` codepoints, where the
/// free functions' permissive conventions apply (surrogates report
/// [`GeneralCategory::Surrogate`], out-of-range values `Unassigned` /
/// [`Script::Unknown`]). Replaces the `unicode-properties`
/// `UnicodeGeneralCategory` and `unicode-script` `UnicodeScript` extension
/// traits in one import.
pub trait Ucd: sealed::Sealed + Copy {
	/// `General_Category` of this scalar; see [`general_category`].
	fn general_category(self) -> GeneralCategory;

	/// Group of the category; see [`general_category_group`].
	#[inline]
	fn general_category_group(self) -> GeneralCategoryGroup {
		self.general_category().group()
	}

	/// Script property of this scalar; see [`script`].
	fn script(self) -> Script;
}

impl Ucd for u32 {
	#[inline]
	fn general_category(self) -> GeneralCategory {
		general_category(self)
	}

	#[inline]
	fn script(self) -> Script {
		script(self)
	}
}

impl Ucd for char {
	#[inline]
	fn general_category(self) -> GeneralCategory {
		general_category(self as u32)
	}

	#[inline]
	fn script(self) -> Script {
		script(self as u32)
	}
}
