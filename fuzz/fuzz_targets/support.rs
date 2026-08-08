pub(crate) fn u16_units(bytes: &[u8]) -> Vec<u16> {
	bytes
		.chunks_exact(2)
		.map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
		.collect()
}

pub(crate) fn u32_units(bytes: &[u8]) -> Vec<u32> {
	bytes
		.chunks_exact(4)
		.map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
		.collect()
}
