// Numeral formatting. All data renders in IBM Plex Mono with tabular-nums;
// negative values use the typographic minus (U+2212) for even columns.

const MINUS = '−';

function group(v: number, decimals: number): string {
	return Math.abs(v).toLocaleString('en-US', {
		minimumFractionDigits: decimals,
		maximumFractionDigits: decimals
	});
}

/** $12,480 / −$3,212 — optionally signed on the positive side. */
export function fmtUsd(v: number, decimals = 0, signed = false): string {
	const sign = v < 0 ? MINUS : signed && v > 0 ? '+' : '';
	return `${sign}$${group(v, decimals)}`;
}

/** 54.3% */
export function fmtPct(v: number, decimals = 1): string {
	const sign = v < 0 ? MINUS : '';
	return `${sign}${group(v, decimals)}%`;
}

/** +0.42R / −1.00R */
export function fmtR(v: number, decimals = 2, signed = true): string {
	const sign = v < 0 ? MINUS : signed && v > 0 ? '+' : '';
	return `${sign}${group(v, decimals)}R`;
}

/** plain number, tabular */
export function fmtNum(v: number, decimals = 2): string {
	const sign = v < 0 ? MINUS : '';
	return `${sign}${group(v, decimals)}`;
}

/** "2025-03-14T10:25:00" -> "10:25" */
export function fmtTime(iso: string): string {
	return iso.slice(11, 16);
}

/** "2025-03-14" (already a date) or ISO timestamp -> date part */
export function fmtDate(iso: string): string {
	return iso.slice(0, 10);
}

/** ISO timestamp -> "2026-07-06 14:22 UTC" */
export function fmtGeneratedAt(iso: string): string {
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return iso;
	const p = (n: number) => String(n).padStart(2, '0');
	return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UTC`;
}
