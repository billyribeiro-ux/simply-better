<script lang="ts">
	import type { Equity } from '$lib/types';
	import { fmtUsd } from '$lib/format';

	let { equity }: { equity: Equity } = $props();

	const W = 1000;
	const H = 380;
	const M = { top: 20, right: 92, bottom: 36, left: 64 };

	interface Geom {
		path: string;
		bands: { x0: number; x1: number }[];
		yTicks: { y: number; label: string }[];
		xTicks: { x: number; label: string }[];
		min: { x: number; y: number; label: string; show: boolean };
		max: { x: number; y: number; label: string; show: boolean };
		final: { x: number; y: number; label: string };
		title: string;
	}

	function niceStep(raw: number): number {
		const pow = Math.pow(10, Math.floor(Math.log10(raw)));
		for (const m of [1, 2, 2.5, 5, 10]) {
			if (m * pow >= raw) return m * pow;
		}
		return 10 * pow;
	}

	const geom: Geom | null = $derived.by(() => {
		const c = equity.curve;
		const dates = equity.dates;
		if (c.length < 2 || dates.length !== c.length - 1) return null;

		const innerW = W - M.left - M.right;
		const innerH = H - M.top - M.bottom;
		let lo = Math.min(...c);
		let hi = Math.max(...c);
		if (hi === lo) {
			lo -= 1;
			hi += 1;
		}
		const pad = (hi - lo) * 0.04;
		lo -= pad;
		hi += pad;

		const x = (i: number) => M.left + (i / (c.length - 1)) * innerW;
		const y = (v: number) => M.top + (1 - (v - lo) / (hi - lo)) * innerH;

		const path = c.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)} ${y(v).toFixed(2)}`).join(' ');

		// drawdown bands: contiguous stretches below the running peak
		const bands: { x0: number; x1: number }[] = [];
		let peak = c[0];
		let start = -1;
		for (let i = 1; i < c.length; i++) {
			if (c[i] < peak) {
				if (start === -1) start = i - 1;
			} else {
				if (start !== -1) {
					bands.push({ x0: x(start), x1: x(i) });
					start = -1;
				}
				peak = c[i];
			}
		}
		if (start !== -1) bands.push({ x0: x(start), x1: x(c.length - 1) });

		const step = niceStep((hi - lo) / 4);
		const yTicks: Geom['yTicks'] = [];
		for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
			yTicks.push({ y: y(v), label: fmtUsd(v) });
		}

		const xTicks: Geom['xTicks'] = [];
		const nLabels = Math.min(5, dates.length);
		for (let j = 0; j < nLabels; j++) {
			const di = Math.round((j * (dates.length - 1)) / Math.max(nLabels - 1, 1));
			xTicks.push({ x: x(di + 1), label: dates[di] });
		}

		let minI = 0;
		let maxI = 0;
		for (let i = 1; i < c.length; i++) {
			if (c[i] < c[minI]) minI = i;
			if (c[i] > c[maxI]) maxI = i;
		}

		return {
			path,
			bands,
			yTicks,
			xTicks,
			// annotations at the endpoints duplicate the axis/final labels — hide them
			min: { x: x(minI), y: y(c[minI]), label: fmtUsd(c[minI]), show: minI !== 0 },
			max: { x: x(maxI), y: y(c[maxI]), label: fmtUsd(c[maxI]), show: maxI !== c.length - 1 },
			final: { x: x(c.length - 1), y: y(c[c.length - 1]), label: fmtUsd(c[c.length - 1]) },
			title:
				`Equity curve from ${dates[0]} to ${dates[dates.length - 1]}: ` +
				`start ${fmtUsd(c[0])}, minimum ${fmtUsd(c[minI])}, ` +
				`maximum ${fmtUsd(c[maxI])}, final ${fmtUsd(c[c.length - 1])}.`
		};
	});
</script>

{#if geom}
	<figure>
		<svg viewBox="0 0 {W} {H}" role="img" preserveAspectRatio="xMidYMid meet">
			<title>{geom.title}</title>

			{#each geom.bands as band, i (i)}
				<rect
					x={band.x0}
					y={M.top}
					width={band.x1 - band.x0}
					height={H - M.top - M.bottom}
					class="dd"
				/>
			{/each}

			{#each geom.yTicks as tick (tick.label)}
				<line class="grid" x1={M.left} y1={tick.y} x2={W - M.right} y2={tick.y} />
				<text class="tick" x={M.left - 8} y={tick.y + 3} text-anchor="end">{tick.label}</text>
			{/each}

			<line class="axis" x1={M.left} y1={M.top} x2={M.left} y2={H - M.bottom} />
			<line class="axis" x1={M.left} y1={H - M.bottom} x2={W - M.right} y2={H - M.bottom} />

			{#each geom.xTicks as tick (tick.label)}
				<text class="tick" x={tick.x} y={H - 12} text-anchor="middle">{tick.label}</text>
			{/each}

			<path class="curve" d={geom.path} />

			{#if geom.max.show}
				<circle cx={geom.max.x} cy={geom.max.y} r="2.5" class="dot" />
				<text class="note" x={geom.max.x} y={geom.max.y - 8} text-anchor="middle">
					{geom.max.label}
				</text>
			{/if}

			{#if geom.min.show}
				<circle cx={geom.min.x} cy={geom.min.y} r="2.5" class="dot dot-loss" />
				<text class="note note-loss" x={geom.min.x} y={geom.min.y + 16} text-anchor="middle">
					{geom.min.label}
				</text>
			{/if}

			<circle cx={geom.final.x} cy={geom.final.y} r="2.5" class="dot" />
			<text class="note" x={geom.final.x + 8} y={geom.final.y + 3} text-anchor="start">
				{geom.final.label}
			</text>
		</svg>
	</figure>
{:else}
	<p class="empty num">No equity data in this run.</p>
{/if}

<style>
	figure {
		border: 1px solid var(--hairline);
		overflow-x: auto;
	}

	svg {
		display: block;
		width: 100%;
		min-width: 640px;
		height: auto;
	}

	.curve {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1.5;
		stroke-linejoin: round;
		stroke-linecap: round;
	}

	.dd {
		fill: var(--loss);
		opacity: 0.08;
	}

	.axis {
		stroke: var(--hairline);
		stroke-width: 1;
	}

	.grid {
		stroke: var(--hairline);
		stroke-width: 0.5;
	}

	.tick,
	.note {
		font-family: var(--font-mono);
		font-variant-numeric: tabular-nums;
		font-size: 10.5px;
		fill: var(--graphite);
	}

	.note {
		fill: var(--ink);
	}

	.note-loss {
		fill: var(--loss);
	}

	.dot {
		fill: var(--ink);
	}

	.dot-loss {
		fill: var(--loss);
	}

	.empty {
		color: var(--graphite);
		font-size: 13px;
		padding: 24px 0;
	}
</style>
