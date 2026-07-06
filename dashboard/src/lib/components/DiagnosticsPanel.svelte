<script lang="ts">
	import type { Diagnostics } from '$lib/types';
	import { fmtNum } from '$lib/format';

	let { diagnostics }: { diagnostics: Diagnostics } = $props();

	const notSeparating = $derived(
		diagnostics.auc_mean_test !== null && diagnostics.auc_mean_test <= 0.52
	);

	// AUC bars deviate from the 0.5 no-skill midline; 0.25 AUC = full half-track
	function dev(auc: number | null): number {
		if (auc === null) return 0;
		return Math.max(-1, Math.min(1, (auc - 0.5) / 0.25));
	}

	const probHist = $derived.by(() => {
		const agg = new Array<number>(20).fill(0);
		for (const f of diagnostics.folds) {
			f.prob_hist.forEach((c, i) => {
				if (i < 20) agg[i] += c;
			});
		}
		const max = Math.max(...agg, 1);
		return agg.map((c) => c / max);
	});
</script>

<div class="panel">
	<h2 class="eyebrow">Model diagnostics</h2>

	<div class="mean num">
		<span
			>AUC(test) mean {diagnostics.auc_mean_test === null
				? '—'
				: fmtNum(diagnostics.auc_mean_test, 3)}</span
		>
		{#if notSeparating}
			<span class="badge">Model not separating — features are the bottleneck.</span>
		{/if}
	</div>

	{#if diagnostics.folds.length}
		<ul class="folds num">
			{#each diagnostics.folds as f (f.fold)}
				<li>
					<span class="fold">{f.fold}</span>
					<span class="track" aria-hidden="true">
						{#if f.auc_val !== null}
							<span
								class="bar val"
								class:neg={f.auc_val < 0.5}
								style:width="{Math.abs(dev(f.auc_val)) * 50}%"
							></span>
						{/if}
						{#if f.auc_test !== null}
							<span
								class="bar test"
								class:neg={f.auc_test < 0.5}
								style:width="{Math.abs(dev(f.auc_test)) * 50}%"
							></span>
						{/if}
					</span>
					<span class="vals">
						{f.auc_test === null ? '—' : fmtNum(f.auc_test, 3)}
						<span class="dim">/ {f.auc_val === null ? '—' : fmtNum(f.auc_val, 3)}</span>
					</span>
					<span class="thr dim">thr {fmtNum(f.threshold, 3)} · n {f.n_test}</span>
				</li>
			{/each}
		</ul>
		<p class="legend num">bar deviates from AUC 0.5 · test <span class="swatch ink"></span> / val
			<span class="swatch graphite"></span></p>

		<h3 class="eyebrow sub">Test probability distribution</h3>
		<div class="barcode" role="img" aria-label="Aggregate histogram of test-fold probabilities across 20 bins from 0 to 1">
			{#each probHist as frac, i (i)}
				<span class="bin" style:height="{Math.max(frac * 100, 2)}%"></span>
			{/each}
		</div>
		<div class="axis num"><span>0.0</span><span>0.5</span><span>1.0</span></div>
	{:else}
		<p class="empty num">No fold diagnostics in this run.</p>
	{/if}
</div>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.mean {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px;
		font-size: 12.5px;
	}

	.badge {
		color: var(--ink);
		background: var(--wash);
		border: 1px solid var(--hairline);
		padding: 3px 10px;
		font-size: 11px;
		letter-spacing: 0.04em;
	}

	.folds {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 5px;
		font-size: 11px;
	}

	.folds li {
		display: grid;
		grid-template-columns: 58px 1fr 92px minmax(120px, auto);
		align-items: center;
		gap: 10px;
	}

	.fold {
		color: var(--graphite);
	}

	.track {
		position: relative;
		display: block;
		height: 10px;
		border-bottom: 1px solid var(--hairline);
	}

	.track::before {
		content: '';
		position: absolute;
		left: 50%;
		top: -2px;
		bottom: -2px;
		width: 1px;
		background: var(--hairline);
	}

	.bar {
		position: absolute;
		left: 50%;
		height: 4px;
		display: block;
	}

	.bar.neg {
		left: auto;
		right: 50%;
	}

	.bar.test {
		bottom: 0;
		background: var(--ink);
	}

	.bar.val {
		bottom: 5px;
		background: var(--graphite);
		opacity: 0.7;
	}

	.vals {
		text-align: right;
		white-space: nowrap;
	}

	.thr {
		white-space: nowrap;
	}

	.dim {
		color: var(--graphite);
	}

	.legend {
		font-size: 10.5px;
		color: var(--graphite);
		display: flex;
		align-items: center;
		gap: 5px;
	}

	.swatch {
		display: inline-block;
		width: 14px;
		height: 4px;
	}

	.swatch.ink {
		background: var(--ink);
	}

	.swatch.graphite {
		background: var(--graphite);
		opacity: 0.7;
	}

	.sub {
		margin-top: 4px;
	}

	.barcode {
		display: flex;
		align-items: flex-end;
		gap: 2px;
		height: 44px;
		border-bottom: 1px solid var(--hairline);
	}

	.bin {
		flex: 1;
		background: var(--ink);
		min-height: 1px;
	}

	.axis {
		display: flex;
		justify-content: space-between;
		font-size: 10px;
		color: var(--graphite);
	}

	.empty {
		color: var(--graphite);
		font-size: 12.5px;
	}

	@media (max-width: 560px) {
		.folds li {
			grid-template-columns: 52px 1fr 88px;
		}

		.thr {
			grid-column: 2 / -1;
			margin-top: -2px;
		}
	}
</style>
